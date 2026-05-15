"""Template parser implementations."""

import csv
import hashlib
import importlib
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import ClassVar

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from prefect.exceptions import MissingContextError
from prefect.logging import get_run_logger
from typing_extensions import override

from anomalog.cache import (
    CachePathsConfig,
    materialize,
)
from anomalog.io_utils import make_spinner_progress
from anomalog.parsers.template.dataset import (
    ExtractedParameters,
    LogTemplate,
    TemplateParser,
    UntemplatedText,
)


@contextmanager
def spellpy_logger_context(
    run_logger: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> Iterator[None]:
    """Forward `spellpy.spell` output into the active experiment logger.

    Raises:
        TypeError: If the resolved logger is not a standard `logging.Logger`.
    """
    experiment_logger: object = run_logger
    if isinstance(run_logger, logging.LoggerAdapter):
        experiment_logger = getattr(run_logger, "logger", None)
    if not isinstance(experiment_logger, logging.Logger):
        msg = "Active run logger is not a standard logging.Logger instance."
        raise TypeError(msg)
    spell_logger = logging.getLogger("spellpy.spell")
    attached_handlers = experiment_logger.handlers.copy()
    previous_level = spell_logger.level
    previous_propagate = spell_logger.propagate

    spell_logger.setLevel(
        (
            experiment_logger.level
            if experiment_logger.level != logging.NOTSET
            else logging.INFO
        ),
    )
    spell_logger.propagate = False
    for handler in attached_handlers:
        spell_logger.addHandler(handler)
    try:
        yield
    finally:
        for handler in attached_handlers:
            spell_logger.removeHandler(handler)
        spell_logger.setLevel(previous_level)
        spell_logger.propagate = previous_propagate


# Note from https://github.com/logpai/logparser/blob/d9d4180784cde9afef990eeeb458591011933f9b/README.md
# Drain3 provides a good example for your reference that is
# built with practical enhancements for production scenarios.
# Whilst other toolkits only provide LogParser
class Drain3Parser(TemplateParser):
    """Drain3-based template miner with Prefect asset caching.

    Instances accept an optional dataset name plus explicit config and cache
    paths so trained state can be persisted per dataset.

    Attributes:
        name (ClassVar[str]): Registry name for the built-in Drain3 parser.

    Args:
        dataset_name (str | None): Optional dataset name used to scope
            persisted Drain3 state.
        config_file (Path | None): Optional Drain3 config file override.
        cache_path (Path | None): Optional explicit cache directory override.
    """

    name: ClassVar[str] = "drain3"

    def __init__(
        self,
        dataset_name: str | None = None,
        config_file: Path | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self.config_file = (
            Path(f"{Path(__file__).parent}/drain3.ini")
            if config_file is None
            else config_file
        )
        self.dataset_name = dataset_name
        self.cache_path = cache_path
        self.cfg_hash = hashlib.sha256(self.config_file.read_bytes()).hexdigest()[:12]
        self.inference_func: (
            Callable[[UntemplatedText], tuple[LogTemplate, ExtractedParameters]] | None
        ) = None
        if self.dataset_name is not None:
            self.resolved_cache_path.mkdir(parents=True, exist_ok=True)

    @property
    def resolved_cache_path(self) -> Path:
        """Return the on-disk cache directory for this parser instance.

        Raises:
            ValueError: If the parser has not been bound to a dataset yet.
        """
        if self.dataset_name is None:
            msg = "Drain3Parser requires a dataset name before runtime use."
            raise ValueError(msg)
        if self.cache_path is not None:
            return self.cache_path
        return CachePathsConfig().cache_root / self.dataset_name / "drain3"

    @property
    def cache_file_path(self) -> Path:
        """Return the resolved cache file path for this parser instance.

        Raises:
            ValueError: If the parser has not been bound to a dataset yet.
        """
        if self.dataset_name is None:
            msg = "Drain3Parser requires a dataset name before runtime use."
            raise ValueError(msg)
        return (
            self.resolved_cache_path
            / f"{self.dataset_name}_drain3_cache_{self.cfg_hash}.db"
        )

    @staticmethod
    def _make_inference_func(
        miner: TemplateMiner,
    ) -> Callable[
        [UntemplatedText],
        tuple[LogTemplate, ExtractedParameters],
    ]:
        """Build an inference callable bound to a trained Drain3 miner.

        Args:
            miner (TemplateMiner): Trained Drain3 miner to bind into the
                returned callable.

        Returns:
            Callable[[UntemplatedText], tuple[LogTemplate, ExtractedParameters]]:
                Inference callable backed by the supplied miner.
        """

        def get_template_and_params_for_log(
            miner: TemplateMiner,
            log_line: UntemplatedText,
        ) -> tuple[LogTemplate, ExtractedParameters]:
            match = miner.match(log_line)
            if match is None:
                msg = f"Log line did not match any template: {log_line}"
                raise ValueError(msg)

            template = match.get_template()
            return template, miner.get_parameter_list(template, log_line)

        return partial(get_template_and_params_for_log, miner)

    def _load_inference_from_cache(self) -> None:
        """Initialise inference_func from the persisted Drain3 state.

        Prefect's asset caching can skip executing the training function on
        repeat runs. Without this hook, `self.inference_func` would remain
        unset even though a trained Drain3 cache exists on disk.

        Raises:
            ValueError: If no persisted Drain3 cache exists yet.
        """
        if not self.cache_file_path.exists():
            msg = "No trained Drain3 cache found. Please (re)train the parser first."
            raise ValueError(msg)

        cache_file = FilePersistence(str(self.cache_file_path))
        config = TemplateMinerConfig()
        config.load(str(self.config_file))

        miner = TemplateMiner(cache_file, config=config)
        self.inference_func = self._make_inference_func(miner)

    @override
    def inference(
        self,
        unstructured_text: UntemplatedText,
    ) -> tuple[LogTemplate, ExtractedParameters]:
        """Return template and parameters for a single unstructured log line.

        Args:
            unstructured_text (UntemplatedText): Raw untemplated log line to
                match against the trained miner.

        Returns:
            tuple[LogTemplate, ExtractedParameters]: Matched template and
                extracted parameter values.

        Raises:
            ValueError: If the parser has not been trained yet.
        """
        if self.inference_func is None:
            msg = "Parser has not been trained yet"
            raise ValueError(msg)

        return self.inference_func(unstructured_text)

    @override
    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[UntemplatedText]],
    ) -> None:
        """Train Drain3 on the dataset's untemplated message stream.

        Args:
            untemplated_text_iterator (Callable[[], Iterator[UntemplatedText]]):
                Zero-argument iterator factory over untemplated message text.
        """
        self.resolved_cache_path.mkdir(parents=True, exist_ok=True)

        # Avoid unstable cache keys from the iterator argument by
        # capturing it in a closure and running a zero-arg task
        # (no INPUTS component).
        # TODO: Handle this more elegantly with a custom CachePolicy
        # that ignores the iterator argument.
        def _run_train() -> None:
            return self._train(untemplated_text_iterator)

        materialize(self.cache_file_path)(_run_train)()

        # The training task might be skipped if the Prefect asset cache hits.
        # Ensure we still have a callable bound to the persisted miner state.
        if self.inference_func is None:
            self._load_inference_from_cache()

    def _train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[UntemplatedText]],
    ) -> None:
        logger = get_run_logger()

        cache_file = FilePersistence(str(self.cache_file_path))

        config = TemplateMinerConfig()
        config.load(str(self.config_file))
        miner = TemplateMiner(cache_file, config=config)

        if self.cache_file_path.exists():
            logger.info(
                "Stale cache file found at %s, deleting before training",
                self.cache_file_path,
            )
            self.cache_file_path.unlink()

        result = None
        with make_spinner_progress() as progress:
            task_id = progress.add_task("Mining logs", total=None)
            for i, log_line in enumerate(untemplated_text_iterator()):
                result = miner.add_log_message(log_line)

                # i+1 to stop overshoot at 0
                if (i + 1) % 1000 == 0:
                    progress.advance(task_id, 1000)

        if result is None:
            logger.warning("No logs were parsed during training")
        else:
            logger.info(
                "Parsed %d logs and mined %d templates",
                i,
                result.get("cluster_count", 0),
            )

        self.inference_func = self._make_inference_func(miner)


@dataclass(slots=True)
class IdentityTemplateParser(TemplateParser):
    """No-op template parser that returns the input string as its template.

    This parser is useful when experiments should operate on exact message text
    rather than mined abstractions, or when tests need deterministic,
    side-effect-free template inference.

    Attributes:
        name (ClassVar[str]): Registry/config name for the identity parser.
        dataset_name (str | None): Optional dataset identifier kept only for
            parity with the shared template-parser contract.
    """

    name: ClassVar[str] = "identity"
    dataset_name: str | None = None

    @override
    def inference(
        self,
        unstructured_text: UntemplatedText,
    ) -> tuple[LogTemplate, ExtractedParameters]:
        """Return the raw text as the template with no parameters.

        Args:
            unstructured_text (UntemplatedText): Raw log text to treat as its
                own template.

        Examples:
            >>> IdentityTemplateParser("demo").inference("hello")
            ('hello', [])

        Returns:
            tuple[LogTemplate, ExtractedParameters]: Raw text and an empty
                parameter list.
        """
        return unstructured_text, []

    @override
    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[UntemplatedText]],
    ) -> None:
        """Ignore the training stream because identity inference is stateless.

        Args:
            untemplated_text_iterator (Callable[[], Iterator[UntemplatedText]]):
                Iterator factory accepted for contract compatibility.
        """
        del untemplated_text_iterator
        # No training needed for the identity parser


@dataclass(slots=True)
class SpellTemplateParser(TemplateParser):
    """Spell-based template parser for DeepLog-style key extraction.

    This parser trains Spell on the provided text stream, then performs
    inference by matching lines against the mined templates. Matching is done
    in template-occurrence order so frequently observed templates win first.

    Attributes:
        name (ClassVar[str]): Registry/config name for the built-in parser.
        dataset_name (str | None): Optional dataset name used for cache paths.
        tau (float): Spell similarity threshold passed to Spell training.
    """

    name: ClassVar[str] = "spell"
    dataset_name: str | None = None
    tau: float = 0.5
    _patterns: list[tuple[str, re.Pattern[str], int]] | None = None

    @override
    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[UntemplatedText]],
    ) -> None:
        """Train Spell templates from the text stream.

        Args:
            untemplated_text_iterator (Callable[[], Iterator[UntemplatedText]]):
                Zero-argument iterator factory over untemplated message text.

        Raises:
            ModuleNotFoundError: If optional `spellpy` is not installed.
        """
        try:
            spell = importlib.import_module("spellpy").spell
        except ModuleNotFoundError as exc:
            msg = "Spell template parsing requires optional dependency 'spellpy'."
            raise ModuleNotFoundError(msg) from exc
        try:
            logger = get_run_logger()
        except MissingContextError:
            logger = logging.getLogger(__name__)

        log_dir = Path.cwd() / ".cache" / "spell"
        log_dir.mkdir(parents=True, exist_ok=True)
        dataset_prefix = self.dataset_name or "dataset"
        raw_path = log_dir / f"{dataset_prefix}_spell_input.log"
        outdir = log_dir / f"{dataset_prefix}_spell_output"
        outdir.mkdir(parents=True, exist_ok=True)

        any_lines = False
        with raw_path.open("w", encoding="utf-8") as handle:
            for log_line in untemplated_text_iterator():
                handle.write(log_line)
                handle.write("\n")
                any_lines = True
        if not any_lines:
            self._patterns = []
            return

        parser = spell.LogParser(
            indir=str(log_dir),
            outdir=str(outdir),
            log_format="<Content>",
            tau=self.tau,
            # AnomaLog only consumes the mined templates, so skip Spell's
            # per-row parameter extraction and the extra main-output append.
            keep_para=False,
        )
        with spellpy_logger_context(logger):
            parser.parse(raw_path.name)

        templates_path = outdir / f"{raw_path.name}_templates.csv"
        rows = _read_spell_template_rows(templates_path)
        self._patterns = [
            (
                template,
                _compile_spell_template_regex(template),
                occurrences,
            )
            for template, occurrences in rows
        ]

    @override
    def inference(
        self,
        unstructured_text: UntemplatedText,
    ) -> tuple[LogTemplate, ExtractedParameters]:
        """Infer template and extracted parameters for one log line.

        Args:
            unstructured_text (UntemplatedText): Raw line to match.

        Returns:
            tuple[LogTemplate, ExtractedParameters]: Matched template and
                captured parameters, or a self-template fallback when unmatched.

        Raises:
            ValueError: If the parser has not been trained yet.
        """
        if self._patterns is None:
            msg = "Parser has not been trained yet"
            raise ValueError(msg)

        for template, pattern, _ in self._patterns:
            match = pattern.fullmatch(unstructured_text)
            if match is None:
                continue
            return template, list(match.groups())
        return unstructured_text, []


def _read_spell_template_rows(path: Path) -> list[tuple[str, int]]:
    """Read Spell template rows in descending occurrence order.

    Args:
        path (Path): Spell templates CSV path.

    Returns:
        list[tuple[str, int]]: `(template, occurrences)` rows sorted by
            descending occurrence.
    """
    rows: list[tuple[str, int]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            template = row.get("EventTemplate")
            occurrences = row.get("Occurrences")
            if template is None or occurrences is None:
                continue
            rows.append((template, int(occurrences)))
    rows.sort(key=itemgetter(1), reverse=True)
    return rows


def _compile_spell_template_regex(template: str) -> re.Pattern[str]:
    """Compile a Spell template into a parameter-capturing regex.

    Args:
        template (str): Spell template containing zero or more `<*>` markers.

    Returns:
        re.Pattern[str]: Full-line regex for inference matching.
    """
    sentinel = "__ANOMALOG_SPELL_WILDCARD__"
    escaped = re.escape(template.replace("<*>", sentinel))
    pattern_text = escaped.replace(sentinel, "(.*?)")
    return re.compile(pattern_text)


# class LogParser(Parser):
#     valid_parsers = [
#         "AEL",
#         "Brain",
#         "Drain",
#         "IPLoM",
#         "LFA",
#         "LKE",
#         "LenMa",
#         "LogCluster",
#         "LogMine",
#         "LogSig",
#         "Logram",
#         "MoLFI",
#         "NuLog",
#         "SHISO",
#         "SLCT",
#         "Spell",
#         "ULP",
#         "logmatch",
#         "utils",
#     ]

#     def __init__(self, dataset: RawDataset, parser):
#         pass
