"""Template parser implementations (Drain3 and identity)."""

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
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
