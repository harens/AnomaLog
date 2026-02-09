import hashlib
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from pathlib import Path

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from prefect.logging import get_run_logger

from anomalog.cache import (
    CachePathsConfig,
    asset_from_local_path,
    materialize,
)
from anomalog.io_utils import make_spinner_progress
from anomalog.template_parsers.templated_dataset import (
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
    def __init__(
        self,
        dataset_name: str,
        config_file: Path | None = None,
        cache_path: Path | None = None,
    ) -> None:
        if config_file is None:
            self.config_file = Path(f"{Path(__file__).parent}/drain3.ini")
        else:
            self.config_file = config_file

        self.dataset_name = dataset_name
        if cache_path is None:
            self.cache_path = (
                CachePathsConfig().cache_root / self.dataset_name / "drain3"
            )
        else:
            self.cache_path = cache_path

        self.cache_path.mkdir(parents=True, exist_ok=True)

        self.cfg_hash = hashlib.sha256(self.config_file.read_bytes()).hexdigest()[:12]

        self.inference_func: (
            Callable[[UntemplatedText], tuple[LogTemplate, ExtractedParameters]] | None
        ) = None

        self.cache_file_path = (
            self.cache_path / f"{self.dataset_name}_drain3_cache_{self.cfg_hash}.db"
        )

    def inference(
        self,
        unstructured_text: UntemplatedText,
    ) -> tuple[LogTemplate, ExtractedParameters]:
        if self.inference_func is None:
            msg = "Parser has not been trained yet"
            raise ValueError(msg)

        return self.inference_func(unstructured_text)

    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[UntemplatedText]],
    ) -> None:
        materialize(asset_from_local_path(self.cache_file_path))(self._train)(
            untemplated_text_iterator,
        )

    def _train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[UntemplatedText]],
    ) -> None:
        logger = get_run_logger()

        cache_file = FilePersistence(self.cache_file_path)  # ty:ignore[invalid-argument-type]

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
            task_id = progress.add_task("Parsing logs", total=None)
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

        def get_template_and_params_for_log(
            miner: TemplateMiner,
            log_line: str,
        ) -> tuple[str, Iterable[str]]:
            match = miner.match(log_line)
            if match is None:
                msg = f"Log line did not match any template: {log_line}"
                raise ValueError(msg)

            template = match.get_template()
            return template, miner.get_parameter_list(template, log_line)

        self.inference_func = partial(
            get_template_and_params_for_log,
            miner,
        )


class IdentityTemplateParser(TemplateParser):
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    def inference(
        self,
        unstructured_text: UntemplatedText,
    ) -> tuple[LogTemplate, ExtractedParameters]:
        return unstructured_text, []

    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[UntemplatedText]],
    ) -> None:
        # No training needed for the identity parser
        pass


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
