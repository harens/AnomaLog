import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from anomalog.datasets.dataset import Dataset
from anomalog.datasets.io_utils import make_spinner_progress

logger = logging.getLogger(__name__)


@runtime_checkable
class Parser(Protocol):
    dataset: Dataset

    def parse(self) -> None: ...

    def get_template_and_params_for_log(
        self, log_line: str
    ) -> tuple[str, list[str]]: ...


# TODO: Have a cache folder for anomalog
# Store the bits of whether a line is anomalous or not in there


# Note from https://github.com/logpai/logparser/blob/d9d4180784cde9afef990eeeb458591011933f9b/README.md
# Drain3 provides a good example for your reference that is
# built with practical enhancements for production scenarios.
# Whilst other toolkits only provide LogParser
class Drain3(Parser):
    def __init__(
        self,
        dataset: Dataset,
        cache_file: FilePersistence | None = None,
        config_file: Path | None = None,
    ) -> None:
        self.dataset = dataset

        if cache_file is None:
            cache_file = FilePersistence(f"{dataset.name}_drain3_cache.db")

        if config_file is None:
            config_file = Path(f"{Path(__file__).parent}/drain3.ini")

        config = TemplateMinerConfig()
        config.load(str(config_file))
        self._miner = TemplateMiner(cache_file, config=config)

    def parse(self) -> None:
        logger.info(f"Starting Drain3 parsing for {self.dataset.name} dataset")
        if self._miner is not None:
            logger.warning(
                "Parser has already been run. Re-running will reset the parser"
            )

        with make_spinner_progress() as progress:
            task_id = progress.add_task("Parsing logs", total=None)
            for i, log_line in enumerate(self.dataset.iter_lines()):
                result = self._miner.add_log_message(log_line)

                if i % 1000 == 0:
                    progress.advance(task_id, 1000)

        logger.info(
            f"Parsed {i + 1:,} log lines and mined {result['cluster_count']} templates."
        )

    def get_template_and_params_for_log(self, log_line: str) -> tuple[str, list[str]]:
        # preprocessed_line = self.dataset.preprocess(log_line).text
        if self._miner is None:
            self.parse()
        assert self._miner is not None  # for type checker
        match = self._miner.match(log_line)
        if match is None:
            raise ValueError(f"Log line did not match any template: {log_line}")

        template = match.get_template()
        return template, self._miner.get_parameter_list(template, log_line)
