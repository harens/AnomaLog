import hashlib
import logging
from functools import partial
from pathlib import Path

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from anomalog.datasets.dataset import ParsedDataset, Parser, RawDataset
from anomalog.datasets.io_utils import make_spinner_progress

logger = logging.getLogger(__name__)

# TODO: Have a cache folder for anomalog
# Store the bits of whether a line is anomalous or not in there


# Note from https://github.com/logpai/logparser/blob/d9d4180784cde9afef990eeeb458591011933f9b/README.md
# Drain3 provides a good example for your reference that is
# built with practical enhancements for production scenarios.
# Whilst other toolkits only provide LogParser
class Drain3Parser(Parser):
    def __init__(
        self,
        config_file: Path | None = None,
    ) -> None:
        if config_file is None:
            self.config_file = Path(f"{Path(__file__).parent}/drain3.ini")
        else:
            self.config_file = config_file

        self.cfg_hash = hashlib.sha256(self.config_file.read_bytes()).hexdigest()[:12]

    # TODO: Handle cleanup of unfinished cache file during crash, so that we don't reuse
    def parse(
        self, raw_dataset: RawDataset, cache_file_path: Path | None = None
    ) -> ParsedDataset:
        if cache_file_path is None:
            stat = raw_dataset.raw_logs_path.stat()
            file_sig = f"{stat.st_size}_{stat.st_mtime_ns}"
            cache_file_path = Path(
                f"{raw_dataset.name}_drain3_cache_{file_sig}_{self.cfg_hash}.db"
            )

        cache_file = FilePersistence(cache_file_path)

        config = TemplateMinerConfig()
        config.load(str(self.config_file))
        miner = TemplateMiner(cache_file, config=config)

        if cache_file_path.exists():
            logger.info(f"Loaded existing Drain3 state from {cache_file_path}")
        else:
            logger.info("No existing Drain3 state found, starting fresh")

            with make_spinner_progress() as progress:
                task_id = progress.add_task("Parsing logs", total=None)
                for i, log_line in enumerate(raw_dataset.iter_lines()):
                    result = miner.add_log_message(log_line)

                    # i+1 to stop overshoot at 0
                    if (i + 1) % 1000 == 0:
                        progress.advance(task_id, 1000)

            logger.info(
                f"Parsed {i + 1:,} logs and mined {result['cluster_count']} templates"
            )

        def get_template_and_params_for_log(
            miner: TemplateMiner, log_line: str
        ) -> tuple[str, list[str]]:
            # preprocessed_line = self.dataset.preprocess(log_line).text
            match = miner.match(log_line)
            if match is None:
                raise ValueError(f"Log line did not match any template: {log_line}")

            template = match.get_template()
            return template, miner.get_parameter_list(template, log_line)

        return ParsedDataset(
            **raw_dataset.base_kwargs(),
            get_template_and_params_for_log=partial(
                get_template_and_params_for_log,
                miner,
            ),
        )


class IdentityParser(Parser):
    def parse(self, raw_dataset: RawDataset) -> ParsedDataset:
        logger.info(f"IdentityParser: Skipping parsing for {raw_dataset.name} dataset")
        return ParsedDataset(
            **raw_dataset.base_kwargs(),
            get_template_and_params_for_log=lambda log_line: (log_line, []),
        )


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
