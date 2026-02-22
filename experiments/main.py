# run logreg on BGL
from pathlib import Path
from pprint import pprint

from anomalog.datasets import build_bgl_dataset
from anomalog.models.runner import ExperimentConfig, ExperimentRunner
from anomalog.models.sklearn_adapters import build_logreg_model


def main() -> None:
    dataset = build_bgl_dataset()

    experiment_runner = ExperimentRunner(
        ExperimentConfig(
            builders=[dataset.group_fixed_window(10)],
            models=[("logreg", build_logreg_model, None)],
            artifact_root=Path("artifacts"),
        ),
    )

    results = experiment_runner.run()
    for result in results:
        pprint(result)  # noqa: T203


if __name__ == "__main__":
    main()
