"""Thunderbird experiment config regression tests."""

from pathlib import Path

from anomalog.sequences import FixedWindowBasis
from experiments.config import FixedSequenceConfig, load_experiment_bundles


def test_thunderbird_manifest_uses_raw_position_fixed_windows() -> None:
    """Thunderbird fixed-window runs should declare raw-position semantics."""
    expected_window_size = 100
    repo_root = Path(__file__).resolve().parents[2]

    bundles = load_experiment_bundles(
        repo_root / "experiments" / "configs" / "datasets" / "thunderbird.toml",
    )

    assert bundles
    for bundle in bundles:
        sequence = bundle.dataset.sequence
        assert isinstance(sequence, FixedSequenceConfig)
        assert sequence.window_basis is FixedWindowBasis.RAW_POSITIONS
        assert sequence.window_alignment_offset == 0
        assert sequence.window_size == expected_window_size
        assert sequence.step == expected_window_size
