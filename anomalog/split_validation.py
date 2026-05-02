"""Shared validation helpers for chronological split fractions."""

from __future__ import annotations


def validate_split_fractions(*, train_frac: float, test_frac: float) -> None:
    """Validate a chronological train/test split pair.

    Args:
        train_frac (float): Requested train prefix fraction.
        test_frac (float): Requested fixed test suffix fraction.

    Raises:
        ValueError: If either fraction is out of range or the pair would
            over-allocate the sequence population.
    """
    if not 0.0 <= train_frac <= 1.0:
        msg = f"train_frac must be between 0 and 1 inclusive, got {train_frac}."
        raise ValueError(msg)
    if not 0.0 <= test_frac <= 1.0:
        msg = f"test_frac must be between 0 and 1 inclusive, got {test_frac}."
        raise ValueError(msg)
    if train_frac + test_frac > 1.0:
        msg = (
            "train_frac and test_frac must sum to no more than 1.0, "
            f"got train_frac={train_frac} and test_frac={test_frac}."
        )
        raise ValueError(msg)
