"""CLI entrypoint for the Thunderbird slice contract audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.audit.thunderbird_slice_audit import audit_thunderbird_slice


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Thunderbird slice audit CLI parser.

    Returns:
        argparse.ArgumentParser: Configured CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="Audit the Thunderbird slice contract from cached parquet data.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Override the AnomaLog cache root (defaults to the user cache).",
    )
    parser.add_argument(
        "--start-line-order",
        type=int,
        default=159_999_999,
        help="Inclusive raw line position at which the slice starts.",
    )
    parser.add_argument(
        "--end-line-order",
        type=int,
        default=169_999_998,
        help="Inclusive raw line position at which the slice ends.",
    )
    return parser


def main() -> int:
    """Run the Thunderbird slice audit and print a compact JSON summary.

    Returns:
        int: Process exit status.
    """
    args = build_arg_parser().parse_args()
    payload = audit_thunderbird_slice(
        cache_root=args.cache_root,
        start_line_order=args.start_line_order,
        end_line_order=args.end_line_order,
    )
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover
