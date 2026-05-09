#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$SCRIPT_DIR"/*.sbatch; do
  [ -e "$script" ] || continue
  sbatch "$script"
done
