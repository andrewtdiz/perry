#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/gc_store_site_inventory.py --self-test
python3 -m unittest tests/test_gc_store_site_inventory.py
python3 scripts/gc_store_site_inventory.py --gate
