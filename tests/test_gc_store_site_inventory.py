from __future__ import annotations

import contextlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/gc_store_site_inventory.py"

spec = importlib.util.spec_from_file_location("gc_store_site_inventory", SCRIPT_PATH)
scanner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = scanner
spec.loader.exec_module(scanner)


VALID_TYPES_RS = """
pub const GC_TYPE_ARRAY: u8 = 1;
pub const GC_TYPE_OBJECT: u8 = 2;
pub const GC_TYPE_MAX: u8 = GC_TYPE_OBJECT;

pub(super) static GC_TYPE_INFO_BY_ID: [Option<GcTypeInfo>; 3] = [
    None,
    Some(gc_type_info_entry(GC_TYPE_ARRAY, "array")),
    Some(gc_type_info_entry(GC_TYPE_OBJECT, "object")),
];
"""


@contextlib.contextmanager
def patched_scanner_root(root: Path, scan_paths: list[Path] | None = None):
    old_root = scanner.REPO_ROOT
    old_scan_paths = scanner.SCAN_PATHS
    scanner.REPO_ROOT = root
    if scan_paths is not None:
        scanner.SCAN_PATHS = scan_paths
    try:
        yield
    finally:
        scanner.REPO_ROOT = old_root
        scanner.SCAN_PATHS = old_scan_paths


def write_minimal_types(root: Path, text: str = VALID_TYPES_RS) -> None:
    types_path = root / "crates/perry-runtime/src/gc/types.rs"
    types_path.parent.mkdir(parents=True, exist_ok=True)
    types_path.write_text(text, encoding="utf-8")


def write_array_source(root: Path, name: str, text: str) -> Path:
    path = root / "crates/perry-runtime/src/array" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class GcStoreSiteInventoryTests(unittest.TestCase):
    def test_scanner_self_tests_pass(self):
        self.assertEqual(scanner.run_self_tests(), 0)

    def test_repo_gate_passes_and_scans_array_directory(self):
        roots = list(scanner.iter_scan_roots())
        self.assertIn(REPO_ROOT / "crates/perry-runtime/src/array/alloc.rs", roots)
        findings, files_scanned, marker_count = scanner.collect_inventory()
        self.assertFalse([finding.render() for finding in findings])
        self.assertGreater(files_scanned, 0)
        self.assertGreater(marker_count, 0)

    def test_gate_subprocess_succeeds(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--gate"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("zero unaudited sites", result.stdout)

    def test_invalid_annotation_class_and_empty_reason_are_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_array_source(
                root,
                "bad.rs",
                "\n".join(
                    [
                        "// GC_STORE_AUDIT(BOGUS): invalid class",
                        "// GC_STORE_AUDIT(BARRIERED):",
                        "pub unsafe fn f(elements: *mut f64, value: f64) {",
                        "    *elements.add(0) = value;",
                        "}",
                    ]
                ),
            )
            with patched_scanner_root(root):
                findings = scanner.scan_file(path)
        invalid = [finding for finding in findings if finding.error_class == "invalid_annotations"]
        self.assertEqual(len(invalid), 2)
        self.assertTrue(any("invalid GC_STORE_AUDIT class" in finding.reason for finding in invalid))
        self.assertTrue(any("empty reason" in finding.reason for finding in invalid))

    def test_stale_annotation_is_reported_without_nearby_store_or_helper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_array_source(
                root,
                "stale.rs",
                "// GC_STORE_AUDIT(BARRIERED): no store follows here\npub fn f() {}\n",
            )
            with patched_scanner_root(root):
                findings = scanner.scan_file(path)
        stale = [finding for finding in findings if finding.error_class == "stale_annotations"]
        self.assertEqual(len(stale), 1)

    def test_array_directory_path_is_gated_when_array_rs_is_absent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_types(root)
            write_array_source(
                root,
                "exposed.rs",
                "pub unsafe fn f(elements: *mut f64, value: f64) {\n    *elements.add(0) = value;\n}\n",
            )
            with patched_scanner_root(root, [Path("crates/perry-runtime/src/array")]):
                findings, files_scanned, _ = scanner.collect_inventory()
        self.assertEqual(files_scanned, 1)
        self.assertTrue(
            any(finding.error_class == "unaudited_store_sites" for finding in findings),
            [(finding.error_class, finding.reason, str(finding.path)) for finding in findings],
        )

    def test_helper_store_classification_accepts_barriered_helpers(self):
        runtime_path = REPO_ROOT / "crates/perry-runtime/src/array/example.rs"
        codegen_path = REPO_ROOT / "crates/perry-codegen/src/expr/example.rs"
        helper_cases = [
            (
                runtime_path,
                ["crate::array::store_array_slot(arr, i, bits);"],
                "array slot-store helper",
            ),
            (
                runtime_path,
                ["crate::gc::runtime_store_external_jsvalue_slot(parent, slot, bits);"],
                "external-slot store helper",
            ),
            (
                runtime_path,
                ["crate::gc::runtime_store_gc_heap_word_slot(parent, slot, bits);"],
                "GC heap-word store helper",
            ),
            (
                runtime_path,
                [
                    "crate::gc::layout_note_slot(parent, index, bits);",
                    "crate::gc::runtime_write_barrier_slot(parent, slot, bits);",
                ],
                "layout-note plus barrier helper pair",
            ),
            (
                codegen_path,
                ["emit_jsvalue_slot_store_on_block(blk, slot, value, parent, idx, true, parent, slot, true);"],
                "shared generated JSValue slot-store helper",
            ),
        ]
        for path, lines, expected in helper_cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, scanner.classify_store_or_helper(path, lines, 0))

    def test_gc_type_metadata_validation_reports_missing_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_types(
                root,
                """
pub const GC_TYPE_ARRAY: u8 = 1;
pub const GC_TYPE_OBJECT: u8 = 2;
pub const GC_TYPE_MAX: u8 = GC_TYPE_OBJECT;

pub(super) static GC_TYPE_INFO_BY_ID: [Option<GcTypeInfo>; 3] = [
    None,
    Some(gc_type_info_entry(GC_TYPE_ARRAY, "array")),
    None,
];
""",
            )
            with patched_scanner_root(root):
                findings = scanner.validate_gc_type_metadata()
        self.assertTrue(
            any("GC_TYPE_INFO_BY_ID[2]" in finding.reason for finding in findings),
            [(finding.error_class, finding.reason, str(finding.path)) for finding in findings],
        )


if __name__ == "__main__":
    unittest.main()
