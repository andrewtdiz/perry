#!/usr/bin/env python3
"""Audit raw GC-relevant store sites.

The generational collector relies on every raw heap/slot write being either
barriered, rooted, initialization-only, pointer-free, or stack-local. This
script scans the first-party paths where raw GC-relevant stores are expected
and requires a nearby `GC_STORE_AUDIT(...)` marker with a reason.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

AUDIT_CLASSES = {
    "BARRIERED",
    "EXTERNAL_BARRIERED",
    "ROOT",
    "INIT",
    "POINTER_FREE",
    "STACK",
}

MARKER_RE = re.compile(
    r"GC_STORE_AUDIT\((" + "|".join(sorted(AUDIT_CLASSES)) + r")\):\s*\S"
)
ANY_MARKER_RE = re.compile(r"GC_STORE_AUDIT\((?P<audit_class>[^)\n]*)\)(?P<suffix>[^\n]*)")

CODEGEN_DEST_RE = re.compile(
    r"\.store\([^,]+,\s*[^,]+,\s*&?(?P<dest>[A-Za-z_][A-Za-z0-9_]*)\)"
)
CODEGEN_EMIT_RAW_STORE_RE = re.compile(r'emit_raw\(format!\("store\b')

RUST_FIELD_STORE_RE = re.compile(
    r"\(\*[^)\n]+\)\.(?P<field>keys_array|entries|elements)\s*="
)
RUST_PROMISE_FIELD_STORE_RE = re.compile(
    r"\(\*[^)\n]+\)\.(?P<field>on_fulfilled|on_rejected|next)\s*="
)
RUST_POINTER_FIELD_STORE_RE = re.compile(
    r"\b(?P<owner>[A-Za-z_][A-Za-z0-9_]*)\.(?P<field>string_ptr)\s*="
)
RUST_GLOBAL_INDEX_STORE_RE = re.compile(
    r"\b(?P<target>[A-Z][A-Z0-9_]*)\s*\[[^\]]+\]\s*="
)
RUST_TLS_INDEX_STORE_RE = re.compile(
    r"\(\*[A-Za-z_][A-Za-z0-9_]*\.get\(\)\)\[[^\]]+\]\s*="
)
RUST_DEREF_INDEX_STORE_RE = re.compile(
    r"\(\*[A-Za-z_][A-Za-z0-9_]*\)\[[^\]]+\]\s*="
)

RUST_PTR_STORE_RE = re.compile(r"\b(?:std::)?ptr::write(?:_unaligned)?\s*\(")
RUST_COPY_RE = re.compile(r"\b(?:std::)?ptr::copy(?:_nonoverlapping)?\s*\(")
RUST_DEREF_ASSIGN_RE = re.compile(
    r"\*(?P<target>[A-Za-z_][A-Za-z0-9_]*)(?:\.add\([^)]*\))?\s*=(?!=)"
)
RUST_ATOMIC_STORE_RE = re.compile(
    r"\b(?P<target>[A-Za-z_][A-Za-z0-9_]*)\.store\s*\(\s*(?P<value>[^,]+)"
)
RUST_ATOMIC_COMPARE_EXCHANGE_RE = re.compile(
    r"\b(?P<target>[A-Za-z_][A-Za-z0-9_]*)\.compare_exchange\s*\(\s*[^,]+,\s*(?P<value>[^,]+)"
)
RUST_ARRAY_STORE_HELPER_RE = re.compile(r"\b(?:crate::array::)?store_array_slot\s*\(")
RUST_OBJECT_STORE_HELPER_RE = re.compile(
    r"\b(?:crate::object::)?(?:store_object_field_slot|store_thread_object_field)\s*\("
)
RUST_THREAD_ARRAY_STORE_HELPER_RE = re.compile(r"\bstore_thread_array_slot\s*\(")
RUST_ARRAY_LAYOUT_HELPER_RE = re.compile(
    r"\b(?:crate::array::)?(?:note_array_slot|rebuild_array_layout(?:_exact)?|replay_array_growth_write_barriers)\s*\("
)
RUST_RUNTIME_JSVALUE_STORE_HELPER_RE = re.compile(
    r"\b(?:crate::gc::)?runtime_store_(?:gc_)?jsvalue_slot\s*\("
)
RUST_EXTERNAL_STORE_HELPER_RE = re.compile(
    r"\b(?:crate::gc::)?runtime_store_external_(?:jsvalue|heap_word)_slot(?:_with_layout)?\s*\("
)
RUST_GC_HEAP_WORD_STORE_HELPER_RE = re.compile(
    r"\b(?:crate::gc::)?runtime_store_gc_heap_word_slot\s*\("
)
RUST_LAYOUT_NOTE_HELPER_RE = re.compile(r"\b(?:crate::gc::)?layout_note_slot\s*\(")
RUST_RUNTIME_BARRIER_HELPER_RE = re.compile(
    r"\b(?:crate::gc::)?runtime_write_barrier_(?:slot|external_slot|gc_slot)\s*\("
)
CODEGEN_JSVALUE_SLOT_STORE_HELPER_RE = re.compile(r"\bemit_jsvalue_slot_store_on_block\s*\(")
HELPER_DEFINITION_RE = re.compile(
    r"\bfn\s+(?:store_array_slot|note_array_slot|rebuild_array_layout|rebuild_array_layout_exact|"
    r"replay_array_growth_write_barriers|runtime_store_(?:gc_)?jsvalue_slot|"
    r"runtime_store_external_(?:jsvalue|heap_word)_slot(?:_with_layout)?|"
    r"runtime_store_gc_heap_word_slot|runtime_write_barrier_(?:slot|external_slot|gc_slot)|"
    r"store_object_field_slot|store_thread_array_slot|store_thread_object_field|"
    r"emit_jsvalue_slot_store_on_block)\s*\("
)


SCAN_PATHS = [
    Path("crates/perry-codegen/src/expr"),
    Path("crates/perry-runtime/src/array"),
    Path("crates/perry-runtime/src/object"),
    Path("crates/perry-runtime/src/closure.rs"),
    Path("crates/perry-runtime/src/json.rs"),
    Path("crates/perry-runtime/src/regex.rs"),
    Path("crates/perry-runtime/src/plugin.rs"),
    Path("crates/perry-runtime/src/thread.rs"),
    Path("crates/perry-runtime/src/promise"),
    Path("crates/perry-runtime/src/map.rs"),
    Path("crates/perry-runtime/src/set.rs"),
    Path("crates/perry-runtime/src/string.rs"),
    Path("crates/perry-runtime/src/typedarray.rs"),
    Path("crates/perry-runtime/src/buffer.rs"),
    Path("crates/perry-stdlib/src"),
]


CODEGEN_HEAP_DEST_HINTS = (
    "arr_header_addr",
    "arr_ptr",
    "byte_ptr",
    "elem_ptr",
    "element_addr",
    "element_ptr",
    "field_addr",
    "field_ptr",
    "g_ref",
    "offset_field_ptr",
    "raw",
    "slot_ptr",
    "storage",
)

RUST_COPY_RISK_HINTS = (
    "arr_elements",
    "dst,",
    "dst)",
    "dst.add",
    "dst_data",
    "dst_elements",
    "elements.add",
    "elements_ptr",
    "fields_ptr",
    "new_ptr",
    "pair_elems",
    "result_elems",
    "rewritten_captures",
    "src_elements",
)

RUST_POINTER_FREE_COPY_HINTS = (
    "body",
    "buf_data",
    "buffer_data",
    "bytes",
    "data_ptr",
    "hash",
    "key_bytes",
    "last_char",
    "part.as_ptr",
    "property_name",
    "source_data",
    "str_bytes",
)

STACK_COPY_HINTS = (
    "heap_buf.as_mut_ptr",
    "regular_args",
    "spread_data",
    "stack_buf.as_mut_ptr",
)

RUST_DEREF_RISK_TARGETS = (
    "arr_data",
    "captures_ptr",
    "dst",
    "dst_captures",
    "dst_data",
    "dst_elements",
    "dst_fields",
    "elements",
    "elements_ptr",
    "fields",
    "new_keys_elements",
    "pair_elems",
    "result_elements",
)

RUST_ATOMIC_ROOT_TARGET_HINTS = (
    "CACHE",
    "CACHED",
    "GLOBAL",
    "ROOT",
    "SINGLETON",
    "PTR",
)

RUST_ATOMIC_ROOT_VALUE_HINTS = (
    "addr",
    "bits",
    "new_ptr",
    "ptr",
    "to_bits",
    "value",
)

RUST_GLOBAL_INDEX_RISK_TARGET_HINTS = (
    "CACHE",
    "GLOBAL",
    "ROOT",
    "TABLE",
)

RUST_GLOBAL_INDEX_RISK_EXACT_TARGETS = {
    "INTERN_TABLE",
    "SMALL_INT_CACHE",
    "TRANSITION_CACHE_GLOBAL",
}

RUST_GLOBAL_INDEX_POINTER_HINTS = (
    "key_ptr",
    "keys_array",
    "next_keys",
    "old_entry",
    "ptr",
    "string_ptr",
)


@dataclass(frozen=True)
class Finding:
    path: Path
    line_no: int
    text: str
    reason: str
    error_class: str = "unaudited_store_sites"

    def render(self) -> str:
        rel = self.path.relative_to(REPO_ROOT)
        return f"{rel}:{self.line_no}: {self.reason}: {self.text.strip()}"


@dataclass(frozen=True)
class AuditMarker:
    path: Path
    line_no: int
    text: str
    audit_class: str
    reason: str


def iter_scan_roots() -> Iterable[Path]:
    for rel in SCAN_PATHS:
        root = REPO_ROOT / rel
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(root.rglob("*.rs"))

    for ext_dir in sorted((REPO_ROOT / "crates").glob("perry-ext-*")):
        src = ext_dir / "src"
        if src.is_dir():
            yield from sorted(src.rglob("*.rs"))


def is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("//") or stripped.startswith("///")


def call_window(lines: list[str], index: int) -> str:
    """Return a small multiline window for classifying split calls."""

    start = index
    end = min(len(lines), index + 6)
    return " ".join(line.strip() for line in lines[start:end])


def nearby_window(lines: list[str], index: int, radius: int = 6) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return " ".join(line.strip() for line in lines[start:end])


def marker_is_valid(match: re.Match[str]) -> bool:
    audit_class = match.group("audit_class").strip()
    suffix = match.group("suffix")
    if audit_class not in AUDIT_CLASSES:
        return False
    if not suffix.lstrip().startswith(":"):
        return False
    reason = suffix.lstrip()[1:].strip()
    return bool(reason)


def parse_audit_markers(path: Path, lines: list[str]) -> tuple[list[AuditMarker], list[Finding]]:
    markers: list[AuditMarker] = []
    invalid: list[Finding] = []
    for index, line in enumerate(lines):
        for match in ANY_MARKER_RE.finditer(line):
            audit_class = match.group("audit_class").strip()
            suffix = match.group("suffix")
            reason_text = suffix.lstrip()[1:].strip() if suffix.lstrip().startswith(":") else ""
            if audit_class not in AUDIT_CLASSES:
                invalid.append(
                    Finding(
                        path,
                        index + 1,
                        line,
                        f"invalid GC_STORE_AUDIT class {audit_class!r}",
                        "invalid_annotations",
                    )
                )
                continue
            if not suffix.lstrip().startswith(":") or not reason_text:
                invalid.append(
                    Finding(
                        path,
                        index + 1,
                        line,
                        "GC_STORE_AUDIT marker has an empty reason",
                        "invalid_annotations",
                    )
                )
                continue
            markers.append(AuditMarker(path, index + 1, line, audit_class, reason_text))
    return markers, invalid


def has_nearby_marker(lines: list[str], index: int) -> bool:
    start = max(0, index - 6)
    end = min(len(lines), index + 7)
    return any(
        any(marker_is_valid(match) for match in ANY_MARKER_RE.finditer(lines[i]))
        for i in range(start, end)
    )


def helper_definition_line(line: str) -> bool:
    return bool(HELPER_DEFINITION_RE.search(line))


def is_runtime_array_path(path: Path) -> bool:
    posix = path.as_posix()
    return posix.endswith("crates/perry-runtime/src/array.rs") or (
        "crates/perry-runtime/src/array/" in posix
    )


def is_risky_codegen_store(line: str) -> bool:
    if CODEGEN_EMIT_RAW_STORE_RE.search(line):
        return True
    match = CODEGEN_DEST_RE.search(line)
    if not match:
        return False
    dest = match.group("dest")
    return dest in CODEGEN_HEAP_DEST_HINTS


def classify_codegen_store_or_helper(line: str) -> str | None:
    if is_risky_codegen_store(line):
        return "raw generated heap/global store"
    if not helper_definition_line(line) and CODEGEN_JSVALUE_SLOT_STORE_HELPER_RE.search(line):
        return "shared generated JSValue slot-store helper"
    return None


def classify_rust_helper(path: Path, lines: list[str], index: int) -> str | None:
    line = lines[index]
    if helper_definition_line(line):
        return None
    window = nearby_window(lines, index)
    atomic_store = RUST_ATOMIC_STORE_RE.search(call_window(lines, index))
    if atomic_store and any(
        hint in atomic_store.group("target").upper() for hint in RUST_ATOMIC_ROOT_TARGET_HINTS
    ):
        return "root/cache atomic store helper"
    if RUST_ARRAY_STORE_HELPER_RE.search(line) or RUST_THREAD_ARRAY_STORE_HELPER_RE.search(line):
        return "barriered array slot-store helper"
    if RUST_OBJECT_STORE_HELPER_RE.search(line):
        return "barriered object slot-store helper"
    if RUST_ARRAY_LAYOUT_HELPER_RE.search(line):
        return "array layout/barrier helper"
    if RUST_EXTERNAL_STORE_HELPER_RE.search(line):
        return "barriered external-slot store helper"
    if RUST_GC_HEAP_WORD_STORE_HELPER_RE.search(line):
        return "barriered GC heap-word store helper"
    if RUST_RUNTIME_JSVALUE_STORE_HELPER_RE.search(line):
        return "barriered runtime JSValue slot-store helper"
    if RUST_LAYOUT_NOTE_HELPER_RE.search(line) and RUST_RUNTIME_BARRIER_HELPER_RE.search(window):
        return "layout-note plus barrier helper pair"
    if RUST_RUNTIME_BARRIER_HELPER_RE.search(line) and RUST_LAYOUT_NOTE_HELPER_RE.search(window):
        return "layout-note plus barrier helper pair"
    return None


def classify_rust_store(path: Path, lines: list[str], index: int) -> str | None:
    line = lines[index]
    window = call_window(lines, index)
    atomic_store = RUST_ATOMIC_STORE_RE.search(window)
    if atomic_store and is_risky_atomic_root_store(
        atomic_store.group("target"), atomic_store.group("value")
    ):
        return "raw atomic cache/global pointer store"

    atomic_cas = RUST_ATOMIC_COMPARE_EXCHANGE_RE.search(window)
    if atomic_cas and is_risky_atomic_root_store(
        atomic_cas.group("target"), atomic_cas.group("value")
    ):
        return "raw atomic cache/global pointer CAS"

    global_index = RUST_GLOBAL_INDEX_STORE_RE.search(line)
    if global_index and is_risky_global_index_store(global_index.group("target"), window):
        return "raw cache/global pointer table store"

    if RUST_TLS_INDEX_STORE_RE.search(line) and is_risky_tls_index_store(window):
        return "raw TLS cache pointer table store"

    if (
        RUST_DEREF_INDEX_STORE_RE.search(line)
        and ("TransitionEntry" in window or "TRANSITION_CACHE" in window)
        and is_risky_global_index_store("TRANSITION_CACHE_GLOBAL", window)
    ):
        return "raw cache/global pointer table store"

    if "crates/perry-runtime/src/promise/" in path.as_posix():
        if RUST_PROMISE_FIELD_STORE_RE.search(line):
            return "raw Promise heap pointer field store"

    pointer_field = RUST_POINTER_FIELD_STORE_RE.search(line)
    if pointer_field:
        return "raw cache/global pointer field store"

    deref = RUST_DEREF_ASSIGN_RE.search(line)
    if deref and any(hint in deref.group("target") for hint in RUST_DEREF_RISK_TARGETS):
        if path.name in {"buffer.rs", "typedarray.rs"}:
            return None
        return "raw direct slot assignment"

    if RUST_FIELD_STORE_RE.search(line):
        return "raw heap pointer field store"

    if RUST_PTR_STORE_RE.search(line):
        if any(hint in window for hint in STACK_COPY_HINTS):
            return "raw stack/temporary argument store"
        return "raw slot write"

    if RUST_COPY_RE.search(line):
        if any(hint in window for hint in STACK_COPY_HINTS):
            return "raw stack/temporary argument copy"
        if path.name in {"string.rs", "buffer.rs", "typedarray.rs"}:
            return None
        if any(hint in window for hint in RUST_POINTER_FREE_COPY_HINTS):
            return None
        if any(hint in window for hint in RUST_COPY_RISK_HINTS):
            return "raw slot copy"
        if is_runtime_array_path(path):
            return "raw array slot copy"
    return None


def classify_rust_store_or_helper(path: Path, lines: list[str], index: int) -> str | None:
    return classify_rust_store(path, lines, index) or classify_rust_helper(path, lines, index)


def classify_store_or_helper(path: Path, lines: list[str], index: int) -> str | None:
    if "crates/perry-codegen/src/expr" in path.as_posix():
        return classify_codegen_store_or_helper(lines[index])
    return classify_rust_store_or_helper(path, lines, index)


def has_nearby_recognized_store_or_helper(path: Path, lines: list[str], index: int) -> bool:
    start = max(0, index - 6)
    end = min(len(lines), index + 7)
    return any(
        not is_comment_or_blank(lines[i]) and classify_store_or_helper(path, lines, i)
        for i in range(start, end)
    )


def is_risky_atomic_root_store(target: str, value: str) -> bool:
    target_upper = target.upper()
    value_lower = value.lower()
    if not any(hint in target_upper for hint in RUST_ATOMIC_ROOT_TARGET_HINTS):
        return False
    return any(hint in value_lower for hint in RUST_ATOMIC_ROOT_VALUE_HINTS)


def is_risky_global_index_store(target: str, window: str) -> bool:
    target_upper = target.upper()
    if target_upper not in RUST_GLOBAL_INDEX_RISK_EXACT_TARGETS and not any(
        hint in target_upper for hint in RUST_GLOBAL_INDEX_RISK_TARGET_HINTS
    ):
        return False
    window_lower = window.lower()
    return target_upper in RUST_GLOBAL_INDEX_RISK_EXACT_TARGETS or any(
        hint in window_lower for hint in RUST_GLOBAL_INDEX_POINTER_HINTS
    )


def is_risky_tls_index_store(window: str) -> bool:
    window_lower = window.lower()
    return "cache" in window_lower and any(
        hint in window_lower for hint in RUST_GLOBAL_INDEX_POINTER_HINTS
    )


def strip_rust_line_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def extract_gc_type_info_array(text: str) -> tuple[str, int] | None:
    match = re.search(r"GC_TYPE_INFO_BY_ID\s*:\s*\[[^\]]+\]\s*=\s*\[", text)
    if not match:
        return None
    start = match.end()
    depth = 1
    index = start
    in_string = False
    escaped = False
    while index < len(text):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start:index], text[:start].count("\n") + 1
        index += 1
    return None


def split_top_level_entries(body: str, start_line: int) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    depth = 0
    in_string = False
    escaped = False
    entry_start = 0
    entry_line = start_line
    line_no = start_line
    for index, ch in enumerate(body):
        if ch == "\n":
            line_no += 1
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            entry = body[entry_start:index]
            entries.append((entry, entry_line))
            entry_start = index + 1
            entry_line = line_no
            while entry_start < len(body) and body[entry_start] in " \t\r\n":
                if body[entry_start] == "\n":
                    entry_line += 1
                entry_start += 1
    tail = body[entry_start:]
    if tail.strip():
        entries.append((tail, entry_line))
    return entries


def validate_gc_type_metadata() -> list[Finding]:
    path = REPO_ROOT / "crates/perry-runtime/src/gc/types.rs"
    if not path.exists():
        return [
            Finding(
                path,
                1,
                "",
                "GC type metadata file is missing",
                "missing_gc_type_metadata",
            )
        ]

    text = path.read_text(encoding="utf-8", errors="replace")
    text_no_comments = strip_rust_line_comments(text)
    raw_constants = {
        name: value
        for name, value in re.findall(
            r"\bpub\s+const\s+(GC_TYPE_[A-Z0-9_]+)\s*:\s*u8\s*=\s*([A-Za-z0-9_]+)\s*;",
            text_no_comments,
        )
    }
    findings: list[Finding] = []
    if "GC_TYPE_MAX" not in raw_constants:
        return [
            Finding(
                path,
                1,
                "",
                "GC_TYPE_MAX is not declared",
                "missing_gc_type_metadata",
            )
        ]

    def resolve_constant(name: str, seen: set[str] | None = None) -> int | None:
        seen = set() if seen is None else seen
        if name in seen:
            return None
        seen.add(name)
        value = raw_constants.get(name)
        if value is None:
            return None
        if value.isdigit():
            return int(value)
        return resolve_constant(value, seen)

    max_value = resolve_constant("GC_TYPE_MAX")
    if max_value is None:
        findings.append(
            Finding(
                path,
                1,
                raw_constants["GC_TYPE_MAX"],
                "GC_TYPE_MAX does not resolve to a concrete GC_TYPE value",
                "missing_gc_type_metadata",
            )
        )
        return findings

    declared_by_value: dict[int, list[str]] = collections.defaultdict(list)
    constant_line: dict[str, int] = {}
    for match in re.finditer(
        r"\bpub\s+const\s+(GC_TYPE_[A-Z0-9_]+)\s*:\s*u8\s*=\s*([A-Za-z0-9_]+)\s*;",
        text_no_comments,
    ):
        name = match.group(1)
        if name == "GC_TYPE_MAX":
            continue
        value = resolve_constant(name)
        if value is None:
            findings.append(
                Finding(
                    path,
                    text_no_comments[: match.start()].count("\n") + 1,
                    match.group(0),
                    f"{name} does not resolve to a concrete value",
                    "missing_gc_type_metadata",
                )
            )
            continue
        declared_by_value[value].append(name)
        constant_line[name] = text_no_comments[: match.start()].count("\n") + 1

    for value, names in sorted(declared_by_value.items()):
        if value < 1 or value > max_value:
            findings.append(
                Finding(
                    path,
                    constant_line.get(names[0], 1),
                    ", ".join(names),
                    f"declared GC type id {value} is outside 1..=GC_TYPE_MAX ({max_value})",
                    "missing_gc_type_metadata",
                )
            )

    for value in range(1, max_value + 1):
        names = declared_by_value.get(value, [])
        if not names:
            findings.append(
                Finding(
                    path,
                    1,
                    f"GC_TYPE_{value}",
                    f"no GC_TYPE_* constant declared for type id {value}",
                    "missing_gc_type_metadata",
                )
            )
        elif len(names) > 1:
            findings.append(
                Finding(
                    path,
                    constant_line.get(names[1], 1),
                    ", ".join(names),
                    f"multiple GC_TYPE_* constants declare type id {value}",
                    "missing_gc_type_metadata",
                )
            )

    array = extract_gc_type_info_array(text_no_comments)
    if array is None:
        findings.append(
            Finding(
                path,
                1,
                "GC_TYPE_INFO_BY_ID",
                "GC_TYPE_INFO_BY_ID table is missing",
                "missing_gc_type_metadata",
            )
        )
        return findings

    body, array_start_line = array
    entries = split_top_level_entries(body, array_start_line)
    if not entries or entries[0][0].strip() != "None":
        findings.append(
            Finding(
                path,
                entries[0][1] if entries else array_start_line,
                entries[0][0] if entries else "",
                "GC_TYPE_INFO_BY_ID index 0 must remain None",
                "missing_gc_type_metadata",
            )
        )

    entry_counts: collections.Counter[str] = collections.Counter(
        match.group(1)
        for match in re.finditer(r"Some\s*\(\s*gc_type_info_entry\s*\(\s*(GC_TYPE_[A-Z0-9_]+)", body)
    )
    expected_by_index = {
        value: names[0]
        for value, names in declared_by_value.items()
        if 1 <= value <= max_value and len(names) == 1
    }

    for value in range(1, max_value + 1):
        expected = expected_by_index.get(value)
        if expected is None:
            continue
        entry_text = entries[value][0] if value < len(entries) else ""
        entry_line = entries[value][1] if value < len(entries) else array_start_line
        entry_match = re.search(
            r"Some\s*\(\s*gc_type_info_entry\s*\(\s*(GC_TYPE_[A-Z0-9_]+)", entry_text
        )
        if not entry_match or entry_match.group(1) != expected:
            findings.append(
                Finding(
                    path,
                    entry_line,
                    entry_text.strip().splitlines()[0] if entry_text.strip() else "",
                    f"GC_TYPE_INFO_BY_ID[{value}] must be Some(gc_type_info_entry({expected}, ...))",
                    "missing_gc_type_metadata",
                )
            )
        count = entry_counts[expected]
        if count != 1:
            findings.append(
                Finding(
                    path,
                    constant_line.get(expected, 1),
                    expected,
                    f"{expected} metadata entry count is {count}, expected exactly 1",
                    "missing_gc_type_metadata",
                )
            )

    return findings


def scan_file(path: Path) -> list[Finding]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    valid_markers, invalid_markers = parse_audit_markers(path, lines)
    findings: list[Finding] = list(invalid_markers)
    for marker in valid_markers:
        if not has_nearby_recognized_store_or_helper(path, lines, marker.line_no - 1):
            findings.append(
                Finding(
                    path,
                    marker.line_no,
                    marker.text,
                    "stale GC_STORE_AUDIT marker has no nearby recognized store/helper",
                    "stale_annotations",
                )
            )

    for index, line in enumerate(lines):
        if is_comment_or_blank(line):
            continue

        reason: str | None = None
        if "crates/perry-codegen/src/expr" in path.as_posix():
            if is_risky_codegen_store(line):
                reason = "raw generated heap/global store"
        else:
            reason = classify_rust_store(path, lines, index)

        if reason and not has_nearby_marker(lines, index):
            findings.append(Finding(path, index + 1, line, reason))

    return findings


def run_self_tests() -> int:
    failures: list[str] = []

    def check(rel_path: str, lines: list[str], expected: str | None) -> None:
        reason = classify_rust_store(REPO_ROOT / rel_path, lines, 0)
        if expected is None:
            if reason is not None:
                failures.append(f"{rel_path}: expected clean, got {reason!r}")
        elif reason is None or expected not in reason:
            failures.append(f"{rel_path}: expected {expected!r}, got {reason!r}")

    check(
        "crates/perry-runtime/src/array.rs",
        ["*dst.add(i) = *src.add(i);"],
        "raw direct slot assignment",
    )
    check(
        "crates/perry-runtime/src/object/field_get_set.rs",
        ["*dst_data.add(i) = *src_data.add(i);"],
        "raw direct slot assignment",
    )
    check(
        "crates/perry-runtime/src/array.rs",
        ["std::ptr::copy_nonoverlapping(src, dst, len as usize);"],
        "raw slot copy",
    )
    check(
        "crates/perry-runtime/src/array.rs",
        [
            "std::ptr::copy(",
            "    elements.add(s as usize),",
            "    elements.add(t as usize),",
            "    count as usize,",
            ");",
        ],
        "raw slot copy",
    )
    check(
        "crates/perry-runtime/src/buffer.rs",
        ["ptr::copy_nonoverlapping(src_data, dst_data, buf_len);"],
        None,
    )
    check(
        "crates/perry-stdlib/src/crypto.rs",
        ["std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());"],
        None,
    )
    check(
        "crates/perry-runtime/src/object/mod.rs",
        ["CACHED.store(value.to_bits(), Ordering::Relaxed);"],
        "raw atomic cache/global pointer store",
    )
    check(
        "crates/perry-runtime/src/object/mod.rs",
        ["match GLOBAL_THIS_PTR.compare_exchange(0, new_ptr, Ordering::AcqRel, Ordering::Acquire) {"],
        "raw atomic cache/global pointer CAS",
    )
    check(
        "crates/perry-runtime/src/string.rs",
        ["SMALL_INT_CACHE[idx] = ptr;"],
        "raw cache/global pointer table store",
    )
    check(
        "crates/perry-runtime/src/string.rs",
        ["entry.string_ptr = key as usize;"],
        "raw cache/global pointer field store",
    )
    check(
        "crates/perry-runtime/src/string.rs",
        [
            "INTERN_TABLE[0] = InternEntry {",
            "    hash: 0xC0DEC0DE,",
            "    string_ptr,",
            "};",
        ],
        "raw cache/global pointer table store",
    )
    check(
        "crates/perry-runtime/src/object/mod.rs",
        [
            "TRANSITION_CACHE_GLOBAL[slot] = TransitionEntry {",
            "    prev_keys,",
            "    key_ptr: kp,",
            "    next_keys,",
            "};",
        ],
        "raw cache/global pointer table store",
    )
    check(
        "crates/perry-runtime/src/object/mod.rs",
        [
            "(*cache.get())[slot] = ShapeCacheEntry {",
            "    shape_id,",
            "    keys_array,",
            "};",
        ],
        "raw TLS cache pointer table store",
    )
    check(
        "crates/perry-runtime/src/object/mod.rs",
        ["NEXT_ID.store(1, Ordering::Relaxed);"],
        None,
    )
    check(
        "crates/perry-runtime/src/object/mod.rs",
        ["READY.store(true, Ordering::Release);"],
        None,
    )
    check(
        "crates/perry-runtime/src/json.rs",
        ["std::ptr::write(slot, JSValue::from_bits(value_bits));"],
        "raw slot write",
    )
    check(
        "crates/perry-runtime/src/regex.rs",
        ["std::ptr::write(elements_ptr.add(i), nanboxed);"],
        "raw slot write",
    )
    check(
        "crates/perry-runtime/src/plugin.rs",
        ["*fields.add(1) = make_nanboxed_string(&name);"],
        "raw direct slot assignment",
    )
    check(
        "crates/perry-runtime/src/plugin.rs",
        ["(*obj).keys_array = keys_arr;"],
        "raw heap pointer field store",
    )
    check(
        "crates/perry-runtime/src/thread.rs",
        ["*arr_elements.add(i) = f64::from_bits(bits);"],
        "raw direct slot assignment",
    )
    check(
        "crates/perry-runtime/src/thread.rs",
        ["*fields_ptr.add(i) = f64::from_bits(bits);"],
        "raw direct slot assignment",
    )
    check(
        "crates/perry-runtime/src/thread.rs",
        ["*keys_elements.add(i) = f64::from_bits(key_val.bits());"],
        "raw direct slot assignment",
    )
    check(
        "crates/perry-runtime/src/promise.rs",
        ["*fields.add(0) = promise_box_handle.get_nanbox_f64();"],
        "raw direct slot assignment",
    )
    check(
        "crates/perry-runtime/src/promise/then.rs",
        ["(*promise).on_fulfilled = callback;"],
        "raw Promise heap pointer field store",
    )
    check(
        "crates/perry-runtime/src/promise/then.rs",
        ["(*promise).next = next;"],
        "raw Promise heap pointer field store",
    )

    if failures:
        print("GC store-site inventory self-test failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("GC store-site inventory self-test passed.")
    return 0


def collect_inventory() -> tuple[list[Finding], int, int]:
    findings: list[Finding] = []
    marker_count = 0
    seen: set[Path] = set()
    for path in iter_scan_roots():
        if path in seen:
            continue
        seen.add(path)
        try:
            marker_count += sum(
                1
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                for match in ANY_MARKER_RE.finditer(line)
                if marker_is_valid(match)
            )
        except OSError:
            pass
        findings.extend(scan_file(path))
    findings.extend(validate_gc_type_metadata())
    return findings, len(seen), marker_count


def write_inventory_json(path: Path, findings: list[Finding], files_scanned: int, marker_count: int) -> None:
    counts = collections.Counter(finding.error_class for finding in findings)
    packet = {
        "schema_version": 1,
        "status": "fail" if findings else "pass",
        "errors": [finding.render() for finding in findings],
        "summary": {
            "files_scanned": files_scanned,
            "audited_sites": marker_count,
            "unaudited_sites": counts["unaudited_store_sites"],
            "invalid_annotations": counts["invalid_annotations"],
            "stale_annotations": counts["stale_annotations"],
            "missing_gc_type_metadata": counts["missing_gc_type_metadata"],
        },
        "unaudited_sites": [
            {
                "path": str(finding.path.relative_to(REPO_ROOT)),
                "line": finding.line_no,
                "reason": finding.reason,
                "text": finding.text.strip(),
            }
            for finding in findings
            if finding.error_class == "unaudited_store_sites"
        ],
        "invalid_annotations": [
            {
                "path": str(finding.path.relative_to(REPO_ROOT)),
                "line": finding.line_no,
                "reason": finding.reason,
                "text": finding.text.strip(),
            }
            for finding in findings
            if finding.error_class == "invalid_annotations"
        ],
        "stale_annotations": [
            {
                "path": str(finding.path.relative_to(REPO_ROOT)),
                "line": finding.line_no,
                "reason": finding.reason,
                "text": finding.text.strip(),
            }
            for finding in findings
            if finding.error_class == "stale_annotations"
        ],
        "missing_gc_type_metadata": [
            {
                "path": str(finding.path.relative_to(REPO_ROOT)),
                "line": finding.line_no,
                "reason": finding.reason,
                "text": finding.text.strip(),
            }
            for finding in findings
            if finding.error_class == "missing_gc_type_metadata"
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--self-test"]:
        return run_self_tests()

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_tests()

    findings, files_scanned, marker_count = collect_inventory()
    if args.json_out:
        write_inventory_json(args.json_out, findings, files_scanned, marker_count)

    if findings:
        print("GC store-site inventory failed:")
        labels = [
            ("unaudited_store_sites", "unaudited store sites"),
            ("invalid_annotations", "invalid GC_STORE_AUDIT annotations"),
            ("stale_annotations", "stale GC_STORE_AUDIT annotations"),
            ("missing_gc_type_metadata", "missing GC type metadata"),
        ]
        for error_class, label in labels:
            grouped = [finding for finding in findings if finding.error_class == error_class]
            if not grouped:
                continue
            print(f"  {label}:")
            for finding in grouped:
                print(f"    {finding.render()}")
        print(
            "\nAccepted marker form: "
            "// GC_STORE_AUDIT(BARRIERED): reason, with class one of "
            + ", ".join(sorted(AUDIT_CLASSES))
        )
        return 1

    print(
        f"GC store-site inventory passed ({files_scanned} files scanned; "
        "zero unaudited sites, zero invalid annotations, "
        "zero stale annotations, zero missing GC type metadata)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
