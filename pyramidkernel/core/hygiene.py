"""Candidate hygiene helpers for editable Python regions.

These helpers fix high-confidence protocol mistakes inside region bodies before
expensive evaluation. They intentionally avoid semantic rewrites.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field


_PYTHON_REGION_PREFIXES = ("forward_stmt_",)
_PYTHON_REGION_NAMES = {"helpers", "user_helpers", "init_body", "forward_body"}


@dataclass(frozen=True)
class HygieneResult:
    body: str
    logs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceSyntaxResult:
    ok: bool
    failure_type: str | None = None
    message: str | None = None


def is_python_region(region: str) -> bool:
    name = str(region or "").strip()
    return name in _PYTHON_REGION_NAMES or name.startswith(_PYTHON_REGION_PREFIXES)


def normalize_region_body(region: str, body: str) -> HygieneResult:
    """Normalize only Python editable regions; leave CUDA/C++ regions unchanged."""
    text = str(body or "").strip("\n")
    if not is_python_region(region):
        return HygieneResult(text, [])
    logs: list[str] = []
    normalized = _normalize_python_indentation(text)
    if normalized != text:
        logs.append(f"normalized_python_region_indent:{region}")
    fixed = _fix_tensor_property_calls(normalized)
    if fixed != normalized:
        logs.append("autofix_tensor_property_call")
    return HygieneResult(fixed, logs)


def validate_python_source(source_code: str) -> SourceSyntaxResult:
    try:
        ast.parse(source_code)
    except SyntaxError as exc:
        return SourceSyntaxResult(False, "python_region_syntax_error", _format_syntax_error(exc))
    return SourceSyntaxResult(True)


def _normalize_python_indentation(body: str) -> str:
    dedented = textwrap.dedent(body).strip("\n")
    if not dedented:
        return dedented
    if _wrapped_body_is_valid(dedented):
        return dedented
    repaired = _repair_indent_by_original_stack(dedented.splitlines())
    if _wrapped_body_is_valid(repaired):
        return repaired
    return dedented


def _repair_indent_by_original_stack(lines: list[str]) -> str:
    rendered: list[str] = []
    # Keep the model's original indentation as a hint, but render a stable
    # four-space logical layout. Generated patches often over-indent a whole
    # nested block while still preserving relative dedents (for example,
    # ``try``/``except``). A stack of raw indentation levels lets us recover
    # those relative transitions without guessing from absolute column counts.
    indent_stack: list[tuple[int, int]] = [(0, 0)]
    for raw_line in lines:
        if not raw_line.strip():
            rendered.append("")
            continue
        original_indent = len(raw_line) - len(raw_line.lstrip(" \t"))
        stripped = raw_line.strip()
        while len(indent_stack) > 1 and original_indent < indent_stack[-1][0]:
            indent_stack.pop()
        if original_indent > indent_stack[-1][0]:
            # In valid Python this increase follows a block opener. Accept an
            # increase without that signal as well: model output may contain
            # harmlessly inconsistent columns, and preserving the nesting is
            # safer than flattening executable statements.
            indent_stack.append((original_indent, indent_stack[-1][1] + 1))
        logical_indent = indent_stack[-1][1]
        rendered.append("    " * logical_indent + stripped)
    return "\n".join(rendered)


def _wrapped_body_is_valid(body: str) -> bool:
    wrapped = "def _pk_region_probe():\n" + textwrap.indent(body or "pass", "    ")
    try:
        ast.parse(wrapped)
    except SyntaxError:
        return False
    return True


def _fix_tensor_property_calls(body: str) -> str:
    # High-confidence PyTorch properties. Do not touch is_contiguous(), which is a method.
    fixed = re.sub(r"(\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\.is_cuda\s*\(\s*\)", r"\1.is_cuda", body)
    fixed = re.sub(r"(\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\.training\s*\(\s*\)", r"\1.training", fixed)
    fixed = re.sub(r"(\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\.dtype\s*\(\s*\)", r"\1.dtype", fixed)
    fixed = re.sub(r"(\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\.device\s*\(\s*\)", r"\1.device", fixed)
    return fixed


def _format_syntax_error(exc: SyntaxError) -> str:
    location = f"line {exc.lineno}"
    if exc.offset:
        location += f", column {exc.offset}"
    return f"{exc.msg} ({location})"
