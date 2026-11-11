"""Identifier charset guard (homoglyph incident 2026-11-04).

During a live file-editing race the class name BothubClient briefly
existed in a source file with a lookalike (Cyrillic) codepoint. The test
suite stayed green because `from __future__ import annotations` makes
annotations lazy strings and internal references never re-import the
name; only an external `from services.llm_search import BothhubClient`
failed. This guard fails fast on any non-ASCII identifier so a poisoned
paste can never reach the suite green again.

Scans all first-party .py files (services/, bot/, scripts/ used by the
pipeline) via ast: every identifier the parser binds (module/class/def,
arguments, names, attributes, imports-as) must be pure ASCII. Strings,
comments and docstrings are NOT checked (prompts are legitimately
Russian).
"""
import ast
import pathlib
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCAN_DIRS = ("services", "bot", "scripts")


def _iter_identifiers(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name
            args = node.args
            for a in args.args + args.kwonlyargs + args.posonlyargs:
                yield a.arg
            if args.vararg:
                yield args.vararg.arg
            if args.kwarg:
                yield args.kwarg.arg
        elif isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.Import,)):
            for alias in node.names:
                yield alias.asname or alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                yield alias.asname or alias.name


def _first_party_files():
    for d in SCAN_DIRS:
        base = REPO / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            yield p


@pytest.mark.parametrize("path", list(_first_party_files()), ids=lambda p: p.name)
def test_identifiers_are_ascii(path: pathlib.Path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad = sorted({name for name in _iter_identifiers(tree)
                  if not name.isascii()})
    assert not bad, f"{path.relative_to(REPO)}: non-ASCII identifiers {bad}"
