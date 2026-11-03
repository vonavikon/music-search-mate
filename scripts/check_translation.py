"""Verify that a translation pass changed only comments and docstrings.

Usage:
    python3 scripts/check_translation.py            # all .py changed vs HEAD
    python3 scripts/check_translation.py <file>...  # specific files

For every file the working tree is compared against HEAD: ast.dump of both
trees must be identical after the docstring of every module/class/function
is replaced with a placeholder. Python comments are not part of the AST,
so comment edits are allowed; every other change (code, string literals,
log messages, regexes) changes the dump and fails the check.

PASS => only comments and docstrings differ from HEAD. Any FAIL means the
edit changed behavior and must be reverted or fixed.
"""
import ast
import pathlib
import subprocess
import sys

DOC_PLACEHOLDER = "<docstring>"


def _git_head_source(rel: str) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"HEAD:{rel}"], capture_output=True, check=False
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8")


class _DocstringNormalizer(ast.NodeTransformer):
    """Replace the docstring of every module/class/def with a placeholder."""

    def _normalize(self, node):
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            placeholder = ast.Expr(value=ast.Constant(value=DOC_PLACEHOLDER))
            ast.copy_location(placeholder, body[0])
            node.body = [placeholder] + body[1:]
        self.generic_visit(node)
        return node

    visit_Module = _normalize
    visit_ClassDef = _normalize
    visit_FunctionDef = _normalize
    visit_AsyncFunctionDef = _normalize


def _normalized_ast(source: str) -> str:
    tree = ast.parse(source)
    tree = _DocstringNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree)


def check_file(path: pathlib.Path, repo_root: pathlib.Path) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    head_source = _git_head_source(rel)
    if head_source is None:
        print(f"SKIP {rel}: not in HEAD (new file)")
        return True

    try:
        old_dump = _normalized_ast(head_source)
    except SyntaxError as exc:
        print(f"FAIL {rel}: HEAD version does not parse ({exc})")
        return False
    try:
        new_source = path.read_text(encoding="utf-8")
        new_dump = _normalized_ast(new_source)
    except SyntaxError as exc:
        print(f"FAIL {rel}: working tree does not parse ({exc})")
        return False

    if old_dump != new_dump:
        print(f"FAIL {rel}: AST differs beyond docstrings (code or strings changed)")
        return False

    print(f"PASS {rel}")
    return True


def main() -> int:
    repo_root = pathlib.Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    args = sys.argv[1:]
    if args:
        files = [pathlib.Path(a).resolve() for a in args]
    else:
        changed = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        files = [(repo_root / f).resolve() for f in changed if f.endswith(".py")]

    if not files:
        print("no changed .py files to check")
        return 0

    results = [check_file(p, repo_root) for p in files]
    ok = sum(results)
    print(f"--- {ok}/{len(results)} files PASS ---")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
