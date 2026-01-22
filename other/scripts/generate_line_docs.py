import argparse
from pathlib import Path


SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def ascii_escape(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def extract_name(prefix: str, stripped: str) -> str:
    remainder = stripped[len(prefix) :].strip()
    name = remainder.split("(")[0].split(":")[0].strip()
    return name or "unknown"


def is_assignment(stripped: str) -> bool:
    if "=" not in stripped:
        return False
    if stripped.startswith(("if ", "elif ", "while ", "for ", "return ", "yield ", "assert ")):
        return False
    if "==" in stripped or "!=" in stripped or "<=" in stripped or ">=" in stripped:
        return False
    if ":=" in stripped:
        return False
    if stripped.lstrip().startswith(("#", "import ", "from ")):
        return False
    return True


def describe_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "blank line for readability."
    if stripped.startswith("#"):
        comment = stripped[1:].strip()
        return f"comment: {ascii_escape(comment)}" if comment else "comment line."
    if stripped.startswith("@"):
        return "decorator for the next definition."
    if stripped.startswith(("\"\"\"", "'''")):
        return "start or end of a triple-quoted string or docstring."
    if "\"\"\"" in stripped or "'''" in stripped:
        return "part of a triple-quoted string or docstring."
    if stripped.startswith("import "):
        return "import module(s) for use in this file."
    if stripped.startswith("from "):
        return "import symbols from a module for use in this file."
    if stripped.startswith("class "):
        name = extract_name("class", stripped)
        return f"define class {name}."
    if stripped.startswith("async def "):
        name = extract_name("async def", stripped)
        return f"define async function {name}."
    if stripped.startswith("def "):
        name = extract_name("def", stripped)
        return f"define function {name}."
    if stripped.startswith("if "):
        return "start conditional branch."
    if stripped.startswith("elif "):
        return "continue conditional with another branch."
    if stripped.startswith("else"):
        return "fallback branch for conditional."
    if stripped.startswith("for "):
        return "start for-loop over an iterable."
    if stripped.startswith("async for "):
        return "start async for-loop over an async iterable."
    if stripped.startswith("while "):
        return "start while-loop."
    if stripped.startswith("with "):
        return "enter context manager block."
    if stripped.startswith("async with "):
        return "enter async context manager block."
    if stripped.startswith("try"):
        return "start exception handling block."
    if stripped.startswith("except"):
        return "handle an exception."
    if stripped.startswith("finally"):
        return "run cleanup regardless of exceptions."
    if stripped.startswith("raise"):
        return "raise an exception."
    if stripped.startswith("assert"):
        return "assert a condition for correctness."
    if stripped.startswith("return"):
        return "return a value from the current function."
    if stripped.startswith("yield"):
        return "yield a value from a generator."
    if stripped.startswith("pass"):
        return "explicit no-op placeholder."
    if stripped.startswith("continue"):
        return "skip to the next loop iteration."
    if stripped.startswith("break"):
        return "break out of the current loop."
    if stripped.startswith("global "):
        return "declare global variable usage."
    if stripped.startswith("nonlocal "):
        return "declare nonlocal variable usage."
    if stripped.startswith("del "):
        return "delete a reference."
    if stripped in {")", "]", "}", "):", "],", "},"}:
        return "close a grouped expression or block."
    if is_assignment(stripped):
        lhs = stripped.split("=", 1)[0].strip()
        return f"assign a value to {ascii_escape(lhs)}."
    if stripped.endswith(":"):
        return "open a new indented block."
    return "execute a statement within the current context."


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return any(skip in parts for skip in SKIP_DIRS)


def iter_py_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if should_skip(path):
            continue
        files.append(path)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate line-by-line intent docs for Python files.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/line_docs"))
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    files = iter_py_files(root)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# Line-by-line Python docs index",
        "",
        f"Root: {ascii_escape(str(root))}",
        f"Total files: {len(files)}",
        "",
    ]

    for path in files:
        rel_path = path.relative_to(root)
        doc_rel = rel_path.with_suffix(rel_path.suffix + ".md")
        doc_path = output_dir / doc_rel
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            content = []

        width = max(4, len(str(len(content))))
        header = f"# Line-by-line documentation for {ascii_escape(str(rel_path))}\n"
        note = "Auto-generated line intent. Lines are escaped to ASCII with unicode escapes.\n\n"
        with doc_path.open("w", encoding="ascii", errors="strict") as handle:
            handle.write(header)
            handle.write("\n")
            handle.write(note)
            for idx, line in enumerate(content, start=1):
                escaped_line = ascii_escape(line)
                intent = describe_line(line)
                handle.write(f"L{idx:0{width}d}: {escaped_line} -- {intent}\n")

        index_lines.append(f"- {ascii_escape(str(rel_path))} -> {ascii_escape(str(doc_rel))}")

    index_path = output_dir / "INDEX.md"
    with index_path.open("w", encoding="ascii", errors="strict") as handle:
        handle.write("\n".join(index_lines))


if __name__ == "__main__":
    main()
