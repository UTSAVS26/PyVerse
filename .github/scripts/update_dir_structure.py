#!/usr/bin/env python3

import os
import argparse
from collections.abc import Iterator


def good_file_paths(top_dir: str = ".") -> Iterator[str]:
    """
    Generator that yields paths of Python or Jupyter Notebook files, excluding certain directories.

    Args:
        top_dir (str): The root directory to start searching from.

    Yields:
        str: File paths relative to the top directory.
    """
    for dir_path, dir_names, filenames in os.walk(top_dir):
        dir_names[:] = [
            d for d in dir_names
            if d != "scripts" and d[0] not in "._" and "venv" not in d
        ]
        for filename in filenames:
            if filename != "__init__.py" and os.path.splitext(filename)[1] in (".py", ".ipynb"):
                yield os.path.join(dir_path, filename).lstrip("./")


def md_prefix(indent_level: int) -> str:
    """
    Generate Markdown prefix for directories and files based on indentation level.

    Args:
        indent_level (int): The level of indentation.

    Returns:
        str: Markdown prefix string.
    """
    return f"{'  ' * indent_level}*" if indent_level else "\n##"


def print_path(old_path: str, new_path: str) -> str:
    """
    Print Markdown formatted directory structure between old_path and new_path.

    Args:
        old_path (str): The previous directory path.
        new_path (str): The current directory path.

    Returns:
        str: The new directory path.
    """
    old_parts = old_path.split(os.sep)
    for i, new_part in enumerate(new_path.split(os.sep)):
        if i >= len(old_parts) or old_parts[i] != new_part:
            print(f"{md_prefix(i)} {new_part.replace('_', ' ').title()}")
    return new_path


def print_directory_md(top_dir: str = ".") -> None:
    """
    Print the directory structure in Markdown format, including links to Python and Jupyter Notebook files.

    Args:
        top_dir (str): The root directory to start searching from.
    """
    old_path = ""
    for filepath in sorted(good_file_paths(top_dir)):
        dirpath, filename = os.path.split(filepath)
        if dirpath != old_path:
            old_path = print_path(old_path, dirpath)
        indent_level = dirpath.count(os.sep) + 1 if dirpath else 0
        url = f"{dirpath}/{filename}".replace(" ", "%20")
        file_title = os.path.splitext(filename.replace("_", " ").title())[0]
        print(f"{md_prefix(indent_level)} [{file_title}]({url})")


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate a Markdown-formatted file structure for Python and Jupyter files.")
    parser.add_argument("directory", nargs="?", default=".", help="The root directory to generate the Markdown structure for (default: current directory).")
    args = parser.parse_args()

    # Check if the directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return

    print_directory_md(args.directory)


if __name__ == "__main__":
    main()
