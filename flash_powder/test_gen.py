#!/usr/bin/env python3
import argparse
import re
from multiprocessing import Pool

# Lets just do this here... such that the multiprocessing forks get it.
import torch


def evaluate_blocks(blocks: list[list[str]]):
    res = []
    locals = {}
    for block in blocks:
        exec("\n".join(block), locals=locals, globals={"torch": torch})
        print(locals)


class RustTestReader:
    @staticmethod
    def find_matching_index(lines, pattern, start=None):
        for i, line in enumerate(lines):
            if start is not None and i < start:
                continue
            if re.search(pattern, line):
                return i
        return None

    def __init__(self, content):
        self._content = content
        self._lines = self._content.split("\n")

    def get_function_lines(self, function_name):
        # Assume perfectly formatted code.
        # fn test_flash_power_conv2d(
        start = RustTestReader.find_matching_index(
            self._lines, rf"fn {function_name}\("
        )
        start_line = self._lines[start]
        indent_count = len(start_line) - len(start_line.lstrip(" "))
        # Find the next `}` line with the same index.
        closing = RustTestReader.find_matching_index(
            self._lines, "^" + " " * indent_count + "}" + "$", start=start
        )
        lines = self._lines[start : closing + 1]
        return lines

    def extract_python(self, lines: list[str]):
        python_blocks = []
        stage = None
        for l in lines:
            if "|PYTHON" in l:
                if stage is not None:
                    raise ValueError("Opened python block while one is open")
                stage = []
                continue
            elif "*/" in l:
                if stage is not None:
                    python_blocks.append(stage)
                stage = None
            if stage is not None:
                stage.append(l.strip())
        return python_blocks

    def run_blocks(self, blocks: list[list[str]]):
        res = evaluate_blocks(blocks)


def run_extract(args):
    with open(args.input) as f:
        d = f.read()

    reader = RustTestReader(d)
    function_lines = reader.get_function_lines(args.test_case)

    statements = reader.extract_python(function_lines)
    print(statements)
    reader.run_blocks(statements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument(
        "input",
        help="The suppression file to operate on",
    )
    extract_parser.add_argument(
        "test_case",
        help="The test case to operate on",
    )
    extract_parser.set_defaults(func=run_extract)

    args = parser.parse_args()

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit()

    args.func(args)
