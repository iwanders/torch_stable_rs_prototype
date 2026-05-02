#!/usr/bin/env python3
import argparse
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

import torch


@dataclass
class Line:
    index: int
    line: str
    filename: str

    def __contains__(self, item):
        return item in self.line


def evaluate_blocks(lines: list[Line], locals) -> dict[str, Any]:
    # raise SyntaxError("custom error message", ("myfile.py", 100, 1, "invalid code here"))
    our_block = dedent("\n".join(a.line for a in lines))
    locals = copy.deepcopy(locals)
    try:
        exec(our_block, locals=locals, globals={"torch": torch})
    except IndentationError as e:
        e.lineno += lines[0].index
        e.filename = lines[0].filename
        raise e
    except SyntaxError as e:
        e.lineno += lines[0].index
        e.filename = lines[0].filename
        raise e
        raise e
    except RuntimeError as e:
        raise e

    return locals


@dataclass
class PythonBlock:
    lines: list[Line]
    results: dict[str, Any]


def evaluate_python_blocks(blocks: list[PythonBlock]) -> dict[str, Any]:
    # raise SyntaxError("custom error message
    return locals


@dataclass
class PythonBlock:
    lines: list[Line]
    results: dict[str, Any]


@dataclass
class RustBlock:
    lines: list[Line]


class RustTestReader:
    @staticmethod
    def find_matching_index(lines, pattern, start=None):
        for i, line in enumerate(lines):
            if start is not None and i < start:
                continue
            if re.search(pattern, line):
                return i
        return None

    def __init__(self, filename: Path):
        with open(filename) as f:
            d = f.read()

        self._content = d
        self._lines = self._content.split("\n")
        self._filename = filename.name

    def get_function_lines(self, function_name) -> list[Line]:
        # Assume perfectly formatted code.
        # fn test_flash_power_conv2d(
        start = RustTestReader.find_matching_index(
            self._lines, rf"fn {function_name}\("
        )
        if start is None:
            raise KeyError(f"Failed to find {function_name}")
        start_line = self._lines[start]
        indent_count = len(start_line) - len(start_line.lstrip(" "))
        # Find the next `}` line with the same index.
        closing = RustTestReader.find_matching_index(
            self._lines, "^" + " " * indent_count + "}" + "$", start=start
        )
        lines = []
        for li in range(start, closing + 1):
            lines.append(Line(index=li, line=self._lines[li], filename=self._filename))

        return lines

    def extract_blocks(self, lines: list[Line]) -> list[RustBlock | PythonBlock]:
        blocks = []
        rust_block = RustBlock(lines=[])
        python_block = None

        def push_block():
            nonlocal rust_block
            nonlocal python_block
            if rust_block is not None:
                blocks.append(rust_block)
            elif python_block is not None:
                blocks.append(python_block)
            rust_block = None
            python_block = None

        for line in lines:
            if "|PYTHON" in line:
                if python_block is not None:
                    raise ValueError("Opened python block while one is open")
                push_block()
                python_block = PythonBlock(lines=[line], results={})
                continue
            elif "*/" in line:
                if python_block is not None:
                    push_block()
                    rust_block = RustBlock(lines=[line])
            if rust_block is not None:
                rust_block.lines.append(line)
            if python_block is not None:
                python_block.lines.append(line)
        push_block()
        return blocks

    def run_blocks(self, blocks: list[RustBlock | PythonBlock]):
        locals = {}
        for b in blocks:
            if isinstance(b, RustBlock):
                pass
            elif isinstance(b, PythonBlock):
                res = evaluate_blocks(b.lines, locals)
                b.results = res
                print(res)
            else:
                raise ValueError(f"Unknown block type {type(b)}")


def run_extract(args):
    reader = RustTestReader(Path(args.input))
    function_lines = reader.get_function_lines(args.test_case)

    blocks = reader.extract_blocks(function_lines)
    for b in blocks:
        print(b)
    reader.run_blocks(blocks)


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
