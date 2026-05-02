#!/usr/bin/env python3
import argparse
import copy
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

import torch


class Color:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    ORANGE = "\033[93m"
    RESET = "\033[0m"
    RED = "\033[91m"
    WHITE = "\033[97m"


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
class RustConstant:
    ident: str
    type: str
    lines: list[Line]


@dataclass
class RustBlock:
    lines: list[Line]

    def find_constants(self) -> list[RustConstant]:
        constants = []
        index = 0
        current_constant = None
        while index < len(self.lines):
            line = self.lines[index]
            res = re.findall("\\s*const ([^ ]+): ?([^ ]+)", line.line)
            if res:
                current_constant = RustConstant(
                    ident=res[0][0], type=res[0][1], lines=[]
                )
            if current_constant is not None:
                current_constant.lines.append(line)
            if line.line.endswith(";"):
                if current_constant is not None:
                    constants.append(current_constant)
                    current_constant = None
            index += 1
        return constants


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
            else:
                raise ValueError(f"Unknown block type {type(b)}")


def run_main(args):

    reader = RustTestReader(Path(args.input))
    function_lines = reader.get_function_lines(args.test_case)

    blocks = reader.extract_blocks(function_lines)
    if args.command == "extract":
        for b in blocks:
            if isinstance(b, PythonBlock):
                our_block = dedent("\n".join(a.line for a in b.lines))
                print(our_block)
        sys.exit(0)

    reader.run_blocks(blocks)
    if args.command == "execute":
        for b in blocks:
            if isinstance(b, PythonBlock):
                our_block = dedent("\n".join(a.line for a in b.lines))
                print(Color.BLUE + our_block + Color.RESET)
                print("--")
                for k, v in b.results.items():
                    print(f"{k} = {v}")
                print("====")
    if args.command == "substitute":
        for i, b in enumerate(blocks):
            if isinstance(b, RustBlock):
                # Find identifiers for constants.
                constants = b.find_constants()
                print(constants)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    def add_common_args(parser):
        parser.add_argument(
            "input",
            help="The suppression file to operate on",
        )
        parser.add_argument(
            "test_case",
            help="The test case to operate on",
        )

    extract_parser = subparsers.add_parser("extract")
    add_common_args(extract_parser)
    extract_parser.set_defaults(func=run_main)

    execute_parser = subparsers.add_parser("execute")
    add_common_args(execute_parser)
    execute_parser.set_defaults(func=run_main)

    substitute_parser = subparsers.add_parser("substitute")
    add_common_args(substitute_parser)
    substitute_parser.set_defaults(func=run_main)

    args = parser.parse_args()

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit()

    args.func(args)
