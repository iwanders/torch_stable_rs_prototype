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


RUST_INTS = [
    "u8",
    "i8",
    "u16",
    "i16",
    "u32",
    "i32",
    "u64",
    "i64",
    "u128",
    "i128",
]
RUST_FLOATS = ["f32", "f64"]
RUST_SCALAR_TYPES = RUST_INTS + RUST_FLOATS


@dataclass
class Line:
    index: int
    line: str
    filename: str

    def __contains__(self, item):
        return item in self.line


def exec_lines(lines: list[Line], locals) -> dict[str, Any]:
    # raise SyntaxError("custom error message", ("myfile.py", 100, 1, "invalid code here"))
    our_block = dedent("\n".join(a.line for a in lines))
    locals = copy.deepcopy(locals)
    try:
        z = exec(our_block, locals=locals, globals={"torch": torch})
        print("\n\n\n", z, "\n\n\n")
    except IndentationError as e:
        e.lineno += lines[0].index
        e.filename = lines[0].filename
        raise e
    except SyntaxError as e:
        e.lineno += lines[0].index
        e.filename = lines[0].filename
        raise e
    except RuntimeError as e:
        raise e

    return locals


def eval_statement(line: Line, locals) -> Any:
    try:
        return eval(line.line, locals=locals, globals={"torch": torch})
    except IndentationError as e:
        e.lineno += lines[0].index
        e.filename = lines[0].filename
        raise e
    except SyntaxError as e:
        e.lineno += lines[0].index
        e.filename = lines[0].filename
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
class RustNode:
    children: list["RustNode | str"]

    def pretty_print(self, indent=0):
        for ci, c in enumerate(self.children):
            if isinstance(c, str):
                print(" " * indent, ci, "" + dedent(repr(c)) + "")
            if isinstance(c, RustNode):
                print(
                    " " * indent,
                    ci,
                    f" node with {len(c.children)} children: ",
                )
                c.pretty_print(indent + 4)

    def group_between_commas(self) -> "RustNode":
        segments = self._split_by_comma_indices()
        res = []
        for si, span in enumerate(segments):
            span_node = RustNode(children=[self.children[x] for x in span])
            res.append(span_node)
            if si + 1 != len(segments):
                res.append(",")
        return RustNode(children=res)

    def _split_by_comma_indices(self) -> list[list[int]]:
        ranges = []
        this_range = []
        for i, v in enumerate(self.children):
            if v == ",":
                if this_range:
                    ranges.append(this_range)
                    this_range = []
                continue
            this_range.append(i)
        if this_range:
            ranges.append(this_range)
        return ranges

    def assemble(self) -> str:
        res = ""

        def recurser(n: RustNode):
            nonlocal res
            for c in n.children:
                if isinstance(c, str):
                    res += c
                elif isinstance(c, RustNode):
                    recurser(c)

        recurser(self)
        return res

    def determine_type(self) -> str:
        is_ref = False
        is_array_esque = False
        rust_type = ""
        for i, z in enumerate(self.children):
            if isinstance(z, str):
                if i == 0 and z.strip() == "&":
                    is_ref = True
                if z.strip() == "[":
                    is_array_esque = True
            if isinstance(z, RustNode):
                # iterate one level deep to see if we can find a type there...
                for c in z.children:
                    if rust_type:
                        break
                    if isinstance(c, str):
                        for t in RUST_SCALAR_TYPES:
                            if t in c:
                                rust_type = t
                                break
        is_ref_str = "&" if is_ref else ""
        if is_array_esque:
            return f"{is_ref_str}[{rust_type}]"
        else:
            return f"{is_ref_str}{rust_type}"


"""
        starts_ref = found_type.startswith("&")
        starts_ref_str = "&" if starts_ref else ""
        # lets only handle array and slice for now.
        inner_type = re.findall("\\[(.+?)\\]", found_type)
        inner_type = inner_type[0] if inner_type else None

        if not inner_type:
            raise NotImplementedError(f"Unsupported type {found_type}")
"""


def format_payload_as_rust(payload, rust_type=None):
    if rust_type is not None:
        is_ref = rust_type.startswith("&")
        rust_type = rust_type.lstrip("&")
        is_array_esque = False
        if rust_type.startswith("[") and rust_type.endswith("]"):
            is_array_esque = True
            rust_type = rust_type[1:-1]
    else:
        is_ref = False
        rust_type = ""
        is_array_esque = isinstance(payload, list)

    print(
        f"FOrmatting, is_ref: {is_ref}, is_array_esque: {is_array_esque}, type: {rust_type}"
    )

    ref_str = "&" if is_ref else ""
    type_str = rust_type

    arr_start_str = "[" if is_array_esque else ""
    arr_end_str = "]" if is_array_esque else ""
    if isinstance(payload, int):
        if rust_type and rust_type not in RUST_INTS:
            raise ValueError(f"Trying to convert {payload} to {rust_type}")
        payload_str = str(payload)
        return f"{ref_str}{arr_start_str}{payload_str}{type_str}{arr_end_str}"

    elif isinstance(payload, float):
        if rust_type and rust_type not in RUST_FLOATS:
            raise ValueError(f"Trying to convert {payload} to {rust_type}")
        payload_str = str(payload)
        return f"{ref_str}{arr_start_str}{payload_str}{type_str}{arr_end_str}"

    elif isinstance(payload, list):
        if isinstance(payload[0], int):
            if rust_type and rust_type not in RUST_INTS:
                raise ValueError(f"Trying to convert {payload} to {rust_type}")
        if isinstance(payload[0], float):
            if rust_type and rust_type not in RUST_FLOATS:
                raise ValueError(f"Trying to convert {payload} to {rust_type}")
        payload_str = [str(p) for p in payload]

        payload_str[0] += type_str
        payload_str = ", ".join(payload_str)
        return f"{ref_str}[{payload_str}]"
    else:
        raise NotImplementedError(
            f"not implemented payload type: {type(payload)}, for {rust_type}"
        )


def test_rust_format_payload():
    assert format_payload_as_rust(3, rust_type=None) == "3"
    assert format_payload_as_rust(3.3, rust_type=None) == "3.3"
    assert format_payload_as_rust(3.3, rust_type="f32") == "3.3f32"
    assert format_payload_as_rust(3.3, rust_type="f32") == "3.3f32"
    assert format_payload_as_rust(3.3, rust_type="&f32") == "&3.3f32"
    assert format_payload_as_rust(3.3, rust_type="&[f32]") == "&[3.3f32]"
    assert format_payload_as_rust(3.3, rust_type="[f32]") == "[3.3f32]"
    assert format_payload_as_rust([3.3, 5.5], rust_type="[f32]") == "[3.3f32, 5.5]"
    assert format_payload_as_rust([3.3, 5.5], rust_type="[]") == "[3.3, 5.5]"
    assert format_payload_as_rust([3, 5], rust_type="[]") == "[3, 5]"
    assert format_payload_as_rust([3, 5], rust_type="&") == "&[3, 5]"
    sys.exit(0)


# test_rust_format_payload()


class RustWrangler:
    """
    A half-baked AST lol.
    """

    def __init__(self, rust_code):
        self._code = rust_code
        self._root = RustWrangler.parse(rust_code)
        assert self._code == self._root.assemble()

    def pretty_print(self):
        self._root.pretty_print()

    def substitute_second_argument(self, payload):
        # First, find the actual function call in the root.
        if not (self._root.children[1] == "(" and self._root.children[3] == ")"):
            raise ValueError(
                "Cannot parse this section of rust code, doesn't look like a function call"
            )
        if not isinstance(self._root.children[2], RustNode):
            raise ValueError(
                "Cannot parse this section of rust code, doesn't look like a function call"
            )
        self._root.children[2] = self._root.children[2].group_between_commas()
        assert self._code == self._root.assemble()

        # Next, the arguments are each in their own node.
        # node, ",", node, "," etc.
        last_argument = self._root.children[2].children[-1]

        found_type = last_argument.determine_type()
        payload = format_payload_as_rust(payload, found_type)
        last_argument.children = [payload]

    def assemble(self) -> str:
        return self._root.assemble()

    @staticmethod
    def parse(rust_code):

        root = RustNode(children=[])
        current: RustNode = root
        stack = []
        pairs = [("(", ")"), ("[", "]")]
        accumulated = ""
        open_lookup = {v: k for k, v in pairs}
        for i, t in enumerate(rust_code):
            if t in [",", ";"]:
                current.children.append(accumulated)
                current.children.append(t)
                accumulated = ""
                continue
            if t in [a[0] for a in pairs]:
                if t == "(" and rust_code[i + 1] == ")":
                    # treat as string:
                    pass
                else:
                    stack.append((t, current))
                    current.children.append(accumulated)
                    current.children.append(t)
                    accumulated = ""
                    new_node = RustNode(children=[])
                    current.children.append(new_node)
                    current = new_node
                    continue
            if t in [a[1] for a in pairs]:
                if t == ")" and rust_code[i - 1] == "(":
                    # treat () as string.
                    pass
                else:
                    current.children.append(accumulated)
                    if stack[-1][0] != open_lookup[t]:
                        raise ValueError("Unbalanced parenthesis?")
                    token, new_current = stack.pop()
                    current = new_current
                    current.children.append(t)
                    accumulated = ""
                    continue
            accumulated += t
        return root


# This pertains to a #PYTHON 'python statement' comment behind a rust statement.
@dataclass
class InlinePython:
    python_statement: str
    index: int
    lines: list[Line]

    @staticmethod
    def create_const(name, found_type, indent, payload) -> Line:

        payload = format_payload_as_rust(payload, found_type)
        return Line(
            0,
            indent + f"const {name}: {found_type} = {payload} ;",
            None,
        )

    def substitute_with(self, payload):
        reconstituted = "\n".join([a.line for a in self.lines])
        print(reconstituted)
        # Okay, so this is the hardest part in all of this...
        # assert_eq!(..., &[[a,b], [c,d]]); // #PYTHON
        # CONST foo: &[f32] = &[1.0, 1.2]; // #PYTHON
        filename = self.lines[0].filename
        index = self.lines[0].index
        # Next, we need to actually parse this to determine where our payload is. Our payload is the right part of the
        # const, or the second argument of the foo(a,b) type call.
        if "const" in reconstituted:
            # We can just do a regex here since we know const is very strictly defined.
            z = re.findall(
                "(\\s*)const ([^:]+): ?([^ ]+) ?=[^;]+; ?(// #PYTHON.*)",
                reconstituted,
                flags=re.DOTALL,
            )
            if not z:
                raise NotImplementedError(
                    f"something very wront with this substitution: {reconstituted}"
                )

            indent = z[0][0]
            name = z[0][1]
            found_type = z[0][2]
            res = InlinePython.create_const(name, found_type, indent, payload)
            res.filename = filename
            res.index = index
            suffix = z[0][3]
            res.line += " " + suffix
            self.lines = [res]
        else:
            # Here we have some function call... usually assert, we need to actually parse.
            wrangler = RustWrangler(reconstituted)
            wrangler.pretty_print()
            wrangler.substitute_second_argument(payload)
            wrangler.pretty_print()
            self.lines = [
                Line(index=index, filename=filename, line=wrangler.assemble())
            ]

    def substituted(self, locals) -> "InlinePython":
        res = InlinePython(
            python_statement=self.python_statement,
            lines=self.lines,
            index=self.index,
        )
        python_line = Line(
            line=self.python_statement,
            index=self.index,
            filename=self.lines[0].filename,
        )
        data = eval_statement(python_line, locals)
        res.substitute_with(data)

        return res


@dataclass
class RustBlock:
    lines: list[Line]

    def find_constants(self) -> list[RustConstant]:
        """
        const INPUT_DATA_PY: &[f32] = &[
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        """
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

    def find_inline_python(self) -> list[InlinePython]:
        """
        assert_eq!(
            d.f32_ref()?,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(d.view(-1).tolist())
        """

        def get_whole_statement(start_index) -> list[Line]:
            index = start_index - 1
            while index != 0:
                line = self.lines[index]
                if (
                    ";" in line.line
                    or line.line.endswith("*/")
                    or line.line.endswith("}")
                ):
                    return self.lines[index + 1 : start_index + 1]
                index -= 1
            return None

        entries = []
        # Easiest to parse backwards from encountering the `#PYTHON` part.
        MAGIC = "#PYTHON"
        for li, line in enumerate(self.lines):
            if MAGIC in line:
                python_statement = line.line[line.line.find(MAGIC) + len(MAGIC) + 1 :]
                rust_statement = get_whole_statement(li)
                entries.append(
                    InlinePython(
                        python_statement=python_statement,
                        lines=rust_statement,
                        index=li,
                    )
                )
        return entries


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

    def write_to(self, path: str | Path):
        path = Path(path)
        with open(path, "w") as f:
            f.write("\n".join(self._lines))

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
                res = exec_lines(b.lines, locals)
                b.results = res
            else:
                raise ValueError(f"Unknown block type {type(b)}")

    def replace_inline(self, replacements: list[tuple[InlinePython, InlinePython]]):
        # We don't want the indices to change on us, so we do it from the bottom up.
        for orig, replacement in sorted(
            replacements, key=lambda x: x[0].lines[0].index, reverse=True
        ):
            self._lines[orig.lines[0].index : orig.lines[-1].index + 1] = [
                a.line for a in replacement.lines
            ]


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
        substituted_entries = []
        for i, b in enumerate(blocks):
            if isinstance(b, RustBlock):
                # Find identifiers for constants.
                # constants = b.find_constants()
                # print(constants)

                inline_subs = b.find_inline_python()
                for s in inline_subs:
                    print(s)
                    substituted_entries.append(
                        (s, s.substituted(blocks[i - 1].results))
                    )

        reader.replace_inline(substituted_entries)
        if not args.dry_run:
            if args.output:
                reader.write_to(args.output)


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
    substitute_parser.add_argument(
        "--dry-run", "-n", default=False, action="store_true", help="Do a dry run"
    )
    substitute_parser.add_argument(
        "--output", "-o", default=None, help="Write to this output file"
    )
    add_common_args(substitute_parser)
    substitute_parser.set_defaults(func=run_main)

    args = parser.parse_args()

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit()

    args.func(args)
