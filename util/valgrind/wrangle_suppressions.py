#!/usr/bin/env python3
import argparse
import re

torch_entry_points = ["fun:torch_call_dispatcher"]


def process_supression(name, suppresion_lines):
    res = []
    for i, line in enumerate(suppresion_lines):
        if i == 1:
            line = line.replace("<insert_a_suppression_name_here>", name)
        res.append(line)
        if line.strip() in torch_entry_points:
            res.append(suppresion_lines[-1])
            break
    return "\n".join(res)


def run_extract(args):
    with open(args.input) as f:
        t = f.read()

    suppressions = []

    lines = t.split("\n")

    next_name = None
    suppression_lines = []
    for line in lines:
        if line.strip().startswith("suppression@"):
            next_name = line.strip()
        if line.strip() == "{":
            suppression_lines.append(line)
        elif line.strip() == "}":
            suppression_lines.append(line)
            if next_name is None:
                raise ValueError(f"Missing name for suppression {suppression_lines}")
            finished_suppresion = process_supression(next_name, suppression_lines)
            suppressions.append(finished_suppresion)
            suppression_lines = []
            next_name = None
        elif suppression_lines:
            suppression_lines.append(line)

    with open(args.output, "w") as f:
        f.write("\n".join(suppressions) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument(
        "input",
        help="The suppression file to operate on",
    )
    extract_parser.add_argument(
        "--output",
        help="The suppression file to write",
    )
    extract_parser.set_defaults(func=run_extract)

    args = parser.parse_args()

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit()

    args.func(args)
