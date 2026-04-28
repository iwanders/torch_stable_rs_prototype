#!/usr/bin/env python3
import argparse


def run_extract(args):
    with open(args.input) as f:
        t = f.read()

    print(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="The suppression file to operate on",
    )
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.set_defaults(func=run_extract)

    args = parser.parse_args()

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit()

    args.func(args)
