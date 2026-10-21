"""`fusionlab` command line: offline experiments on recorded captures.

Commands are grouped in modules that mirror the source tree. Each module's
`register(commands)` adds its subcommands and binds a handler with
`set_defaults(handler=...)`; `main()` only parses and dispatches. To add a
command, add a parser and handler to the matching module (or a new module
listed in COMMAND_GROUPS). Handlers import heavy optional dependencies lazily
so unrelated commands work without them.
"""

import argparse

from fusionlab.cli import evaluation, inertial, recordings, vision, visualization

# Registration order is the order commands appear in `fusionlab --help`.
COMMAND_GROUPS = (recordings, inertial, vision, visualization, evaluation)


def build_parser() -> argparse.ArgumentParser:
    """Create the parser with every command group registered."""
    parser = argparse.ArgumentParser(description="Fusion Lab offline experiments")
    # Subcommands share one entry point; each experiment declares its own
    # inputs and assumptions before dispatching to the processing modules.
    commands = parser.add_subparsers(dest="command", required=True)
    for group in COMMAND_GROUPS:
        group.register(commands)
    return parser


def main() -> None:
    """Parse arguments and run the chosen command's handler."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"fusionlab: {exc}\n")
