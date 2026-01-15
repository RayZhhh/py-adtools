import argparse
import sys

# Import the main function of the auto_server
from adtools.evaluator.auto_server import main as auto_server_main


def main():
    parser = argparse.ArgumentParser(
        description="ADTools CLI for various utilities.",
        formatter_class=argparse.RawTextHelpFormatter,  # Preserve formatting for help messages
    )

    # Define subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve subcommand
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the Auto-Evaluation Server. All arguments are passed directly to the server.",
        formatter_class=argparse.RawTextHelpFormatter,  # Preserve formatting for help messages
    )

    # The 'serve' command doesn't define its own arguments here.
    # Instead, it will capture all subsequent arguments and pass them to auto_server_main.
    # This simplifies argument parsing, as auto_server_main already handles its own args.

    # Parse arguments provided to the main 'adtools' command
    # We only parse the initial command (e.g., 'adtools serve')
    # and then pass the rest of the arguments to the subcommand's main function.

    # If no subcommand is given, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Parse the main command and subcommand
    args = parser.parse_args(sys.argv[1:2])  # Only parse 'adtools' and 'serve'

    if args.command == "serve":
        # Pass all remaining arguments (from sys.argv[2:]) to the auto_server_main function.
        # This effectively makes 'adtools serve ARGS...' behave like 'adtools.evaluator.auto_server.main(ARGS...)'.
        # We need to temporarily replace sys.argv for auto_server_main to parse correctly.
        original_sys_argv = sys.argv
        sys.argv = [original_sys_argv[0]] + original_sys_argv[
            2:
        ]  # Keep script name, then add server args

        try:
            auto_server_main()
        finally:
            sys.argv = original_sys_argv  # Restore sys.argv
    else:
        # This case should ideally not be reached if subparsers are configured correctly
        # and 'dest="command"' is used.
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
