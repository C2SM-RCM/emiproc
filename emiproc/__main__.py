"""Command line interface for emiproc.

It is deprecated since v1 but could be used for utilities if needed.
"""
import argparse

if __name__ == '__main__':
    # Create a parser for the command line arguments
    # The help message displays a deprecated notice
    parser = argparse.ArgumentParser(
        description='Deprecated command line use of emiproc. Use emiproc v1.',
    )

    # parse the command line arguments
    args = parser.parse_args()


    parser.print_help()

