"""Train or evaluate the BrainWeb saturation-recovery comparison."""

import argparse

from pinqi_mrf.saturation_recovery import evaluate, train


def main() -> None:
    """Dispatch the selected command to its argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'evaluate'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = train.main if args.command == 'train' else evaluate.main
    command(args.arguments)


if __name__ == '__main__':
    main()
