"""`coach-stats` -- describe the dataset and validate the judge against human labels."""

from __future__ import annotations

import argparse

from ..config import Settings
from ..stats import format_report
from ..storage import fetch_turns, open_db


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="coach-stats",
        description="Dataset statistics and judge-vs-human validation."
    )
    parser.add_argument("--db", default=settings.db_path, help="SQLite path")
    parser.add_argument(
        "--session", default=None, help="Restrict to one session (default: all)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with open_db(args.db) as conn:
        print(format_report(fetch_turns(conn, args.session)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
