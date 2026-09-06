"""`coach-label` -- replay a session and attach human labels to its turns.

The human label is the ground truth the LLM judge is validated against, so it is
collected separately and after the fact: labelling in the moment would just be
the judge's opinion with extra steps.

Labels: 1 = the turn helped, 0 = neutral, -1 = the turn was bad.
"""

from __future__ import annotations

import argparse
import textwrap

from ..config import Settings
from ..storage import fetch_turns, list_sessions, open_db, set_human_label

PROMPT = "label [1 good / 0 neutral / -1 bad / enter=skip / q=quit] > "
LABELS = {"1": 1, "+1": 1, "0": 0, "-1": -1}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="coach-label",
        description="Replay a session and attach human labels to its turns."
    )
    parser.add_argument("--db", default=settings.db_path, help="SQLite path")
    parser.add_argument("--session", default=None, help="Session id to replay")
    parser.add_argument("--list", action="store_true", help="List sessions and exit")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include turns that already carry a label (default: unlabeled only)",
    )
    return parser.parse_args(argv)


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.indent(textwrap.fill(text, width=88), indent)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with open_db(args.db) as conn:
        sessions = list_sessions(conn)
        if args.list or not args.session:
            if not sessions:
                print(f"No sessions in {args.db}.")
                return 1
            print(f"{'session_id':<16}{'started':<28}{'turns':>6}  notes")
            for row in sessions:
                notes = row["session_notes"] or ""
                print(
                    f"{row['session_id']:<16}{row['started_at']:<28}"
                    f"{row['turn_count']:>6}  {notes}"
                )
            if not args.session:
                print("\nPick one:  coach-label --session <session_id>")
            return 0

        turns = fetch_turns(conn, args.session)
        if not turns:
            print(f"No turns for session {args.session!r} in {args.db}.")
            return 1

        pending = [t for t in turns if args.all or t.human_label is None]
        print(
            f"session {args.session}: {len(turns)} turns, "
            f"{len(pending)} to label. Ctrl-D or q to stop.\n"
        )

        labeled = 0
        for turn in pending:
            scores = turn.judge_scores
            judge_line = (
                "judge: (not scored -- last turn of the session)"
                if not scores
                else "judge: "
                + " ".join(f"{k}={v}" for k, v in scores.items())
                + f"  | {turn.judge_rationale or ''}"
            )
            print("-" * 88)
            print(f"turn {turn.turn_index}  action={turn.action}  policy={turn.policy_id}")
            print(f"  state: {turn.state}")
            print("  coachee:")
            print(wrap(turn.user_message, "    "))
            print("  coach:")
            print(wrap(turn.response_text, "    "))
            print(f"  {judge_line}")
            if turn.human_label is not None:
                print(f"  existing label: {turn.human_label:+d}")

            try:
                answer = input(PROMPT).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if answer in ("q", "quit"):
                break
            if answer == "":
                continue
            if answer not in LABELS:
                print(f"  ignored {answer!r}; expected 1, 0, -1, or enter to skip")
                continue
            set_human_label(conn, turn.session_id, turn.turn_index, LABELS[answer])
            labeled += 1

        total_labeled = sum(1 for t in fetch_turns(conn) if t.human_label is not None)
        print(f"\nlabeled {labeled} turn(s) this pass; {total_labeled} labeled in the db.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
