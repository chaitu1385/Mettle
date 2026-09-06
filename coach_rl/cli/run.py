"""`coach-run` -- hold a coaching session in the terminal and log every turn."""

from __future__ import annotations

import argparse
import asyncio
import sys

from ..config import Settings
from ..llm import LLM
from ..policy import build_policy
from ..prompts import PROMPT_VERSION
from ..storage import open_db

BANNER = """coach-rl session
  Type your message and press enter. Blank line to skip.
  /quit   end the session   /state  show the last logged state + action
  Ctrl-D also ends the session.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="coach-run", description="Run an instrumented coaching session."
    )
    parser.add_argument("--db", default=settings.db_path, help="SQLite path")
    parser.add_argument("--model", default=settings.model, help="Anthropic model id")
    parser.add_argument(
        "--eps",
        type=float,
        default=settings.eps,
        help="Exploration rate; 0 disables the epsilon wrapper",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed the exploration RNG")
    parser.add_argument("--notes", default=None, help="Session notes stored on the row")
    parser.add_argument(
        "--show-judge",
        action="store_true",
        help="Print judge scores as they land (they arrive a turn late)",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    # Imported here so `--help` works without the API key or langgraph installed.
    from ..session import CoachSession

    policy = build_policy(eps=args.eps, seed=args.seed)
    llm = LLM(model=args.model)

    with open_db(args.db) as conn:
        session = CoachSession(
            llm=llm,
            policy=policy,
            conn=conn,
            notes=args.notes,
            verbose_judge=args.show_judge,
        )
        print(BANNER)
        print(
            f"session {session.session_id} | model {args.model} | "
            f"policy {policy.policy_id} | prompts {PROMPT_VERSION} | db {args.db}\n"
        )

        last_record = None
        try:
            while True:
                try:
                    user_message = await asyncio.to_thread(input, "you > ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                message = user_message.strip()
                if not message:
                    continue
                if message in ("/quit", "/exit"):
                    break
                if message == "/state":
                    if last_record is None:
                        print("(no turns yet)\n")
                    else:
                        print(f"  state:  {last_record.state}")
                        print(f"  action: {last_record.action} ({last_record.policy_id})")
                        print(f"  probs:  {last_record.action_probs}\n")
                    continue

                try:
                    reply, last_record = await session.turn(message)
                except Exception as exc:
                    print(f"[error] turn failed, nothing logged: {exc}\n", file=sys.stderr)
                    continue

                marker = " *explore*" if last_record.explored else ""
                print(f"\ncoach [{last_record.action}{marker}] > {reply}\n")
        finally:
            print("closing session (waiting for outstanding judge calls)...")
            await session.close()
            if session.judge.failures:
                print(f"judge failures: {len(session.judge.failures)}", file=sys.stderr)
                for failure in session.judge.failures:
                    print(f"  {failure}", file=sys.stderr)
            print(f"session {session.session_id}: {session.turn_index} turns logged to {args.db}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
