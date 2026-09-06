# coach-rl

An RL-instrumented executive coaching agent. The coaching is the wrapper; the
point is to learn the decision-loop shape — action selection, reward design,
exploration, and eventually off-policy evaluation. This milestone produces
correctly-shaped trainable data. **No learned policy yet.**

Everything here exists to make one thing true: every turn writes a row that a
learning algorithm could actually use later. If the row is wrong, the coaching
was pointless.

## The loop

```
user message
     │
     ▼
classify_state ──► select_action ──► generate_response ──► reply
  (LLM, typed)      (policy, no LLM)   (LLM, action as
  Pydantic-validated                     instruction)
                                              │
                                              ▼
                                     row written to SQLite
                                              │
    next user message ────────────────────────┘
              │
              ▼
           judge (separate LLM call, async, scores the PREVIOUS turn)
              │
              ▼
    UPDATE that row with the reward
```

## Action space (fixed)

`ask`, `reflect`, `challenge`, `advise`, `summarize`

Five actions, defined in `coach_rl/actions.py`, enforced by a SQL `CHECK`
constraint. The LLM never chooses an action — it only receives one as an
instruction. Adding an action invalidates every previously logged probability
vector, so it is a dataset break, not a feature.

## Design decisions worth knowing

**The full probability distribution is logged, not just the chosen action.**
Off-policy evaluation needs `π_b(a | s)` for *every* action, and it cannot be
reconstructed after the fact. So a deterministic rule logs its base distribution
`[0, 0, 1, 0, 0]` *and* the epsilon-mixed distribution the sample actually came
from:

```json
{"order": ["ask","reflect","challenge","advise","summarize"],
 "probs": [0.05, 0.05, 0.8, 0.05, 0.05],
 "base_probs": [0, 0, 1, 0, 0],
 "explored": false}
```

The mixing is exact rather than approximate —
`p(a) = (1-ε)·base(a) + ε/(N-1)·(1-base(a))` — so the logged vector is the true
sampling distribution even when the wrapped policy is stochastic. A test asserts
the empirical action frequencies match the logged vector.

**The policy is an interface, not a function.** `select(state) -> (action,
action_probs, policy_id)`. `RulePolicy` implements coaching heuristics;
`EpsilonWrapper` wraps *any* policy (ε = 0.2 by default). A learned policy later
only has to satisfy the same signature — nothing else changes.

Rule priority, since several rules fire at once and the order *is* the policy:

| # | condition | action | why it sits here |
|---|-----------|--------|------------------|
| 1 | `session_phase == closing` | `summarize` | a session that never lands is wasted |
| 2 | `user_intent == seek_advice` | `advise` | honour what was actually asked for |
| 3 | `abs(emotional_valence) >= 0.5` | `reflect` | do not challenge someone mid-feeling |
| 4 | `repeated_theme_flag` | `challenge` | a loop is a request for friction |
| 5 | uncertain or vague | `ask` | gather state |
| 6 | otherwise | `ask` | cheapest default |

**The judge scores the previous turn, not the current one.** Forward movement is
only visible in the reaction, so a turn cannot be scored until the user's next
message exists. It therefore runs asynchronously alongside the next turn and
`UPDATE`s the row when it lands. Consequence: **the last turn of every session is
never judged**, on purpose — inventing a score would poison the reward column.

**`prompt_version` is on every row, and the prompts are fingerprinted.**
`tests/test_prompt_version.py` hashes the prompt text and fails if it changed
without a version bump. Mixing prompt versions in one training set silently
corrupts it: the same state maps to different behaviour and nothing downstream
can tell. `coach-stats` also warns if a dataset spans multiple prompt versions
or models.

**The judge is a proxy reward, so it gets validated.** Once ≥ 40 turns carry a
human label, `coach-stats` reports Spearman correlations between each judge
dimension (and the summed score) and the human label, a judge-band × label
confusion table, the rationales for every outright disagreement, and a warning
for any correlation below 0.3. An unvalidated LLM judge is worse than no reward:
it looks plausible and ranks turns wrongly.

**No server-side refusal fallbacks.** A silent mid-session switch to a different
model would put two models' behaviour under one `model_name`. A refusal should be
visible in this project, not routed around.

## Data model

`turns` — one row per turn:

| column | notes |
|---|---|
| `session_id`, `turn_index` | composite primary key |
| `timestamp` | UTC ISO-8601 |
| `state_json` | the validated `TurnState` |
| `action` | `CHECK` constrained to the five actions |
| `action_probs_json` | full distribution + base distribution + explore flag |
| `policy_id`, `eps` | e.g. `rule/v1+eps0.2` |
| `response_text`, `user_message` | |
| `judge_scores_json`, `judge_rationale` | NULL until the next user message arrives |
| `human_label` | nullable; `CHECK` constrained to -1 / 0 / 1 |
| `model_name`, `prompt_version` | |

`sessions` — `session_id`, `started_at`, `ended_at`, `session_notes`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -r requirements.txt
cp .env.example .env             # add ANTHROPIC_API_KEY
```

## Use

```bash
coach-run                        # a coaching session; every turn is logged
coach-run --eps 0.3 --seed 7     # more exploration, reproducible draws

coach-label --list               # list sessions
coach-label --session <id>       # replay it, attach -1 / 0 / +1 to each turn

coach-stats                      # dataset stats + judge validation
```

The CLIs are also runnable without installing: `python -m coach_rl.cli.run`.

Runtime settings live in `coach_rl/config.py` (analysis thresholds live in
`stats.py`, next to the maths they govern) and can be overridden by env or
`.env`:
`COACH_RL_MODEL` (default `claude-opus-5`), `COACH_RL_DB` (default
`coach_rl.db`), `COACH_RL_EPS` (default `0.2`).

## Structure

The package is layered -- imports point downwards only. One rule is worth a test
(`tests/test_layering.py`): each third-party boundary lives in exactly one
module, so the domain never learns who the vendor is.

```
5  cli/run.py, cli/replay.py, cli/stats.py   terminal entry points
4  session.py     turn orchestration and row writing
   report.py      rendering of the stats (no computation)
3  graph.py       LangGraph: classify_state -> select_action -> generate_response
   judge.py       the async judge and its task tracker
   stats.py       descriptive stats + judge validation (pure functions)
2  storage.py     SQLite schema and accessors
   policy.py      Policy interface, RulePolicy, EpsilonWrapper
1  schemas.py     TurnState, JudgeScores (Pydantic)
   prompts.py     prompt text + PROMPT_VERSION + fingerprint
   llm.py         the Anthropic client -- the only module that knows the vendor
0  actions.py     the fixed action space
   config.py      settings and thresholds
```

Consequences worth stating:

- `anthropic` is imported only by `llm.py`, `langgraph` only by `graph.py`,
  `scipy` only by `stats.py`.
- `graph.py`, `judge.py`, and `session.py` call exactly two methods on the LLM,
  so the whole decision loop runs against a stub with no API key — which is how
  `tests/test_session_flow.py` exercises it.
- Computation and presentation are separate: `stats.py` returns numbers,
  `report.py` turns them into text. The validation rule ("this dimension fails")
  is defined once, on `Correlation.below_threshold`.
- Constructing a `CoachSession` writes nothing; `CoachSession.start()` opens the
  session row. No I/O hidden in a constructor.

## Tests

```bash
pytest
```

Covering the SQLite schema and its constraints, the state and judge models, the
policy interface and epsilon mixing, the correlation maths (pinned to
hand-computed Spearman values), report rendering, the prompt-version guard, the
turn-loop ordering with a stub LLM, and the vendor-boundary rule. **The LLM
calls themselves are not tested**, and neither is scipy: the tests cover this
codebase's own logic and nothing else.

## Out of scope for this milestone

Learned policy, off-policy estimators (IPS / SNIPS / doubly robust), web UI,
auth. The point of logging `action_probs` now is that the estimators can be
added later without re-collecting data.

## Next

1. Run sessions until there are a few hundred turns.
2. Label ≥ 40 of them and check the judge actually tracks your labels.
3. Only then: fit a policy, and evaluate it off-policy against the logged
   distributions.
