# TurboContext Mission

## What we are building

Cross-session context memory for Claude Code. When a developer uses
`/turbocontext`, Claude recalls which files and strategies worked in similar
past sessions and records the outcome of the current session.

The system gets slightly better every time it's used — not through RL or
parameter optimization, but through simple accumulation of experience.

## Core metric

**Recall precision**: % of recommended files that Claude actually reads.
Target: > 60%.

## What success looks like

A developer types `/turbocontext review the auth module`. The system returns:
"3 similar sessions found. In those sessions, `src/auth/login.ts`,
`src/auth/middleware.ts`, and `src/types.ts` were most useful. The strategy
'check token validation first, then authorization logic' worked well twice."

Claude reads those files first, follows the strategy, completes the review,
and records the session. Next time someone reviews auth code, the
recommendations are slightly better.

No RL. No Thompson sampling. No eligibility traces. Just remembering what
worked.
