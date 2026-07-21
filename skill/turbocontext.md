---
name: turbocontext
description: |
  Cross-session context memory for Claude Code. Recalls what worked in similar past 
  sessions and records outcomes for future recall. Usage: /turbocontext <task>
---

# TurboContext — Session Memory for Claude Code

When invoked with `/turbocontext`, follow this protocol:

## Phase 1: RECALL

1. Call `turbocontext_recall` with the user's task description and current working directory.
   Optionally pass `taskType` for better matching (code_review, code_generation, debugging, etc.).
2. Review the recommendations:
   - **Recommended files**: Read the top 3-5 files that were useful in similar past sessions.
   - **Recommended strategies**: Consider approaches that worked before.
   - **Summary**: Read the summary paragraph for quick context.

## Phase 2: EXECUTE

Use Claude's native reasoning. Follow a 3-round loop (understand → execute → verify):

1. **Understand**: Analyze the task, read relevant files (prioritize recommended files), plan the approach.
2. **Execute**: Implement the solution. If code generation, produce the code. If analysis, produce the analysis.
3. **Verify**: Check the output. For code: does it compile? Are edge cases handled? For analysis: are claims supported?

If quality is insufficient, return to step 1 with the new understanding (max 3 rounds).

## Phase 3: RECORD

1. Self-assess quality on a 0-1 scale:
   - 0.9-1.0: Excellent, production-ready, all edge cases handled
   - 0.7-0.9: Good, minor issues only
   - 0.5-0.7: Acceptable but needs refinement
   - 0.0-0.5: Incomplete or incorrect
2. Call `turbocontext_record` with:
   - Task description, type, working directory
   - Files read and files modified during the session
   - 1-3 sentence strategy summary
   - Outcome (success/partial/failure)
   - Self-assessment score
   - Brief notes on what worked and what to try next time
   - Number of rounds used

## Output Format

After recording, summarize for the user:

```
## TurboContext Session

**Recall:** Found {N} similar sessions. Top recommendations:
- {file1} (used in {count} past sessions)
- {file2} (used in {count} past sessions)

**Strategy:** {what approach you took}

**Outcome:** {success|partial|failure} | Quality: {score}%

**Recorded:** Session saved for future recall ({total} sessions in corpus)
```

## Configuration

Data stored at `~/.turbocontext/sessions.json` and `~/.turbocontext/sessions.idx`.
No API keys, no external services, no model calls beyond Claude Code itself.
