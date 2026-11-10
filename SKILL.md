# Council — Agent Skill

This skill tells an agent when to call the `council_deliberate` tool exposed by
the council-mcp server.

## When to call the council

Call `council_deliberate` **before answering** when:

1. **Architecture decisions** — system design, service layout, "should I migrate
   X to Y", "what is the best approach for…".
2. **Debugging strategy** — complex diagnostics where the root cause is unclear.
3. **High-risk changes** — any operational change that could take down a service.
4. **Model choice questions** — "which model is best for X", "should we switch
   providers".

Trigger words: "architecture", "migrate", "refactor", "redesign", "best
approach", "should I", "what is the right way", "is it safe to", "production",
"critical".

Then answer from the synthesis, noting dissent and confidence.

## Skip the council for

- Simple status queries ("is the service running?").
- Trivia and knowledge lookup.
- CRUD operations (add this, delete that).
- Decisions that have already been made.

## Degradation path

The server may report per-provider errors or degraded advocates. If it does,
state the failures and give your own answer rather than pretending full coverage.
Check `contributors_count` before claiming multiple models agreed.
