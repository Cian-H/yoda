# yoda

Experimental custom system 1 decision engine architecture.

## Purpose

Yoda is an experimental System 1 decision engine designed for fast, intuitive, low-latency heuristic evaluation while remaining rigorously grounded by symbolic constraints and calibrated uncertainty. The engine operates at the intersection of rapid semantic retrieval and formal neuro-symbolic constraint verification:
- **Semantic Encoding & Retrieval**: Fast embedding spaces and transformer representations for incoming queries, context payloads, and environment states.
- **Neuro-Symbolic Integration (NeSy)**: Grounding decisions against hard logic rules, invariants, and ontological constraints without sacrificing inference speed.
- **Probabilistic Methods & Uncertainty Quantification**: Leveraging continuous differentiable logic relaxations (e.g. Logic Tensor Networks (LTN), Kleene/Belnap multi-valued semantics) to differentiate epistemic uncertainty from true ambiguity.
- **Explainability & Auditing (XAI)**: Native attribution and decision trace auditing to verify why a particular intuitive decision path was selected or pruned.

The project is implemented in Python 3.14 utilizing PyTorch, PyTorch Lightning, Optuna, Safetensors, Polars, and Pydantic within a reproducible Nix (`devenv`) environment.

## Rules

### Always

Five things, on every turn, whichever agent is reading this.

- **Confirm the reading before building.** When a request is short and admits more than one reading, say in one line which reading you are acting on, then act. Before the work, not after it. A wrong reading is cheap to correct at one line and expensive to correct at one commit.
- **Answer the request that was made.** Not the adjacent one you can answer more impressively. If a rule below would have you produce an artifact the request did not ask for, the request wins and the artifact waits to be offered.
- **Declare the task boundary.** State in one line at the top of each turn whether it continues the current task or opens a new one — e.g. *"Task: continuing 'add password reset' — refinement to the previous turn"* or *"Task: new — 'wire up SES'. Previous task committed at abc1234, closed"*. A commit closes the current task by default; the next turn is presumed new unless it is a fix-up on the just-committed work. Explicit user signals (*"now let's..."*, *"moving on..."*, *"unrelated:"*, *"different topic:"*) always open a new task. When the signal is ambiguous, **continue** — the cost of a mis-continuation is a longer prompt file; the cost of a mis-new-task is directory spam.
- **One prompt file per task**, under `.docs/prompts/`, amended as the task continues (not one per turn); the work itself; a commit (granularity to judgement — often one per task, sometimes two when refinements deserve separation); a push. Stage by explicit path, never `git add -A`.
- **Capture deferrals** as one file per idea under `.docs/todos/`, and remove an entry in the commit that satisfies its trigger. This especially applies to **proactive-discipline rules** (testing, metrics, telemetry — see below): when the user says *"skip this for now"*, don't drop it silently — capture a todo with a revisit trigger like *"next commit that touches this subsystem"* so the discipline gets picked up when the deferral's premise no longer holds.

### Read before you act

The files under `.agents/rules/` are **reference, and are deliberately not preloaded**. Read the file when its trigger fires, and read it *before* acting rather than after: each exists to stop a specific mistake that is expensive to undo, and reaching for one after the code is written is the failure it was meant to prevent. If a trigger is ambiguous, read the file.

Rules come in two flavors. **Reactive** rules (changes, UI, layered architecture, frontend) fire only when their specific surface is being touched — read them then, follow them then. **Proactive-discipline** rules (testing, metrics, telemetry) fire on *every* relevant work unit when opted in — read them once per session and apply the discipline on every code change, not only when the discipline's artifact is already being touched. If a project opted into metrics and you're building a new subsystem, ship events for it in the same commit; don't wait to be asked.

| When | Read |
| --- | --- |
| the full per-task loop, once per session before the first commit | [`workflow.md`](.agents/rules/workflow.md) |
| a new dependency, module, layer or pattern | [`workflow.md`](.agents/rules/workflow.md) §2 (ADR) |
| a new, changed or deleted code path, or a new failure branch | [`workflow.md`](.agents/rules/workflow.md) §3 (telemetry) |
| writing Python | [`best-practices.md`](.agents/rules/best-practices.md) |
| writing or removing a deferred-idea entry | [`workflow-todos.md`](.agents/rules/workflow-todos.md) |
| writing production code — tests ship in the same commit | [`workflow-testing.md`](.agents/rules/workflow-testing.md) |

Each row states the *condition* and the *file*, not what the file is about. If two rows fit, read both.

### Measurement habit

The rules budget is small (a few kilobytes of always-loaded material) but three larger line items compete for the same window: the conversation itself (grows every turn), MCP tool schemas (varies by connected servers), and per-host system prompts. Check `/context` occasionally when a session starts feeling forgetful; the culprit is usually one of those three, not this file.

For MCP specifically: this project's project-scoped servers, if any, are declared in `.mcp.json` at the repo root (see `.mcp.example.json` for the starter template and the safety pattern — pinned versions, name-based allowlist, per-project denylist for account-level connectors). Every server contributes tool-schema tokens to every session; a browser MCP is typically ~9k tokens for 45 tools, and heavy account-level connectors can top 20k each. On Claude Code, personal denies live in the gitignored `.claude/settings.local.json`'s `deniedMcpServers` array; on other hosts, the equivalent lives in the host's own settings.

Architecture decisions and their trade-offs live in [`.docs/adrs/`](.docs/adrs/) — read these before making structural changes.

## Run

```bash
devenv shell (or uv sync)
uv run pytest -o pythonpath=src
uv run ruff check .
uv run marimo edit notebooks/workbench.py
```

## Architecture map

```text
├── src/yoda/              # Core package implementation
│   ├── architecture/      # Decision payload schemas and model definitions
│   ├── nesy/              # Neuro-symbolic integration and rule constraints
│   ├── probabilistic/     # Uncertainty quantification, logic relaxations, and sampling
│   └── xai/               # Explainability tools and decision auditing
├── experiments/           # Training scripts, benchmarking runs, and evaluation sweeps
├── notebooks/             # Interactive Marimo notebooks for rapid prototyping
└── tests/                 # Unit, schema validation, and property-based test suites
```

## Conventions (summary)

See [`.agents/rules/best-practices.md`](.agents/rules/best-practices.md) for full detail.

## Domain Context & Background

Neuro-symbolic decision engine (NeSy logic constraints + continuous LTN/Kleene relaxations, transformer embeddings, uncertainty quantification, XAI decision traces; benchmarks against Jev/Laya/CLM/Julia-1; nix/devenv GPU environment).
