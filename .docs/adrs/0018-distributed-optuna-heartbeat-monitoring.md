# 0018. Distributed Optuna Heartbeat Monitoring

- **Status**: Accepted
- **Date**: 2026-10-13

## Context

In hyperparameter optimization sweeps for Yoda, distributed execution (across multiple Colab runtimes, spot VMs, or parallel processes) allows accelerating search spaces by having multiple workers pull from and report to a shared database storage (PostgreSQL, MySQL, or persistent SQLite).

Previously:
1. **Zombie Trials**: If a worker node crashed, disconnected, or was preempted while running a trial, Optuna left the trial indefinitely in `TrialState.RUNNING`.
2. **Duplicate Baseline Seeds**: When multiple distributed workers booted up concurrently, each executed `study.enqueue_trial(CHAMPION_TRIAL_PARAMS)`, duplicating the seed baseline trial across workers.
3. **Storage Contention**: Concurrent SQLite accesses lacked a generous lock wait timeout, causing `sqlite3.OperationalError: database is locked` on simultaneous writes.

## Decision

1. **Storage Heartbeat Integration in `yoda.hyperopt.optuna`**:
   - `init_study_and_metadata` configures `optuna.storages.RDBStorage` with `heartbeat_interval` (default: 30 seconds), `grace_period` (default: `2 * heartbeat_interval`), and `heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback(max_retry=3)`.
   - On worker initialization, `fail_stale_trials(study)` is executed to immediately fail and requeue any abandoned trials left behind by previous aborted workers.
   - For SQLite backends, `connect_args={"timeout": 60.0}` is configured to handle concurrent transaction contention across workers.

2. **Idempotent Trial Enqueuing**:
   - Sweep scripts check existing study trial parameters before calling `study.enqueue_trial`, ensuring only one baseline trial is queued regardless of how many distributed workers join the study.

3. **CLI Arguments**:
   - Added `--heartbeat-interval` and `--grace-period` flags to `build_base_study_parser` to allow tuning heartbeat cadence for different network topologies.

```mermaid
sequenceDiagram
  autonumber
  participant W1 as Worker 1 (Active)
  participant DB as Shared RDB Storage
  participant W2 as Worker 2 (Joining)

  W1->>DB: Ask next trial (Trial #5)
  W1->>DB: Record heartbeat every 30s
  Note over W1: Worker 1 crashes / preempted
  Note over DB: Heartbeat ceases; Grace period (60s) expires
  W2->>DB: fail_stale_trials(study)
  DB-->>W2: Marks Trial #5 as FAIL
  DB-->>W2: Re-enqueues Trial #5 via RetryHeartbeatStaleTrialCallback
  W2->>DB: Ask next trial
  DB-->>W2: Returns re-enqueued Trial #5
```

## Consequences

- **Positive**: Distributed worker crashes or preemption automatically trigger trial recovery without manual database intervention.
- **Positive**: Multiple workers joining an active study no longer spam duplicate baseline seed trials.
- **Positive**: Zero performance overhead on local single-worker runs while providing fault-tolerant distributed capability.
