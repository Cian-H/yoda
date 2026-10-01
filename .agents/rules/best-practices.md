<!-- best-practices: refined · sources: [Astral uv docs, Ruff docs, PyTorch 2.x docs, Pydantic v2 docs, Polars docs] · accessed: 2026-10-01 · lang: Python · arch: FLAT -->
# Best Practices

Patterns and conventions established for the **yoda** System 1 Decision Engine. This project targets modern **Python 3.14** using the Astral toolchain (`uv`, `ruff`), `pydantic` v2, `torch` 2.4+, `pytorch-lightning`, `polars`, and `hypothesis`. Apply these practices when authoring schemas, implementing neuro-symbolic operators, designing probabilistic logic relaxations, or refactoring core modules.

## Architecture

This codebase is organised as a **flat, modular neuro-symbolic research library and experiment sandbox** (`src/yoda/` with topic-focused submodules):
- `yoda.architecture`: Pydantic payload schemas (`DecisionPayload`, `QueryContext`), state representations, and base model definitions.
- `yoda.nesy`: Neuro-symbolic integration, logic rule checkers, invariants, and continuous relaxation operators (Logic Tensor Networks, t-norms).
- `yoda.probabilistic`: Uncertainty calibration, Kleene/Belnap 3-valued/4-valued logic, and epistemic uncertainty estimators.
- `yoda.xai`: Explainable AI auditing, feature attribution, and decision trace provenance.

Dependencies flow inward from experiment scripts and notebooks to the core library. Modules within `src/yoda` communicate via explicit Pydantic schemas and typed dataclasses. See [`workflow.md`](./workflow.md) for the one-prompt-one-commit task discipline.

## Repository & Data Layer

- **Tabular Data**: Use `polars` for dataset processing and benchmarking metrics rather than legacy pandas, leveraging multi-threaded lazy execution (`LazyFrame.collect()`).
- **Model Checkpoints**: Use `safetensors` for storing and loading model weights rather than pickle-based torch saves ([Safetensors documentation](https://huggingface.co/docs/safetensors/)).
- **Data Access Encapsulation**: Encapsulate dataset loading and fixture generation behind explicit loader functions or classes in `src/yoda/data/` or `experiments/`.
- **Deterministic Ordering**: Sort all returned evaluation records and telemetry outputs deterministically (e.g. by query ID or timestamp).

## Service & Decision Engine Pattern

- **Stateless Evaluators**: Decision engines and rule checkers should be stateless callable components receiving query payloads and returning structured decisions.
- **Explicit Invariant Checking**: Hard symbolic constraints must be evaluated with explicit boolean or continuous relaxation bounds, returning clear satisfaction scores.
- **Fail-Safe Fallbacks**: If heuristic evaluation encounters numerical instability or out-of-distribution input, fall back cleanly with calibrated epistemic uncertainty rather than raising uncaught exceptions.

## Dependency Injection & Inversion

- **Constructor Injection**: Inject embedding models, rule sets, and threshold parameters into engine constructors or factory functions (`__init__`), avoiding hidden global singletons.
- **Configurability**: Model hyperparameters, relaxation temperatures, and uncertainty thresholds should be configured via Pydantic settings or explicit argument structures.
- **Testability**: Evaluators must accept test-double encoders or mock semantic vectors to enable fast unit testing without loading heavyweight GPU models.

## Code Style

- **Python 3.14 Idioms**: Utilize modern Python syntax (`|` union types, generic type parameters `class Model[T]:`, `typing.Self`).
- **Type Annotations**: Mandatory type hints on all public functions, classes, and methods. Code is linted using `ruff` with Pyflakes, pycodestyle, Bugbear, and comprehensions enabled.
- **Line Length & Formatting**: Code conforms to 100-character line length enforced by `ruff format` and `ruff check`.
- **Validation at Boundaries**: Ingested raw payloads and external query dictionaries must be validated into Pydantic models (`DecisionPayload.model_validate(...)`). Once validated, internal code works with typed attributes.
- **Immutability & Pure Functions**: Favor pure mathematical functions and non-mutating transformations for tensor logic operations (e.g., t-norm aggregators).

## File & Naming Conventions

- **Modules**: Snake case (`schema.py`, `logic_relaxations.py`).
- **Classes**: PascalCase (`DecisionPayload`, `QueryContext`, `LogicTensorEvaluator`).
- **Constants**: `UPPER_SNAKE_CASE` at module scope.
- **Private Members**: Prefix non-public functions or helper classes with a single underscore (`_compute_tnorm`).

## Data & Formatting

- **Tensors**: Maintain consistent tensor shape documentation in docstrings (e.g., `(batch_size, embedding_dim)`).
- **Serialization**: Use `model.model_dump()` or `model.model_dump_json()` for Pydantic serialization.
- **Metrics**: Standardize evaluation metric keys across benchmarks (`latency_ms`, `constraint_satisfaction`, `epistemic_uncertainty`).

## Testing Notes

Follow [`workflow-testing.md`](./workflow-testing.md) for testing discipline:
- **Pytest**: Place tests under `tests/` matching module topics (`test_schema.py`, `test_nesy.py`).
- **Property-Based Testing**: Use `hypothesis` for verifying neuro-symbolic logic invariants and continuous relaxation boundary conditions (e.g., verifying that t-norms satisfy associativity and monotonicity across `[0, 1]`).
- **Execution**: Run test suites using `uv run pytest` (with `pythonpath = ["src"]` configured in `pyproject.toml`).

## What NOT to do

- **Don't use mutable default arguments** in functions or schema fields (use `Field(default_factory=...)` in Pydantic).
- **Don't use `pickle` for model artifacts**; use `safetensors`.
- **Don't bypass type validation** when ingesting queries into the decision engine.
- **Don't run unpinned stochastic tests**; seed PyTorch, NumPy, and Python RNGs in fixtures when testing probabilistic models.
- **Don't commit `.env` files or temporary benchmark run checkpoints** to git.
- **Don't bundle unrelated changes in one commit**; adhere to one task = prompt + commit per `workflow.md`.

## References

- [Astral uv Documentation](https://docs.astral.sh/uv/) (accessed 2026-10-01)
- [Ruff Linter & Formatter](https://docs.astral.sh/ruff/) (accessed 2026-10-01)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/) (accessed 2026-10-01)
- [PyTorch 2.x Documentation](https://pytorch.org/docs/stable/index.html) (accessed 2026-10-01)
- [Polars Documentation](https://pola.rs/) (accessed 2026-10-01)
