# Yoda: System 1 Decision Engine

> *"Do or do not. There is no try."*
>
> System 1 reasoning in a nutshell: no hesitation, instinctual decisions.

## Overview

This repository is my testing ground and experimental sandbox for developing a
custom System 1 decision engine architecture.

The goal is to explore architectures capable of fast, intuitive, low-latency
heuristic evaluation and decision-making, while remaining grounded by symbolic
constraints and calibrated uncertainty. This project draws inspiration, query
representations, and comparative benchmarks across a spectrum of systems
(including **Jev**, **Laya**, **CLM**, and **Julia-1**) while aiming to create a
custom architecture optimised for this specific task of system 1 decision
making.

## Core Concepts & Architecture

The architecture explores the intersection of rapid semantic retrieval and
rigorous constraint verification:

- **Semantic Encoding & Retrieval (Transformers):** Fast embedding spaces for
  encoding incoming queries, context payloads, and environmental states.
- **Neuro-Symbolic Integration (NeSy):** Grounding decisions against hard logic
  rules, invariants, and ontological constraints without sacrificing inference
  speed.
- **Probabilistic Methods & Uncertainty Quantification:** Drawing directly from
  methods, leveraging continuous differentiable logic relaxations (e.g., LTN,
  Kleene/Belnap semantics) to differentiate epistemic uncertainty from true
  ambiguity.
- **Explainability & Auditing (XAI):** Built-in attribution and decision trace
  auditing to verify *why* a particular intuitive path was selected or pruned.

## Project Structure

```text
├── data/                  # Dataset management, raw inputs, and fixtures
├── experiments/           # Training scripts, benchmarking runs, and evaluation sweeps
├── models/                # Serialized model weights and checkpoints
├── notebooks/             # Interactive Marimo notebooks for rapid prototyping
├── src/yoda/              # Core package implementation
│   ├── architecture/      # Decision payload schemas and model definitions
│   ├── nesy/              # Neuro-symbolic integration and rule constraints
│   ├── probabilistic/     # Uncertainty quantification, logic relaxations, and sampling
│   └── xai/               # Explainability tools and decision auditing
└── tests/                 # Unit, schema validation, and property-based test suites
```

## License

This project is licensed under the Apache License 2.0. See [license.md](license.md) for details.

