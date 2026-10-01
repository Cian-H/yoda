# List available recipes
default:
    @just --list

# Run lint and test suites
check:
    uv run ruff check .
    uv run pytest

# Launch Marimo interactive workbench
workbench:
    uv run marimo edit notebooks/workbench.py

# Remove build, bytecode, and cache artefacts
clean:
    rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .mypy_cache/ htmlcov/ .coverage
    find . -type d -name __pycache__ -exec rm -rf {} +

# Pre-commit hook: cross-check staged paths against trigger index in agents.md
[no-exit-message]
check-rules:
    #!/usr/bin/env bash
    set -euo pipefail

    mapfile -t staged < <(git diff --cached --name-only --diff-filter=ACM)
    if [ ${#staged[@]} -eq 0 ]; then exit 0; fi

    prompt_file=""
    for f in "${staged[@]}"; do
      case "$f" in .docs/prompts/*.md) prompt_file="$f"; break ;; esac
    done
    if [ -z "$prompt_file" ]; then
      prompt_file=$(ls -1t .docs/prompts/*.md 2>/dev/null | head -n 1 || true)
    fi
    if [ -z "$prompt_file" ] || [ ! -f "$prompt_file" ]; then
      echo "check-rules: no prompt file staged or on disk — soft-skip"
      exit 0
    fi
    if ! grep -q '^## Consulted rules' "$prompt_file"; then
      echo "check-rules: $prompt_file has no '## Consulted rules' section — soft-skip"
      echo "  (add the section per .agents/rules/workflow.md §1 for the check to fire)"
      exit 0
    fi

    declare -A needs=()
    for path in "${staged[@]}"; do
      case "$path" in
        *.py)
          needs["best-practices.md"]=1
          ;;
      esac
      case "$path" in
        src/yoda/*.py)
          needs["workflow-testing.md"]=1
          ;;
        .docs/todos/*.md)
          needs["workflow-todos.md"]=1
          ;;
        .docs/adrs/[0-9]*.md)
          needs["workflow.md"]=1
          ;;
      esac
    done
    if [ ${#needs[@]} -eq 0 ]; then exit 0; fi

    consulted=$(awk '/^## Consulted rules/{f=1; next} /^## /{f=0} f' "$prompt_file")

    missing=()
    for rule in "${!needs[@]}"; do
      if ! grep -qF "$rule" <<<"$consulted"; then
        missing+=("$rule")
      fi
    done

    if [ ${#missing[@]} -eq 0 ]; then exit 0; fi

    echo "check-rules: staged paths fire trigger(s) not named in"
    echo "  $prompt_file's '## Consulted rules' section:"
    for rule in "${missing[@]}"; do
      echo "  - $rule"
    done
    echo
    echo "Fix: open the rule file and confirm you followed it, then add a line to"
    echo "'## Consulted rules' naming it. If the trigger doesn't apply to this"
    echo "change, waive it explicitly with the reason: '<rule> (n/a — <one-line>)'"
    exit 1
