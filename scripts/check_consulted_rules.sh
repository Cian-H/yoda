#!/usr/bin/env bash
# Cross-check: staged paths → trigger index → prompt file's Consulted rules.
# See AGENTS.md's "Read before you act" table for what each rule stops.

set -euo pipefail

mapfile -t staged < <(git diff --cached --name-only --diff-filter=ACM)
if [ ${#staged[@]} -eq 0 ]; then exit 0; fi

# The prompt file being committed, or the most recent one on disk if we're
# amending / continuing a task. Missing = soft-skip.
prompt_file=""
for f in "${staged[@]}"; do
  case "$f" in .docs/prompts/*.md) prompt_file="$f"; break ;; esac
done
if [ -z "$prompt_file" ]; then
  prompt_file=$(ls -1t .docs/prompts/*.md 2>/dev/null | head -n 1 || true)
fi
if [ -z "$prompt_file" ] || [ ! -f "$prompt_file" ]; then
  echo "check_consulted_rules: no prompt file staged or on disk — soft-skip"
  exit 0
fi
if ! grep -q '^## Consulted rules' "$prompt_file"; then
  echo "check_consulted_rules: $prompt_file has no '## Consulted rules' section — soft-skip"
  echo "  (add the section per .agents/rules/workflow.md §1 for the check to fire)"
  exit 0
fi

# Path patterns that fire triggers. Add arms here when a new path-shaped
# row lands in AGENTS.md's trigger index. Interview flags gate each arm;
# a project without a flag skips the arm entirely.
declare -A needs=()
for path in "${staged[@]}"; do
  case "$path" in
    *) ;;
  esac
done
if [ ${#needs[@]} -eq 0 ]; then exit 0; fi

# Extract the Consulted rules block for cross-checking. A rule is satisfied
# either by a plain mention or by an explicit waiver: `<rule> (n/a — reason)`.
consulted=$(awk '/^## Consulted rules/{f=1; next} /^## /{f=0} f' "$prompt_file")

missing=()
for rule in "${!needs[@]}"; do
  if ! grep -qF "$rule" <<<"$consulted"; then
    missing+=("$rule")
  fi
done

if [ ${#missing[@]} -eq 0 ]; then exit 0; fi

echo "check_consulted_rules: staged paths fire trigger(s) not named in"
echo "  $prompt_file's '## Consulted rules' section:"
for rule in "${missing[@]}"; do
  echo "  - $rule"
done
echo
echo "Fix: open the rule file and confirm you followed it, then add a line to"
echo "'## Consulted rules' naming it. If the trigger doesn't apply to this"
echo "change, waive it explicitly with the reason: '<rule> (n/a — <one-line>)'"
exit 1
