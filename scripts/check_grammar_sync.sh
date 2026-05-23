#!/usr/bin/env bash
# check_grammar_sync.sh
# Verifies that grammars/*.g4 source files match the grammar headers embedded in
# the generated src/*Parser.py files (ANTLR 4.10 stamp check).
#
# Exit 0 = sync OK
# Exit 1 = mismatch detected (regeneration required)
#
# Usage:
#   bash scripts/check_grammar_sync.sh
#
# Part of the matrix-GM integration (M1.5). Run before every commit that touches grammar files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRAMMARS_DIR="$REPO_ROOT/grammars"
SRC_DIR="$REPO_ROOT/src"
ERRORS=0

check_grammar() {
    local grammar="$1"          # e.g. SOGA
    local g4_file="$GRAMMARS_DIR/${grammar}.g4"
    local parser_file="$SRC_DIR/${grammar}Parser.py"
    local lexer_file="$SRC_DIR/${grammar}Lexer.py"

    if [[ ! -f "$g4_file" ]]; then
        echo "ERROR: grammar source not found: $g4_file"
        ERRORS=$((ERRORS + 1))
        return
    fi

    for gen_file in "$parser_file" "$lexer_file"; do
        if [[ ! -f "$gen_file" ]]; then
            echo "ERROR: generated file not found: $gen_file"
            ERRORS=$((ERRORS + 1))
            continue
        fi

        # Check that the generated file was produced from this grammar name
        if ! head -3 "$gen_file" | grep -q "Generated from ${grammar}.g4 by ANTLR 4.10"; then
            echo "ERROR: $gen_file header does not match 'Generated from ${grammar}.g4 by ANTLR 4.10'"
            echo "       Run: java -jar antlr-4.10-complete.jar -Dlanguage=Python3 -visitor -listener $g4_file -o $SRC_DIR/"
            ERRORS=$((ERRORS + 1))
        else
            echo "OK  $gen_file"
        fi
    done
}

check_grammar "SOGA"
check_grammar "ASGMT"
check_grammar "TRUNC"

if [[ $ERRORS -gt 0 ]]; then
    echo ""
    echo "check_grammar_sync.sh: $ERRORS error(s) found. Regenerate before committing."
    exit 1
else
    echo ""
    echo "check_grammar_sync.sh: all grammar files in sync."
    exit 0
fi
