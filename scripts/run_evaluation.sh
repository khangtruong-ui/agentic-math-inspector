#!/usr/bin/env bash
# Runs benchmark evaluation comparing baseline configurations A, B, and C
set -e

SOURCE="${1:-controlled}"
PROBLEMS="${2:-5}"
SEED="${3:-42}"

echo "Starting comparative evaluation on source: $SOURCE ($PROBLEMS problems, seed: $SEED)..."
mia evaluate --source "$SOURCE" --configs "A,B,C" --num-problems "$PROBLEMS" --seed "$SEED"
