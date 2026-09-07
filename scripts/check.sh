#!/usr/bin/env bash

set -e

echo "==> Building"
cmake --build build

echo "==> Running tests"
ctest --test-dir build --output-on-failure

echo "==> Running lint"
cmake --build build --target lint

echo "==> All checks passed"