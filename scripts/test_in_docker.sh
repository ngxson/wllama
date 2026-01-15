#!/usr/bin/env bash

export TEST_ARGS="$@"
SCRIPT_DIR=$(dirname -- "$BASH_SOURCE")
docker-compose -f "$SCRIPT_DIR/docker-compose.yml" up --no-log-prefix --build --abort-on-container-exit --exit-code-from wllama-test --remove-orphans wllama-test