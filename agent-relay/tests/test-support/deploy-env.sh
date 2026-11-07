node() { "$SCENARIO_NODE" --import "$SCENARIO_PRELOAD" "$@"; }
pnpm() { node "$SCENARIO_PORT" "$@"; }
