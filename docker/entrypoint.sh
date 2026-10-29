#!/bin/sh
# In Container Apps lyra.toml arrives as a secret env var; Lyra reads it from
# /lyra_data/config. Compose mounts the file instead and leaves LYRA_TOML unset.
set -e
if [ -n "$LYRA_TOML" ]; then
  printf '%s' "$LYRA_TOML" > /lyra_data/config/lyra.toml
fi
unset LYRA_TOML
exec "$@"
