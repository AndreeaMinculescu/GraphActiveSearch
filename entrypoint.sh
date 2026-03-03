#!/bin/bash
set -e

FLAG_FILE="/active_search/.julia_instantiated"

# AMD64: instantiate only if not done yet
if [ "$(uname -m)" = "x86_64" ] && [ ! -f "$FLAG_FILE" ]; then
    echo "Instantiating Julia packages for AMD64 on first run..."
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
    touch "$FLAG_FILE"
fi

# Execute the command
exec "$@"
