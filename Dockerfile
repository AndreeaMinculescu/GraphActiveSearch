# syntax=docker/dockerfile:1.4
FROM julia:1.12 AS base

WORKDIR /active_search

# Copy project files
COPY Project.toml Manifest.toml ./

# Copy source code
COPY src/ ./src/

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Detect architecture
ARG TARGETARCH

# ARM64: instantiate packages at build time
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        julia --project=. -e 'using Pkg; Pkg.instantiate()'; \
    fi

# Set entrypoint for all architectures
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["julia", "--project=.", "src/active_search_obsTopo.jl"]