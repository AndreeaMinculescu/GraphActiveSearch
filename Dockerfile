FROM julia:1.12

WORKDIR /active_search

# Install dependencies
COPY Project.toml Manifest.toml ./
RUN julia --project=. -e "using Pkg; Pkg.instantiate()"

# Copy source code
COPY src/ ./src/

# Default script (can be overridden at runtime)
CMD ["julia", "--project=.", "src/active_search_obsTopo.jl"]