# syntax=docker/dockerfile:1
# =============================================================================
# Multi-stage Dockerfile for mcp-gateway
#
# Produces a single runtime image containing:
#   - Rust TCP shim (shim-mcp-gateway) on :8090
#   - Python FastAPI gateway on :8087
#   - Config, idle-timeout .env, and helper scripts
#
# The Rust shim runs as PID 1 and cold-starts the Python gateway on demand.
# =============================================================================

# ── Stage 1: Rust builder ────────────────────────────────────────────────────
FROM rust:1.82-slim-bookworm AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache dependency layer — dummy build downloads crates
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    cargo build --release 2>&1; \
    rm -rf src

# Real build — Cargo.lock + target/ are shared via cache mounts
COPY src/ src/
RUN --mount=type=cache,target=/build/target \
    cargo build --release && \
    cp target/release/shim-mcp-gateway /shim-mcp-gateway && \
    strip /shim-mcp-gateway


# ── Stage 2: Python dependency stage ─────────────────────────────────────────
FROM python:3.12-slim-bookworm AS py-deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency manifests first for layer caching
COPY pyproject.toml README.md ./

# Install project deps into a venv
RUN python -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "hatchling>=1.0" && \
    pip install --no-cache-dir .  # installs all [project.dependencies] (not editable)


# ── Stage 3: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

LABEL maintainer="AI Engineering Team"
LABEL description="MCP-compliant gateway service with Rust cold-start shim"
LABEL version="0.1.0"

# Install runtime dependencies (curl for health checks, ca-certificates for TLS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1001 mcp && \
    useradd -u 1001 -g mcp -m -d /app mcp

WORKDIR /app

# ── Copy Python venv from py-deps stage ──────────────────────────────────────
COPY --from=py-deps /app/.venv /app/.venv

# ── Copy Rust shim from rust-builder stage ───────────────────────────────────
COPY --from=rust-builder /shim-mcp-gateway /usr/local/bin/shim-mcp-gateway

# ── Copy application code ────────────────────────────────────────────────────
COPY src/ ./src/
COPY config/ ./config/

# ── Copy helper scripts (adapted for Docker — no PID file, no lsof) ─────────
COPY start_mcp_gateway.sh /usr/local/bin/start_mcp_gateway.sh
RUN chmod +x /usr/local/bin/start_mcp_gateway.sh && \
    chown -R mcp:mcp /app

# Make sure Python can find the venv
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    MCP_GATEWAY_ENV_FILE="/app/config/idle_timeout.env" \
    DEPLOYMENT_MODE=docker

# Expose ports:
#   8090 — Rust shim (cold-start proxy, primary entry point for clients)
#   8087 — Python FastAPI gateway (internal, started on demand by shim)
EXPOSE 8090 8087

# Switch to non-root user
USER mcp

# The Rust shim is the entry point — it cold-starts the Python gateway
# when the first request arrives. This replaces auto_start_shim.sh's
# PID-file/lsof dance, since Docker manages the single process.
ENTRYPOINT ["/usr/local/bin/shim-mcp-gateway"]
