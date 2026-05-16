"""
configure_otel() — P4-07: Idempotent OTel tracing initializer.

Wraps the existing setup_tracing() with:
  - Env-var-driven conditional init (OTEL_EXPORTER_OTLP_ENDPOINT must be set)
  - _is_tracing_initialized guard (safe to call multiple times in lifespan)
  - FastAPIInstrumentor auto-instrumentation (health + metrics excluded)
  - atexit shutdown hook to flush BatchSpanProcessor on clean exit

Usage in any service's lifespan startup:
    from src.middleware.tracing import configure_otel
    configure_otel(service_name="ai-agents", app=app)
"""

from __future__ import annotations

import atexit
import logging
import os

logger = logging.getLogger(__name__)

# Idempotency guard — configure_otel() is safe to call multiple times
_is_tracing_initialized: bool = False


def configure_otel(
    service_name: str,
    app: object | None = None,
    *,
    excluded_urls: str = "health,metrics",
) -> bool:
    """
    Initialize OpenTelemetry tracing if OTEL_EXPORTER_OTLP_ENDPOINT is set.

    Args:
        service_name: Service name written to every span's resource attributes.
        app: FastAPI application instance. When provided, FastAPIInstrumentor
             is applied so all HTTP routes emit spans automatically.
        excluded_urls: Comma-separated URL substrings to skip (health probes,
                       metrics endpoints). Defaults to "health,metrics".

    Returns:
        True if tracing was initialized, False if already initialized or
        OTEL_EXPORTER_OTLP_ENDPOINT is unset.
    """
    global _is_tracing_initialized

    if _is_tracing_initialized:
        logger.debug("configure_otel: already initialized, skipping")
        return False

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        logger.debug("configure_otel: OTEL_EXPORTER_OTLP_ENDPOINT not set, tracing disabled")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)

        # Flush in-flight spans on clean process exit
        atexit.register(provider.shutdown)

        # Auto-instrument FastAPI routes if app is provided
        if app is not None:
            try:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

                FastAPIInstrumentor().instrument_app(
                    app,
                    excluded_urls=excluded_urls,
                )
                logger.info(
                    "configure_otel: FastAPIInstrumentor applied",
                    extra={
                        "service_name": service_name,
                        "excluded_urls": excluded_urls,
                    },
                )
            except ImportError:
                logger.warning(
                    "configure_otel: opentelemetry-instrumentation-fastapi not installed; "
                    "HTTP spans will not be emitted automatically"
                )

        _is_tracing_initialized = True
        logger.info(
            "configure_otel: tracing initialized",
            extra={
                "service_name": service_name,
                "otlp_endpoint": otlp_endpoint,
            },
        )
        return True

    except ImportError as exc:
        logger.warning(
            "configure_otel: opentelemetry SDK not available — tracing disabled",
            extra={"error": str(exc)},
        )
        return False
