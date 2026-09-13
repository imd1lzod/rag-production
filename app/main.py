import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv


from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
)

from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.agent import ProductionAgent

load_dotenv()
logger = get_logger()

security: SecurityPipeline = None
cache: ResponseCache = None
metrics: MetricsCollector = None
agent: ProductionAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global security, cache, metrics, agent

    settings = get_settings()

    logger.info(
        "Starting up the application...",
        extra={
            "extra_data": {
                "primary_model": settings.primary_model,
                "tracing_enabled": settings.langchain_tracing,
            }
        },
    )

    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()

    logger.info("Application started successfully.")

    yield

    logger.info(
        "Shutting down the application...",
        extra={"extra_data": {"environment": metrics.summary}},
    )


limiter = Limiter(
    key_func=get_remote_address, default_limits=[get_settings().ratelimit]
)


app = FastAPI(
    title="RAG Production API",
    description="A FastAPI application for a Retrieval-Augmented Generation (RAG) system with caching, rate limiting, and monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(
        "Rate limit exceeded",
        extra={
            "extra_data": {"client_ip": request.client.host, "path": request.url.path}
        },
    )
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().ratelimit)
@traceable
async def chat(request: Request, body: ChatRequest):
    """
    Flow
    1. Security check (injection + PII masking)
    2. Cache lookup
    3. Langgraph agent invoke
    4. Ouput validation
    5. Cache store
    6. Return response
    """

    with RequestTimer() as timer:
        security_notes = []

        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning(
                "Request blocked due to security policy",
                extra={
                    "extra_data": {
                        "reason": notes,
                        "thread_id": body.thread_id,
                    }
                },
            )
            metrics.record_request(latency_ms=0, error=True, cache_hit=False, input_tokens=0, output_tokens=0)
            raise HTTPException(
                status_code=400, detail="Request blocked due to security policy"
            )

        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(
                latency_ms=0,
                cache_hit=True,
                input_tokens=0,
                output_tokens=0,
            )

            logger.info(
                "Cache hit for request",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                        "cached": True,
                    }
                },
            )
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time=0,
            )


        try:
            result = agent.invoke(cleaned_message)

        except Exception as e:
            logger.error(f"Agent invocation failed: {str(e)}", extra={"extra_data": {"thread_id": body.thread_id, "error": str(e)}})

            metrics.record_request(latency_ms=0, error=True, cache_hit=False, input_tokens=0, output_tokens=0)

            raise HTTPException(
                status_code=500, detail="An error occurred while processing your request. Please try again later."
            )

        response_text = result["response"]
        model_used = result["model_used"]

        validated_response, validation_warnings = security.check_output(response_text)
        security_notes.extend(validation_warnings)

        cache.set(cleaned_message, validated_response)

    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_time_ms,
        cache_hit=False,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    if security_notes:
        logger.info(
            "Security notes for request",
            extra={
                "extra_data": {
                    "thread_id": body.thread_id,
                    "security_notes": security_notes,
                }
            },
        ) 

    logger.info(
        "Request processed successfully",
        extra={
            "extra_data": {
                "thread_id": body.thread_id,
                "model_used": model_used,
                "cached": False,
                "processing_time_ms": round(timer.elapsed_time_ms, 2),
            }
        },
    )

    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        processing_time=round(timer.elapsed_time_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "unhealthy",
        version="1.0.0",
        checks=checks,
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    summary = metrics.summary
    print(f"Metrics summary: {summary}")  # Debugging line to print metrics summary

@app.get("/cache/stats")
async def cache_stats():
    stats = cache.stats
    return JSONResponse(content=stats)