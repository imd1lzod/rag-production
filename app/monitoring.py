import logging
import time
import json
from datetime import datetime, timezone
from typing import Any, Callable
from functools import wraps


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line_no": record.lineno,
        }

        if hasattr(record, "extra_data"):
            log_record["extra_data"] = record.extra_data

        return json.dumps(log_record)


def get_logger(
    name: str = "production-api", level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class MetricsCollector:

    def __init__(self):
        self.requests_total = 0
        self.errors_total = 0
        self.cache_hits_total = 0
        self.cache_misses_total = 0
        self.latency_sum = 0.0
        self.latency_count = 0
        self.input_tokens_total = 0
        self.output_tokens_total = 0

    def record_request(
        self,
        latency_ms: float,
        cache_hit: bool,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
    ):
        self.requests_total += 1
        self.latency_sum += latency_ms
        self.latency_count += 1
        self.input_tokens_total += input_tokens
        self.output_tokens_total += output_tokens

        if error:
            self.errors_total += 1
        if cache_hit:
            self.cache_hits_total += 1
        else:
            self.cache_misses_total += 1

    @property
    def summary(self) -> dict[str, Any]:
        avg_latency = (
            self.latency_sum / self.latency_count if self.latency_count > 0 else 0
        )

        error_rate = (
            (self.errors_total / self.requests_total) * 100
            if self.requests_total > 0
            else 0
        )

        cache_total = self.cache_hits_total + self.cache_misses_total

        cache_hit_rate = (
            (self.cache_hits_total / cache_total) * 100 if cache_total > 0 else 0
        )

        return {
            "total_requests": self.requests_total,
            "total_errors": self.errors_total,
            "error_rate": f"{error_rate:.2f}%",
            "cache_hits": self.cache_hits_total,
            "cache_misses": self.cache_misses_total,
            "cache_hit_rate": f"{cache_hit_rate:.2f}%"  ,
            "avg_latency_ms": avg_latency,
            "total_input_tokens": self.input_tokens_total,
            "total_output_tokens": self.output_tokens_total,
        }


class RequestTimer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed_time_ms = (self.end_time - self.start_time) * 1000