from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):

    message: str = Field(..., 
                min_length=1,
                max_length=1000,
                description="The user's message to the agent"
                )

    thread_id: str = Field(
        default="default",
        description="Conversation thread ID to maintain context across multiple messages"
    )

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time: float
    timestamp: str = Field(default_factory=lambda: datetime.now())

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    checks: dict = {}

class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    error_rate: str
    avg_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int

class ErrorResponse(BaseModel):
    error: str
    details: str | None = None
    request_id: str | None = None


