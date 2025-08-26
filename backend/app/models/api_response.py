from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any, Generic, TypeVar
from datetime import datetime
from enum import Enum

# Generic type for response data
T = TypeVar('T')

class ResponseStatus(str, Enum):
    """API response status types."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"

class ErrorCode(str, Enum):
    """Standard error codes."""
    # Authentication errors
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    
    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    
    # Resource errors
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    
    # Service errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    
    # Business logic errors
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    SUBSCRIPTION_REQUIRED = "SUBSCRIPTION_REQUIRED"
    API_LIMIT_EXCEEDED = "API_LIMIT_EXCEEDED"
    MARKET_CLOSED = "MARKET_CLOSED"
    
    # Data errors
    DATA_NOT_AVAILABLE = "DATA_NOT_AVAILABLE"
    STALE_DATA = "STALE_DATA"
    PROCESSING_ERROR = "PROCESSING_ERROR"

class APIError(BaseModel):
    """API error model."""
    code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    field: Optional[str] = Field(None, description="Field that caused the error (for validation errors)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

class PaginationInfo(BaseModel):
    """Pagination information."""
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=1000, description="Number of items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")

class APIResponse(BaseModel, Generic[T]):
    """Generic API response model."""
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    errors: Optional[List[APIError]] = Field(None, description="List of errors")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    processing_time: Optional[float] = Field(None, ge=0, description="Processing time in seconds")
    
    # Pagination (for list responses)
    pagination: Optional[PaginationInfo] = Field(None, description="Pagination information")
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional response metadata")

class SuccessResponse(APIResponse[T]):
    """Success response model."""
    status: ResponseStatus = Field(default=ResponseStatus.SUCCESS, description="Response status")
    
class ErrorResponse(APIResponse[None]):
    """Error response model."""
    status: ResponseStatus = Field(default=ResponseStatus.ERROR, description="Response status")
    data: None = Field(default=None, description="No data for error responses")
    errors: List[APIError] = Field(..., description="List of errors")

class ValidationErrorResponse(ErrorResponse):
    """Validation error response model."""
    def __init__(self, validation_errors: List[Dict[str, Any]], **kwargs):
        errors = []
        for error in validation_errors:
            errors.append(APIError(
                code=ErrorCode.VALIDATION_ERROR,
                message=error.get('msg', 'Validation error'),
                field='.'.join(str(loc) for loc in error.get('loc', [])),
                details=error
            ))
        super().__init__(errors=errors, message="Validation failed", **kwargs)

class PaginatedResponse(APIResponse[List[T]]):
    """Paginated response model."""
    pagination: PaginationInfo = Field(..., description="Pagination information")
    
class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    
    # Service dependencies
    database: str = Field(..., description="Database connection status")
    redis: str = Field(..., description="Redis connection status")
    external_apis: Dict[str, str] = Field(..., description="External API status")
    
    # Performance metrics
    uptime: float = Field(..., ge=0, description="Service uptime in seconds")
    memory_usage: float = Field(..., ge=0, description="Memory usage percentage")
    cpu_usage: float = Field(..., ge=0, description="CPU usage percentage")
    
class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    channel: Optional[str] = Field(None, description="Message channel")
    user_id: Optional[str] = Field(None, description="Target user ID")
    
class WebSocketResponse(BaseModel):
    """WebSocket response model."""
    type: str = Field(..., description="Response type")
    status: ResponseStatus = Field(..., description="Response status")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[APIError] = Field(None, description="Error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
class BatchRequest(BaseModel):
    """Batch request model."""
    requests: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100, description="List of requests")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    
class BatchResponse(BaseModel):
    """Batch response model."""
    batch_id: str = Field(..., description="Batch identifier")
    total_requests: int = Field(..., ge=0, description="Total number of requests")
    successful_requests: int = Field(..., ge=0, description="Number of successful requests")
    failed_requests: int = Field(..., ge=0, description="Number of failed requests")
    
    responses: List[APIResponse] = Field(..., description="Individual responses")
    
    # Batch metadata
    started_at: datetime = Field(..., description="Batch processing start time")
    completed_at: datetime = Field(..., description="Batch processing completion time")
    processing_time: float = Field(..., ge=0, description="Total processing time in seconds")
    
class RateLimitInfo(BaseModel):
    """Rate limit information."""
    limit: int = Field(..., ge=0, description="Rate limit")
    remaining: int = Field(..., ge=0, description="Remaining requests")
    reset_at: datetime = Field(..., description="Rate limit reset time")
    retry_after: Optional[int] = Field(None, ge=0, description="Retry after seconds")
    
class APIMetrics(BaseModel):
    """API metrics model."""
    endpoint: str = Field(..., description="API endpoint")
    method: str = Field(..., description="HTTP method")
    
    # Request metrics
    total_requests: int = Field(..., ge=0, description="Total requests")
    successful_requests: int = Field(..., ge=0, description="Successful requests")
    failed_requests: int = Field(..., ge=0, description="Failed requests")
    
    # Performance metrics
    average_response_time: float = Field(..., ge=0, description="Average response time in seconds")
    min_response_time: float = Field(..., ge=0, description="Minimum response time in seconds")
    max_response_time: float = Field(..., ge=0, description="Maximum response time in seconds")
    
    # Error metrics
    error_rate: float = Field(..., ge=0, le=1, description="Error rate (0 to 1)")
    most_common_errors: List[Dict[str, Union[str, int]]] = Field(..., description="Most common error codes")
    
    # Time period
    period_start: datetime = Field(..., description="Metrics period start")
    period_end: datetime = Field(..., description="Metrics period end")
    
class CacheInfo(BaseModel):
    """Cache information model."""
    key: str = Field(..., description="Cache key")
    hit: bool = Field(..., description="Whether it was a cache hit")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    size: Optional[int] = Field(None, description="Cache entry size in bytes")
    
class APIResponseWithCache(APIResponse[T]):
    """API response with cache information."""
    cache_info: Optional[CacheInfo] = Field(None, description="Cache information")
    
class StreamingResponse(BaseModel):
    """Streaming response model."""
    stream_id: str = Field(..., description="Stream identifier")
    event_type: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    sequence: int = Field(..., ge=0, description="Event sequence number")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    
class FileUploadResponse(BaseModel):
    """File upload response model."""
    file_id: str = Field(..., description="Uploaded file identifier")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., ge=0, description="File size in bytes")
    content_type: str = Field(..., description="File content type")
    upload_url: Optional[str] = Field(None, description="File access URL")
    
    # Upload metadata
    uploaded_at: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    expires_at: Optional[datetime] = Field(None, description="File expiration timestamp")
    
class ExportResponse(BaseModel):
    """Data export response model."""
    export_id: str = Field(..., description="Export identifier")
    format: str = Field(..., description="Export format (csv, json, xlsx)")
    status: str = Field(..., description="Export status")
    
    # Export details
    total_records: int = Field(..., ge=0, description="Total records exported")
    file_size: Optional[int] = Field(None, ge=0, description="Export file size in bytes")
    download_url: Optional[str] = Field(None, description="Download URL")
    
    # Timestamps
    requested_at: datetime = Field(..., description="Export request timestamp")
    completed_at: Optional[datetime] = Field(None, description="Export completion timestamp")
    expires_at: Optional[datetime] = Field(None, description="Download link expiration")
    
# Utility functions for creating standard responses

def create_success_response(
    data: T = None,
    message: str = "Success",
    metadata: Optional[Dict[str, Any]] = None,
    pagination: Optional[PaginationInfo] = None
) -> APIResponse[T]:
    """Create a success response."""
    return APIResponse(
        status=ResponseStatus.SUCCESS,
        message=message,
        data=data,
        metadata=metadata,
        pagination=pagination
    )

def create_error_response(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    field: Optional[str] = None
) -> ErrorResponse:
    """Create an error response."""
    error = APIError(
        code=error_code,
        message=message,
        details=details,
        field=field
    )
    return ErrorResponse(
        message=message,
        errors=[error]
    )

def create_validation_error_response(
    validation_errors: List[Dict[str, Any]]
) -> ValidationErrorResponse:
    """Create a validation error response."""
    return ValidationErrorResponse(validation_errors)

def create_paginated_response(
    data: List[T],
    pagination: PaginationInfo,
    message: str = "Success"
) -> PaginatedResponse[T]:
    """Create a paginated response."""
    return PaginatedResponse(
        status=ResponseStatus.SUCCESS,
        message=message,
        data=data,
        pagination=pagination
    )