"""
Kibana Logger Integration - sends logs to Kibana/Elasticsearch endpoint

This logger sends LLM request/response logs to a Kibana-compatible endpoint.
Similar to the reference implementation, it provides batching with configurable
batch size and flush intervals.

`async_log_success_event` - used by litellm proxy to send logs to Kibana
`log_success_event` - sync version of logging to Kibana

For batching specific details see CustomBatchLogger class
"""

import asyncio
import os
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload


# Default configuration (matching reference implementation)
KIBANA_DEFAULT_BATCH_SIZE = 20
KIBANA_DEFAULT_FLUSH_INTERVAL = 20  # seconds
KIBANA_DEFAULT_ENDPOINT = "https://analytics.xyne.juspay.in/xyne-prod-logs"


class KibanaLogger(CustomBatchLogger):
    """
    Logger that sends LLM request/response data to a Kibana/Elasticsearch endpoint.

    Logs include top-level fields for easy filtering in Kibana:
    - sessionId: from litellm_session_id (trace_id)
    - userEmail: from API key metadata
    - userId: from API key metadata
    - teamId: from API key metadata

    Environment variables:
        KIBANA_ENDPOINT - The endpoint to send logs to
                         (default: https://analytics.xyne.juspay.in/xyne-prod-logs)
        KIBANA_BATCH_SIZE - Number of events per batch (default: 20)
        KIBANA_FLUSH_INTERVAL - Flush interval in seconds (default: 20)
        KIBANA_API_KEY - API key for authentication (required)
    """

    def __init__(self, **kwargs):
        """
        Initialize the Kibana logger.
        """
        try:
            verbose_logger.debug("KibanaLogger: Initializing Kibana logger")

            # Get configuration from environment
            self.endpoint = os.getenv("KIBANA_ENDPOINT", KIBANA_DEFAULT_ENDPOINT)
            self.api_key = os.getenv("KIBANA_API_KEY")

            # API key is required
            if not self.api_key:
                raise ValueError(
                    "KIBANA_API_KEY is not set. Set 'KIBANA_API_KEY=<your-api-key>' in environment variables."
                )

            batch_size = int(
                os.getenv("KIBANA_BATCH_SIZE", str(KIBANA_DEFAULT_BATCH_SIZE))
            )
            flush_interval = int(
                os.getenv("KIBANA_FLUSH_INTERVAL", str(KIBANA_DEFAULT_FLUSH_INTERVAL))
            )

            # Initialize HTTP client
            self.async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )

            # Initialize flush lock and start periodic flush
            self.flush_lock = asyncio.Lock()
            asyncio.create_task(self.periodic_flush())

            super().__init__(
                **kwargs,
                flush_lock=self.flush_lock,
                batch_size=batch_size,
                flush_interval=flush_interval,
            )

            verbose_logger.debug(
                f"KibanaLogger: Initialized with endpoint={self.endpoint}, "
                f"batch_size={batch_size}, flush_interval={flush_interval}"
            )

        except Exception as e:
            verbose_logger.exception(
                f"KibanaLogger: Error initializing Kibana logger: {str(e)}"
            )
            raise e

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        """
        Async log success events to Kibana.

        Creates a Kibana payload and adds it to the in-memory queue.
        Payload is flushed based on batch size or flush interval.
        """
        try:
            verbose_logger.debug("KibanaLogger: Logging success event")
            await self._log_event(
                kwargs, response_obj, start_time, end_time, status="success"
            )
        except Exception as e:
            verbose_logger.exception(
                f"KibanaLogger: Error logging success event - {str(e)}\n{traceback.format_exc()}"
            )

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        """
        Async log failure events to Kibana.
        """
        try:
            verbose_logger.debug("KibanaLogger: Logging failure event")
            await self._log_event(
                kwargs, response_obj, start_time, end_time, status="failure"
            )
        except Exception as e:
            verbose_logger.exception(
                f"KibanaLogger: Error logging failure event - {str(e)}\n{traceback.format_exc()}"
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Sync log success events to Kibana (immediate send, no batching).
        """
        try:
            verbose_logger.debug("KibanaLogger: Logging success event (sync)")
            payload = self._create_kibana_payload(
                kwargs, response_obj, start_time, end_time, status="success"
            )

            headers = self._get_headers()

            # Use sync HTTP client
            with httpx.Client() as client:
                response = client.post(
                    url=self.endpoint,
                    json=[payload],
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()

            verbose_logger.debug(
                f"KibanaLogger: Successfully sent log, status_code={response.status_code}"
            )
        except Exception as e:
            verbose_logger.exception(
                f"KibanaLogger: Error in sync log - {str(e)}\n{traceback.format_exc()}"
            )

    async def _log_event(
        self,
        kwargs: Dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
        status: str,
    ):
        """
        Internal method to create payload and add to queue.
        """
        payload = self._create_kibana_payload(
            kwargs, response_obj, start_time, end_time, status
        )
        self.log_queue.append(payload)

        verbose_logger.debug(
            f"KibanaLogger: Event added to queue. Queue size: {len(self.log_queue)}. "
            f"Will flush in {self.flush_interval} seconds or at batch size {self.batch_size}"
        )

        # Check if we should flush immediately
        if len(self.log_queue) >= self.batch_size:
            await self.flush_queue()

    def _create_kibana_payload(
        self,
        kwargs: Dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
        status: str,
    ) -> Dict:
        """
        Create the payload to send to Kibana.

        Includes top-level fields for easy Kibana filtering:
        - sessionId (from litellm_session_id/trace_id)
        - userEmail (from user_api_key_user_email)
        - userId (from user_api_key_user_id)
        - teamId (from user_api_key_team_id)
        """
        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )

        # Generate unique event ID
        event_id = f"evt_{int(datetime.now().timestamp() * 1000)}_{uuid.uuid4().hex[:8]}"

        # Build the payload
        payload: Dict[str, Any] = {
            "eventId": event_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "eventType": "llm_request",
            "status": status,
        }

        if standard_logging_object:
            # Extract metadata for top-level filter fields
            metadata = standard_logging_object.get("metadata", {}) or {}

            # TOP-LEVEL FILTER FIELDS for easy Kibana queries
            payload["sessionId"] = standard_logging_object.get("trace_id", "")
            payload["userEmail"] = metadata.get("user_api_key_user_email", "")
            payload["userId"] = metadata.get("user_api_key_user_id", "")
            payload["teamId"] = metadata.get("user_api_key_team_id", "")

            # Request details
            payload["requestId"] = standard_logging_object.get("id", "")
            payload["traceId"] = standard_logging_object.get("trace_id", "")
            payload["callType"] = standard_logging_object.get("call_type", "")
            payload["model"] = standard_logging_object.get("model", "")
            payload["modelId"] = standard_logging_object.get("model_id", "")
            payload["apiBase"] = standard_logging_object.get("api_base", "")
            payload["stream"] = standard_logging_object.get("stream", False)
            payload["cacheHit"] = standard_logging_object.get("cache_hit", False)

            # Cost & tokens
            payload["cost"] = standard_logging_object.get("response_cost", 0)
            payload["tokens"] = {
                "total": standard_logging_object.get("total_tokens", 0),
                "prompt": standard_logging_object.get("prompt_tokens", 0),
                "completion": standard_logging_object.get("completion_tokens", 0),
            }
            payload["responseTime"] = standard_logging_object.get("response_time", 0)
            payload["startTime"] = standard_logging_object.get("startTime")
            payload["endTime"] = standard_logging_object.get("endTime")

            # Additional user info
            payload["endUser"] = standard_logging_object.get("end_user", "")
            payload["requestTags"] = standard_logging_object.get("request_tags", [])

            # Request/Response content
            messages = standard_logging_object.get("messages")
            if messages:
                payload["messages"] = messages

            response = standard_logging_object.get("response")
            if response:
                payload["response"] = response

            # Full metadata (sanitized)
            payload["metadata"] = self._sanitize_metadata(metadata)

            # Error info if failure
            if status == "failure":
                payload["error"] = {
                    "message": standard_logging_object.get("error_str", ""),
                    "details": standard_logging_object.get("error_information", {}),
                }
        else:
            # Fallback to extracting from kwargs directly
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or {}

            # Top-level filter fields from metadata
            payload["sessionId"] = metadata.get("litellm_session_id", "")
            payload["userEmail"] = metadata.get("user_api_key_user_email", "")
            payload["userId"] = metadata.get("user_api_key_user_id", "")
            payload["teamId"] = metadata.get("user_api_key_team_id", "")

            payload["model"] = kwargs.get("model", "")
            payload["messages"] = kwargs.get("messages", [])
            payload["user"] = kwargs.get("user", "")
            payload["metadata"] = self._sanitize_metadata(metadata)

            # Add response if available
            if response_obj:
                try:
                    payload["response"] = (
                        dict(response_obj)
                        if hasattr(response_obj, "__dict__")
                        else str(response_obj)
                    )
                except Exception:
                    payload["response"] = str(response_obj)

            # Calculate response time
            try:
                response_time = (end_time - start_time).total_seconds() * 1000
                payload["responseTime"] = response_time
            except Exception:
                pass

        # Add system metadata
        payload["systemMetadata"] = {
            "hostname": os.getenv("HOSTNAME", ""),
            "podName": os.getenv("POD_NAME", ""),
            "service": os.getenv("SERVICE_NAME", "litellm"),
            "environment": os.getenv("ENVIRONMENT", "production"),
        }

        return payload

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """
        Sanitize metadata to remove sensitive fields.
        """
        if not isinstance(metadata, dict):
            return {}

        sensitive_keys = [
            "api_key",
            "apikey",
            "secret",
            "password",
            "token",
            "credential",
            "authorization",
            "auth",
            "private",
        ]

        # Internal litellm metadata to skip
        internal_keys = ["endpoint", "caching_groups", "previous_models"]

        clean_metadata = {}
        for key, value in metadata.items():
            # Skip internal litellm metadata
            if key in internal_keys:
                continue
            # Skip sensitive keys
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                continue
            clean_metadata[key] = value

        return clean_metadata

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for the HTTP request.
        """
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    async def async_send_batch(self):
        """
        Send the batched logs to Kibana endpoint.
        """
        try:
            if not self.log_queue:
                verbose_logger.debug("KibanaLogger: No events in queue to send")
                return

            verbose_logger.debug(
                f"KibanaLogger: Sending batch of {len(self.log_queue)} events to {self.endpoint}"
            )

            headers = self._get_headers()

            # Send the batch
            response = await self.async_client.post(
                url=self.endpoint,
                json=self.log_queue,
                headers=headers,
                timeout=30.0,
            )

            response.raise_for_status()

            verbose_logger.debug(
                f"KibanaLogger: Successfully sent batch, status_code={response.status_code}"
            )

        except Exception as e:
            verbose_logger.exception(
                f"KibanaLogger: Error sending batch - {str(e)}\n{traceback.format_exc()}"
            )
