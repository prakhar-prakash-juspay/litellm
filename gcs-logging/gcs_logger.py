"""
Production Logger with GCS Support for LiteLLM Proxy Server
Logs to local files and separate GCS buckets for success/error events
"""

from litellm.integrations.custom_logger import CustomLogger
import litellm
import json
import time
import uuid
import os
from datetime import datetime


class ProductionGCSLogger(CustomLogger):
    """Production logger with local file and GCS bucket support"""

    def __init__(self):
        super().__init__()
        self.gcs_client = None
        self.success_bucket = None
        self.error_bucket = None
        self._initialize_gcs()

    def _initialize_gcs(self):
        """Initialize GCS client and buckets"""
        try:
            from google.cloud import storage

            success_bucket_name = os.getenv("GCS_SUCCESS_BUCKET_NAME")
            error_bucket_name = os.getenv("GCS_ERROR_BUCKET_NAME")
            service_account_path = os.getenv("GCS_PATH_SERVICE_ACCOUNT")

            if not success_bucket_name or not error_bucket_name:
                print("⚠️  GCS bucket names not set. GCS logging disabled.")
                return

            if service_account_path and os.path.exists(service_account_path):
                self.gcs_client = storage.Client.from_service_account_json(
                    service_account_path
                )
            else:
                self.gcs_client = storage.Client()

            self.success_bucket = self.gcs_client.bucket(success_bucket_name)
            self.error_bucket = self.gcs_client.bucket(error_bucket_name)
            print(f"✅ GCS initialized: {success_bucket_name}, {error_bucket_name}")

        except ImportError:
            print("⚠️  google-cloud-storage not installed")
            self.gcs_client = None
        except Exception as e:
            print(f"❌ GCS initialization error: {e}")
            self.gcs_client = None

    def _upload_to_gcs(self, data: dict, bucket, log_type: str):
        """Upload log data to GCS bucket"""
        if not self.gcs_client or not bucket:
            return

        try:
            date = datetime.utcnow().strftime("%Y-%m-%d")
            correlation_id = data.get("correlation_id", str(uuid.uuid4()))

            if log_type == "success":
                # Success logs: department/team/user/{date}_{correlation_id}.json
                user_data = data.get("user", {})
                department = user_data.get("department") or "unknown_dept"
                team = user_data.get("team_alias") or "unknown_team"
                user_email = user_data.get("email", "unknown")
                username = user_email.split("@")[0] if user_email else "unknown"

                # Sanitize folder names (remove special characters)
                department = department.replace("/", "_").replace(" ", "_")
                team = team.replace("/", "_").replace(" ", "_")
                username = username.replace("/", "_").replace(" ", "_")

                filename = f"{date}_{correlation_id}.json"
                gcs_path = f"{department}/{team}/{username}/{filename}"
            else:
                # Error logs: model/{date}_{correlation_id}.json
                model_data = data.get("model", {})
                model_name = (
                    model_data.get("requested")
                    or model_data.get("deployment")
                    or "unknown_model"
                )

                # Sanitize model name
                model_name = model_name.replace("/", "_").replace(" ", "_")

                filename = f"{date}_{correlation_id}.json"
                gcs_path = f"{model_name}/{filename}"

            blob = bucket.blob(gcs_path)
            blob.upload_from_string(
                json.dumps(data, indent=2, default=str), content_type="application/json"
            )

        except Exception as e:
            print(f"❌ GCS upload error: {e}")

    def log_pre_api_call(self, model, messages, kwargs):
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log successful requests for LLM training history"""
        try:
            correlation_id = getattr(response_obj, "id", None) or str(uuid.uuid4())
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {})

            success_log = {
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "timestamp_iso": datetime.utcnow().isoformat(),
                "type": "SUCCESS",
                "user": {
                    "email": metadata.get("user_api_key_user_email"),
                    "user_id": metadata.get("user_api_key_user_id"),
                    "team_alias": metadata.get("user_api_key_team_alias"),
                    "department": metadata.get("user_api_key_metadata", {}).get(
                        "department"
                    ),
                },
                "model": {
                    "requested": kwargs.get("model"),
                    "used": getattr(response_obj, "model", None),
                    "deployment": metadata.get("deployment"),
                    "model_group": metadata.get("model_group"),
                    "mode": metadata.get("model_info", {}).get("mode"),
                },
                "conversation": {
                    "messages": kwargs.get("messages", []),
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "top_p": kwargs.get("top_p"),
                    "frequency_penalty": kwargs.get("frequency_penalty"),
                    "presence_penalty": kwargs.get("presence_penalty"),
                    "tools": kwargs.get("tools"),
                    "tool_choice": kwargs.get("tool_choice"),
                },
                "response": {},
                "usage": {},
                "cost": 0,
                "timing": {
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "duration_seconds": (
                        (end_time - start_time).total_seconds()
                        if start_time and end_time
                        else None
                    ),
                    "llm_api_duration_ms": metadata.get("llm_api_duration_ms"),
                },
            }

            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                success_log["response"] = {
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "content": None,
                    "tool_calls": None,
                    "function_call": None,
                }

                if hasattr(choice, "message"):
                    message = choice.message
                    success_log["response"]["content"] = getattr(
                        message, "content", None
                    )
                    success_log["response"]["tool_calls"] = getattr(
                        message, "tool_calls", None
                    )
                    success_log["response"]["function_call"] = getattr(
                        message, "function_call", None
                    )

            if hasattr(response_obj, "usage"):
                usage = response_obj.usage
                success_log["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }

            try:
                success_log["cost"] = litellm.completion_cost(
                    completion_response=response_obj
                )
            except Exception:
                success_log["cost"] = 0

            self._upload_to_gcs(success_log, self.success_bucket, "success")

        except Exception as e:
            print(f"Error logging success: {e}")
            import traceback

            traceback.print_exc()

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failed requests for debugging"""
        try:
            correlation_id = str(uuid.uuid4())
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {})

            error_log = {
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "timestamp_iso": datetime.utcnow().isoformat(),
                "type": "ERROR",
                "user": {
                    "email": metadata.get("user_api_key_user_email"),
                    "user_id": metadata.get("user_api_key_user_id"),
                    "team_alias": metadata.get("user_api_key_team_alias"),
                    "department": metadata.get("user_api_key_metadata", {}).get(
                        "department"
                    ),
                },
                "model": {
                    "requested": kwargs.get("model"),
                    "deployment": metadata.get("deployment"),
                    "model_group": metadata.get("model_group"),
                    "api_base": litellm_params.get("api_base"),
                    "provider": litellm_params.get("custom_llm_provider"),
                },
                "request": {
                    "messages_count": len(kwargs.get("messages", [])),
                    "first_message": (
                        kwargs.get("messages", [{}])[0].get("content", "")[:100]
                        if kwargs.get("messages")
                        else None
                    ),
                    "max_tokens": kwargs.get("max_tokens"),
                    "route": metadata.get("user_api_key_request_route"),
                },
                "error": {
                    "type": type(response_obj).__name__,
                    "message": str(response_obj),
                    "exception": str(kwargs.get("exception", "")),
                    "traceback": str(kwargs.get("traceback_exception", "")),
                },
                "timing": {
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "duration_seconds": (
                        (end_time - start_time).total_seconds()
                        if start_time and end_time
                        else None
                    ),
                    "llm_api_duration_ms": metadata.get("llm_api_duration_ms"),
                },
            }

            self._upload_to_gcs(error_log, self.error_bucket, "error")

        except Exception as e:
            print(f"Error logging failure: {e}")
            import traceback

            traceback.print_exc()


# Handler instance
logger_instance = ProductionGCSLogger()


if __name__ == "__main__":
    print("=" * 80)
    print("Production Logger with GCS Support")
    print("=" * 80)
    print("\n📝 Logs to:")
    print("   • GCS_SUCCESS_BUCKET_NAME (cloud)")
    print("   • GCS_ERROR_BUCKET_NAME (cloud)")
    print("\n🔧 Environment Variables:")
    print("   GCS_SUCCESS_BUCKET_NAME - Success logs bucket")
    print("   GCS_ERROR_BUCKET_NAME - Error logs bucket")
    print("   GCS_PATH_SERVICE_ACCOUNT - Service account JSON (optional)")
    print("\n📝 Config usage:")
    print("litellm_settings:")
    print("  callbacks: gcs_logger.logger_instance")
    print("=" * 80)
