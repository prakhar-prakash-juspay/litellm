"""
Conversation Memory Logger with Postgres Backend and GCS Backup
Implements three-table architecture for conversation tracking with hash-based merging
"""

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase
from litellm._logging import verbose_logger
import litellm
import json
import time
import uuid
import os
import hashlib
import asyncio
import asyncpg
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import quote
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class ConversationMemoryManager:
    """Manages conversation history with Postgres backend and hash-based merging"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
        self.gcs_backup_interval = int(os.getenv("GCS_BACKUP_INTERVAL_SECONDS", "900"))
        self.conversation_expiry_days = int(os.getenv("CONVERSATION_EXPIRY_DAYS", "7"))
        self.max_hash_depth = int(os.getenv("MAX_HASH_DEPTH", "5"))
        self._backup_task = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize database pool and create tables"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                self.pool = await asyncpg.create_pool(
                    self.database_url, min_size=2, max_size=10, command_timeout=60
                )

                await self._create_tables()
                self._initialized = True
                verbose_logger.info("✅ ConversationMemoryManager initialized")

                # Start background backup task
                self._backup_task = asyncio.create_task(self._periodic_cleanup())

            except Exception as e:
                verbose_logger.exception(
                    f"❌ Failed to initialize ConversationMemoryManager: {e}"
                )
                raise

    async def _create_tables(self):
        """Create the three required tables"""
        async with self.pool.acquire() as conn:
            # Table 1: Tool-call ID ↔ Conversation ID
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_tool_calls (
                    tool_call_id TEXT PRIMARY KEY,
                    conversation_id UUID NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_used_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """
            )

            # Table 2: Conversation ID ↔ Messages (source of truth)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    conversation_id UUID PRIMARY KEY,
                    messages JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_used_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    last_backed_up_at TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    user_id TEXT,
                    user_email TEXT,
                    assistant_name TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """
            )

            # Table 3: hash(user + LLM response) ↔ Conversation ID
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_hashes (
                    message_hash TEXT PRIMARY KEY,
                    conversation_id UUID NOT NULL,
                    position INTEGER NOT NULL,
                    message_number INTEGER NOT NULL,
                    user_message TEXT,
                    assistant_message TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(conversation_id, position)
                )
            """
            )

            # Create indexes for performance
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tool_call_conversation 
                ON conversation_tool_calls(conversation_id);
            """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_last_used 
                ON conversation_messages(last_used_at);
            """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_hash_conversation 
                ON conversation_hashes(conversation_id);
            """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_user 
                ON conversation_messages(user_id);
            """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_email 
                ON conversation_messages(user_email);
            """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_assistant 
                ON conversation_messages(assistant_name);
            """
            )

            verbose_logger.info("✅ Conversation memory tables created/verified")

    def _compute_message_hash(self, user_msg: str, assistant_msg: str) -> str:
        """Compute SHA256 hash of user + assistant message pair"""
        combined = f"{user_msg}|||{assistant_msg}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    async def _find_conversation_by_hash_matching(
        self,
        messages: List[Dict[str, Any]],
        user_email: Optional[str] = None,
        assistant_name: Optional[str] = None,
    ) -> Optional[Tuple[str, int]]:
        """
        Bottom-up hash matching to find existing conversation.
        Checks from bottom up, moves one level up if hash present, max depth from MAX_HASH_DEPTH env (default 5).
        Uses user_email and assistant_name filters for faster queries.
        Returns (conversation_id, merge_position) if found, None otherwise.
        """
        if not self._initialized or not messages:
            return None

        try:
            # Extract user-assistant pairs from the end (bottom-up)
            pairs = []
            pair_number = 0
            for i in range(len(messages) - 1, 0, -1):
                if (
                    messages[i].get("role") == "assistant"
                    and messages[i - 1].get("role") == "user"
                ):
                    user_msg = str(messages[i - 1].get("content", ""))
                    assistant_msg = str(messages[i].get("content", ""))
                    msg_hash = self._compute_message_hash(user_msg, assistant_msg)
                    pairs.append(
                        (msg_hash, pair_number)
                    )  # Use proper pair_number counter
                    pair_number += 1

            if not pairs:
                return None

            # Limit depth to MAX_HASH_DEPTH (default 5)
            pairs = pairs[: self.max_hash_depth]

            # Try to match hashes from most recent backwards
            async with self.pool.acquire() as conn:
                for msg_hash, offset in pairs:
                    # Build query with optional user_email/assistant_name filters for faster lookups
                    if user_email and assistant_name:
                        result = await conn.fetchrow(
                            """
                            SELECT ch.conversation_id, ch.position
                            FROM conversation_hashes ch
                            JOIN conversation_messages cm ON ch.conversation_id = cm.conversation_id
                            WHERE ch.message_hash = $1 AND cm.user_email = $2 AND cm.assistant_name = $3
                            ORDER BY ch.created_at DESC
                            LIMIT 1
                        """,
                            msg_hash,
                            user_email,
                            assistant_name,
                        )
                    elif user_email:
                        result = await conn.fetchrow(
                            """
                            SELECT ch.conversation_id, ch.position
                            FROM conversation_hashes ch
                            JOIN conversation_messages cm ON ch.conversation_id = cm.conversation_id
                            WHERE ch.message_hash = $1 AND cm.user_email = $2
                            ORDER BY ch.created_at DESC
                            LIMIT 1
                        """,
                            msg_hash,
                            user_email,
                        )
                    else:
                        result = await conn.fetchrow(
                            """
                            SELECT conversation_id, position
                            FROM conversation_hashes
                            WHERE message_hash = $1
                            ORDER BY created_at DESC
                            LIMIT 1
                        """,
                            msg_hash,
                        )

                    if result:
                        conversation_id = str(result["conversation_id"])
                        merge_position = result["position"]
                        verbose_logger.info(
                            f"🔍 Found conversation match: {conversation_id} at position {merge_position} (depth: {pairs.index((msg_hash, offset)) + 1}/{len(pairs)})"
                        )
                        return (conversation_id, merge_position)

        except Exception as e:
            verbose_logger.exception(f"❌ Hash matching error: {e}")

        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((asyncpg.PostgresError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def get_or_create_conversation(
        self,
        tool_call_id: Optional[str],
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        assistant_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get existing conversation or create new one.
        Returns (conversation_id, merged_messages)
        """
        if not self._initialized:
            await self.initialize()

        conversation_id = None
        merged_messages = messages.copy()

        try:
            async with self.pool.acquire() as conn:
                # Step 1: Check if tool_call_id exists
                if tool_call_id:
                    result = await conn.fetchrow(
                        """
                        SELECT conversation_id
                        FROM conversation_tool_calls
                        WHERE tool_call_id = $1
                    """,
                        tool_call_id,
                    )

                    if result:
                        conversation_id = str(result["conversation_id"])
                        verbose_logger.info(
                            f"🔗 Found conversation via tool_call_id: {conversation_id}"
                        )

                # Step 2: Try hash-based matching if no tool_call_id match
                if not conversation_id:
                    hash_match = await self._find_conversation_by_hash_matching(
                        messages, user_email, assistant_name
                    )
                    if hash_match:
                        conversation_id, merge_pos = hash_match
                        # Load existing conversation and merge
                        existing = await conn.fetchrow(
                            """
                            SELECT messages
                            FROM conversation_messages
                            WHERE conversation_id = $1::uuid
                        """,
                            conversation_id,
                        )

                        if existing:
                            existing_messages = existing["messages"]
                            # Parse JSON if it's a string
                            if isinstance(existing_messages, str):
                                existing_messages = json.loads(existing_messages)
                            merged_messages = self._merge_messages(
                                existing_messages, messages, merge_pos
                            )
                            verbose_logger.info(
                                f"🔀 Merged messages at position {merge_pos}"
                            )

                # Step 3: Create new conversation if none found
                if not conversation_id:
                    conversation_id = str(uuid.uuid4())
                    verbose_logger.info(
                        f"🆕 Creating new conversation: {conversation_id}"
                    )

                # Step 4: Store/update conversation (reset last_backed_up_at to NULL on update)
                await conn.execute(
                    """
                    INSERT INTO conversation_messages 
                    (conversation_id, messages, last_used_at, updated_at, last_backed_up_at, message_count, user_id, user_email, assistant_name, metadata)
                    VALUES ($1::uuid, $2::jsonb, NOW(), NOW(), NULL, $3, $4, $5, $6, $7::jsonb)
                    ON CONFLICT (conversation_id) 
                    DO UPDATE SET 
                        messages = $2::jsonb,
                        last_used_at = NOW(),
                        updated_at = NOW(),
                        last_backed_up_at = NULL,
                        message_count = $3,
                        user_email = $5,
                        assistant_name = $6,
                        metadata = $7::jsonb
                """,
                    conversation_id,
                    json.dumps(merged_messages),
                    len(merged_messages),
                    user_id,
                    user_email,
                    assistant_name,
                    json.dumps(metadata or {}),
                )

                # Step 5: Update tool_call mapping if provided
                if tool_call_id:
                    await conn.execute(
                        """
                        INSERT INTO conversation_tool_calls 
                        (tool_call_id, conversation_id, last_used_at, metadata)
                        VALUES ($1, $2::uuid, NOW(), $3::jsonb)
                        ON CONFLICT (tool_call_id) 
                        DO UPDATE SET 
                            last_used_at = NOW(),
                            metadata = $3::jsonb
                    """,
                        tool_call_id,
                        conversation_id,
                        json.dumps(metadata or {}),
                    )

                # Step 6: Update hash mappings
                await self._update_hash_mappings(conn, conversation_id, merged_messages)

        except Exception as e:
            verbose_logger.exception(f"❌ Error in get_or_create_conversation: {e}")
            # Return original messages on error
            conversation_id = str(uuid.uuid4())

        return (conversation_id, merged_messages)

    def _merge_messages(
        self, existing: List[Dict], new: List[Dict], merge_position: int
    ) -> List[Dict]:
        """
        Merge messages using merge_position directly (O(n) performance).
        merge_position indicates where the hash matched in the existing conversation.
        We keep existing messages up to that point and append ONLY NEW messages (not duplicates).
        """
        # Ensure existing and new are lists of dicts
        if isinstance(existing, str):
            existing = json.loads(existing)
        if isinstance(new, str):
            new = json.loads(new)

        # merge_position tells us which user-assistant pair matched
        # Calculate where to cut existing (keep up to and including the matched pair)
        cutoff_index = (merge_position * 2) + 2  # Each pair = 2 messages

        # Find the matched pair in new messages to avoid duplicates
        # Count user-assistant pairs in new until we find the merge_position-th pair
        pair_count = 0
        new_cutoff_index = len(new)  # Default: if not found, append nothing

        for i in range(len(new) - 1):
            if new[i].get("role") == "user" and new[i + 1].get("role") == "assistant":
                if pair_count == merge_position:
                    # Found the matched pair in new - start appending after it
                    new_cutoff_index = i + 2
                    break
                pair_count += 1

        # Keep existing up to matched pair, append only NEW messages after match
        return existing[:cutoff_index] + new[new_cutoff_index:]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((asyncpg.PostgresError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _update_hash_mappings(
        self, conn: asyncpg.Connection, conversation_id: str, messages: List[Dict]
    ):
        """Update hash mappings incrementally - only add new hashes"""
        try:
            # Get the highest existing position to know where to start
            result = await conn.fetchrow(
                """
                SELECT MAX(position) as max_pos
                FROM conversation_hashes
                WHERE conversation_id = $1::uuid
            """,
                conversation_id,
            )

            start_position = (
                (result["max_pos"] + 1)
                if result and result["max_pos"] is not None
                else 0
            )

            # Only create hash mappings for new user-assistant pairs
            position = 0
            for i in range(len(messages) - 1):
                if (
                    messages[i].get("role") == "user"
                    and messages[i + 1].get("role") == "assistant"
                ):
                    # Only insert if this position is new
                    if position >= start_position:
                        user_msg = str(messages[i].get("content", ""))
                        assistant_msg = str(messages[i + 1].get("content", ""))
                        msg_hash = self._compute_message_hash(user_msg, assistant_msg)

                        await conn.execute(
                            """
                            INSERT INTO conversation_hashes 
                            (message_hash, conversation_id, position, message_number, user_message, assistant_message)
                            VALUES ($1, $2::uuid, $3, $4, $5, $6)
                            ON CONFLICT (message_hash) DO NOTHING
                        """,
                            msg_hash,
                            conversation_id,
                            position,
                            i + 1,  # message_number is 1-indexed
                            user_msg[:500],
                            assistant_msg[:500],
                        )

                    position += 1

        except Exception as e:
            verbose_logger.exception(f"❌ Error updating hash mappings: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((asyncpg.PostgresError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def cleanup_expired_conversations(self):
        """Remove conversations that haven't been used in N days"""
        if not self._initialized:
            return

        try:
            expiry_date = datetime.now() - timedelta(days=self.conversation_expiry_days)

            async with self.pool.acquire() as conn:
                # Get expired conversation IDs
                expired = await conn.fetch(
                    """
                    SELECT conversation_id
                    FROM conversation_messages
                    WHERE last_used_at < $1
                """,
                    expiry_date,
                )

                if expired:
                    expired_ids = [str(row["conversation_id"]) for row in expired]

                    # Use transaction for atomic cleanup (all or nothing)
                    async with conn.transaction():
                        # Delete from all three tables in order
                        await conn.execute(
                            """
                            DELETE FROM conversation_hashes
                            WHERE conversation_id = ANY($1::uuid[])
                        """,
                            expired_ids,
                        )

                        await conn.execute(
                            """
                            DELETE FROM conversation_tool_calls
                            WHERE conversation_id = ANY($1::uuid[])
                        """,
                            expired_ids,
                        )

                        await conn.execute(
                            """
                            DELETE FROM conversation_messages
                            WHERE conversation_id = ANY($1::uuid[])
                        """,
                            expired_ids,
                        )

                    verbose_logger.info(
                        f"🗑️  Cleaned up {len(expired_ids)} expired conversations"
                    )

        except Exception as e:
            verbose_logger.exception(f"❌ Error cleaning up conversations: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def backup_to_gcs(
        self,
        gcs_bucket_name: str,
        gcs_base: GCSBucketBase,
        service_account_path: Optional[str],
    ):
        """Backup only unsynced conversations to GCS as individual files and mark them as backed up"""
        if not self._initialized:
            return

        try:
            async with self.pool.acquire() as conn:
                # Get only conversations that need backup (updated since last backup or never backed up)
                cutoff_date = datetime.now() - timedelta(
                    days=self.conversation_expiry_days
                )
                conversations = await conn.fetch(
                    """
                    SELECT conversation_id, messages, created_at, last_used_at, 
                           message_count, user_id, user_email, assistant_name, metadata
                    FROM conversation_messages
                    WHERE last_used_at >= $1
                    AND (last_backed_up_at IS NULL OR updated_at > last_backed_up_at)
                    ORDER BY last_used_at DESC
                """,
                    cutoff_date,
                )

                if not conversations:
                    verbose_logger.info("📦 No active conversations to backup")
                    return

                # Construct request headers once
                headers = await gcs_base.construct_request_headers(
                    service_account_json=service_account_path, vertex_instance=None
                )

                # Upload each conversation as individual file: {timestamp}_{conversation_id}.json
                timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
                backed_up_ids = []

                for conv in conversations:
                    conversation_data = {
                        "conversation_id": str(conv["conversation_id"]),
                        "messages": conv["messages"],
                        "created_at": conv["created_at"].isoformat(),
                        "last_used_at": conv["last_used_at"].isoformat(),
                        "message_count": conv["message_count"],
                        "user_id": conv["user_id"],
                        "user_email": conv["user_email"],
                        "assistant_name": conv["assistant_name"],
                        "metadata": conv["metadata"],
                        "backup_timestamp": datetime.utcnow().isoformat(),
                    }

                    # Upload to GCS with individual file name
                    conv_id = str(conv["conversation_id"])
                    gcs_path = f"conversation_backups/{timestamp}_{conv_id}.json"

                    try:
                        json_data = json.dumps(conversation_data, indent=2, default=str)
                        await gcs_base._log_json_data_on_gcs(
                            headers=headers,
                            bucket_name=gcs_bucket_name,
                            object_name=gcs_path,
                            logging_payload=json_data,
                        )
                        backed_up_ids.append(conv_id)
                        verbose_logger.info(
                            f"📦 Backed up conversation {conv_id} to GCS: {gcs_path}"
                        )
                    except Exception as e:
                        verbose_logger.error(
                            f"❌ Failed to backup conversation {conv_id}: {e}"
                        )
                        continue

                # Mark successfully backed up conversations
                if backed_up_ids:
                    await conn.execute(
                        """
                        UPDATE conversation_messages
                        SET last_backed_up_at = NOW()
                        WHERE conversation_id = ANY($1::uuid[])
                    """,
                        backed_up_ids,
                    )
                    verbose_logger.info(
                        f"✅ Marked {len(backed_up_ids)} conversations as backed up"
                    )

        except Exception as e:
            verbose_logger.exception(f"❌ Error backing up to GCS: {e}")
            raise

    async def _periodic_cleanup(self):
        """Periodically cleanup expired conversations"""
        while True:
            try:
                await asyncio.sleep(self.gcs_backup_interval)
                await self.cleanup_expired_conversations()
                verbose_logger.info("⏰ Periodic cleanup completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                verbose_logger.exception(f"❌ Error in periodic cleanup: {e}")

    async def close(self):
        """Cleanup resources"""
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass

        if self.pool:
            await self.pool.close()

        self._initialized = False
        verbose_logger.info("✅ ConversationMemoryManager closed")


class ConversationMemoryLogger(CustomLogger):
    """
    LiteLLM logger with conversation memory tracking and GCS backup.

    Features:
    - Tracks conversations with tool-call IDs
    - Hash-based message merging for conversation continuity
    - Postgres backend for reliable storage
    - Periodic GCS backups
    - Automatic expiry of old conversations
    """

    def __init__(self):
        super().__init__()
        self.database_url = os.getenv("DATABASE_URL")
        self.gcs_bucket_name = os.getenv(
            "GCS_CONVERSATION_BUCKET_NAME", os.getenv("GCS_SUCCESS_BUCKET_NAME")
        )
        self.service_account_path = os.getenv("GCS_PATH_SERVICE_ACCOUNT")
        self.enable_memory = (
            os.getenv("ENABLE_CONVERSATION_MEMORY", "true").lower() == "true"
        )
        self.enable_gcs_backup = (
            os.getenv("ENABLE_GCS_BACKUP", "true").lower() == "true"
        )

        # Parse allowed assistants list (comma-separated)
        assistants_env = os.getenv("CONVERSATION_MEMORY_ASSISTANTS", "")
        if assistants_env.strip():
            # Normalize to lowercase for case-insensitive matching
            self.allowed_assistants = set(
                name.strip().lower()
                for name in assistants_env.split(",")
                if name.strip()
            )
            verbose_logger.info(
                f"🔍 Conversation memory filtering enabled. "
                f"Tracking only: {', '.join(sorted(self.allowed_assistants))}"
            )
        else:
            # If not set, track all assistants
            self.allowed_assistants = None
            verbose_logger.info(
                "📝 Conversation memory tracking ALL assistants (no filter)"
            )

        # Initialize GCS base
        self.gcs_base = None
        if self.enable_gcs_backup and self.gcs_bucket_name:
            self.gcs_base = GCSBucketBase(bucket_name=self.gcs_bucket_name)
            verbose_logger.info(f"✅ GCS backup enabled: {self.gcs_bucket_name}")

        # Initialize conversation memory manager
        self.memory_manager: Optional[ConversationMemoryManager] = None
        if self.enable_memory and self.database_url:
            self.memory_manager = ConversationMemoryManager(self.database_url)
            verbose_logger.info("✅ Conversation memory enabled")

            # Schedule GCS backups if enabled
            if self.enable_gcs_backup and self.gcs_base:
                asyncio.create_task(self._periodic_gcs_backup())
        else:
            verbose_logger.warning(
                "⚠️  Conversation memory disabled (missing DATABASE_URL or disabled)"
            )

    async def _periodic_gcs_backup(self):
        """Periodically backup conversations to GCS"""
        while True:
            try:
                await asyncio.sleep(self.memory_manager.gcs_backup_interval)
                if self.memory_manager and self.gcs_base:
                    await self.memory_manager.backup_to_gcs(
                        self.gcs_bucket_name, self.gcs_base, self.service_account_path
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                verbose_logger.exception(f"❌ Error in periodic GCS backup: {e}")

    def _extract_tool_call_ids_from_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract all tool_call_ids from messages (supports both OpenAI and Anthropic formats).

        OpenAI format: tool_calls array in assistant messages
        Anthropic format: tool role messages with tool_call_id field
        """
        tool_call_ids = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            # OpenAI format: check assistant messages for tool_calls
            if message.get("role") == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get("id")
                            if tool_call_id and tool_call_id not in tool_call_ids:
                                tool_call_ids.append(tool_call_id)

            # Anthropic format: check tool role messages for tool_call_id
            elif message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and tool_call_id not in tool_call_ids:
                    tool_call_ids.append(tool_call_id)

        return tool_call_ids

    def _extract_tool_call_id(self, kwargs: Dict, response_obj: Any) -> Optional[str]:
        """Extract tool call ID from request or response (supports both OpenAI and Anthropic formats)"""
        # First, try to extract from all messages using inline parser
        messages = kwargs.get("input", kwargs.get("messages", []))
        if messages:
            # Extract tool_call_ids from both OpenAI and Anthropic formats
            tool_call_ids = self._extract_tool_call_ids_from_messages(messages)
            if tool_call_ids:
                # Return the most recent tool_call_id (last one in list)
                return tool_call_ids[-1]

        # Check response for tool calls (OpenAI format)
        if hasattr(response_obj, "choices") and response_obj.choices:
            choice = response_obj.choices[0]
            if hasattr(choice, "message"):
                message = choice.message
                if hasattr(message, "tool_calls") and message.tool_calls:
                    # Return first tool call ID
                    return message.tool_calls[0].id

        # Check kwargs for tool call ID in metadata
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {}) or litellm_params.get(
            "litellm_metadata", {}
        )
        tool_call_id = metadata.get("tool_call_id")

        return tool_call_id

    def _extract_user_id(self, kwargs: Dict) -> Optional[str]:
        """Extract user ID from request metadata"""
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {}) or litellm_params.get(
            "litellm_metadata", {}
        )

        return (
            metadata.get("user_api_key_user_id")
            or metadata.get("user_id")
            or metadata.get("user_api_key_user_email")
        )

    def _extract_user_email(self, kwargs: Dict) -> Optional[str]:
        """Extract user email from metadata"""
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {}) or litellm_params.get(
            "litellm_metadata", {}
        )

        return metadata.get("user_api_key_user_email")

    def _detect_assistant_name(self, kwargs: Dict) -> Optional[str]:
        """Detect which coding assistant/client is making the request (stored as metadata only, not used for filtering)"""
        messages = kwargs.get("input", kwargs.get("messages", []))
        tools = kwargs.get("tools", [])
        model = kwargs.get("model", "").lower()

        # Check model name for Claude/Anthropic (user may switch models mid-conversation)
        if "claude" in model or "anthropic" in model:
            return "Claude"

        # Extract system prompt and convert to lowercase for case-insensitive matching
        system_prompt = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_prompt = content.strip()
                elif isinstance(content, list):
                    system_prompt = "".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict)
                    ).strip()
                break

        # Case-insensitive comparison
        system_prompt_lower = system_prompt.lower()

        # Detect by system prompt patterns (case-insensitive)
        if "you are xyne" in system_prompt_lower:
            return "Xyne"
        if "you are roo" in system_prompt_lower:
            return "Roo Code"
        if "you are cline" in system_prompt_lower:
            return "Cline"
        if "you are kilo" in system_prompt_lower:
            return "Kilo Code"
        if "you are cursor" in system_prompt_lower:
            return "Cursor"
        if (
            "you are claude code" in system_prompt_lower
            or "claude code, anthropic" in system_prompt_lower
        ):
            return "Claude Code"
        if "you are opencode" in system_prompt_lower:
            return "OpenCode"
        if "payment transaction log analysis" in system_prompt_lower:
            return "Payment Log Agent"
        if "configuration analysis agent" in system_prompt_lower:
            return "Configuration Analysis Agent"
        if "sdk log analysis agent" in system_prompt_lower:
            return "SDK Log Analysis Agent"
        if "shopping agent" in system_prompt_lower:
            return "Shopping Agent"
        if "refund transaction log analysis" in system_prompt_lower:
            return "Refund Log Analysis Agent"
        if "support triage assistant" in system_prompt_lower:
            return "Support Triage"
        if "browser agent" in system_prompt_lower:
            return "Browser Agent"
        if "intent classifier" in system_prompt_lower:
            return "Intent Classifier"
        if "root cause analysis" in system_prompt_lower:
            return "RCA Agent"

        # Generic patterns
        if "you are a highly skilled software engineer" in system_prompt_lower:
            return "Generic Coding Assistant"
        if "you are ai assistant" in system_prompt_lower:
            return "Generic AI Assistant"

        return None

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Synchronous success hook (not used)"""
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Synchronous failure hook (not used)"""
        pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log successful requests and track conversation memory"""
        if not self.enable_memory or not self.memory_manager:
            return

        try:
            # Extract messages from request
            messages = kwargs.get("input", kwargs.get("messages", []))
            if not messages:
                return

            # Detect assistant name early to filter
            assistant_name = self._detect_assistant_name(kwargs)

            # Check if this assistant should be tracked
            if self.allowed_assistants is not None:
                # Skip if assistant is unknown OR not in allowed list (case-insensitive)
                if (
                    assistant_name is None
                    or assistant_name.lower() not in self.allowed_assistants
                ):
                    verbose_logger.info(
                        f"⏭️  Skipping conversation memory for: {assistant_name or 'Unknown Assistant'} "
                        f"(allowed: {', '.join(sorted(self.allowed_assistants))})"
                    )
                    return

            # Add assistant response to messages
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                if hasattr(choice, "message"):
                    message = choice.message
                    assistant_message = {
                        "role": "assistant",
                        "content": getattr(message, "content", None),
                    }

                    # Extract reasoning_content if available
                    reasoning_content = getattr(message, "reasoning_content", None)
                    if reasoning_content:
                        assistant_message["reasoning_content"] = reasoning_content

                    # Include tool calls if present
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        assistant_message["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ]

                    messages = messages + [assistant_message]

            # Extract metadata
            tool_call_id = self._extract_tool_call_id(kwargs, response_obj)
            user_id = self._extract_user_id(kwargs)
            user_email = self._extract_user_email(kwargs)

            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or litellm_params.get(
                "litellm_metadata", {}
            )

            request_metadata = {
                "model": kwargs.get("model"),
                "deployment": metadata.get("deployment"),
                "user_email": user_email,
                "assistant_name": assistant_name,
                "team": metadata.get("user_api_key_team_alias"),
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Store conversation
            conversation_id, merged_messages = (
                await self.memory_manager.get_or_create_conversation(
                    tool_call_id=tool_call_id,
                    messages=messages,
                    user_id=user_id,
                    user_email=user_email,
                    assistant_name=assistant_name,
                    metadata=request_metadata,
                )
            )

            verbose_logger.info(
                f"💾 Stored conversation {conversation_id} with {len(merged_messages)} messages"
            )

        except Exception as e:
            verbose_logger.exception(f"❌ Error in async_log_success_event: {e}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failed requests (optional - for debugging)"""
        try:
            # Only log if there's an actual error object
            if response_obj and str(response_obj) != "None":
                error_msg = str(response_obj)[:200]
                verbose_logger.warning(f"❌ Request failed: {error_msg}")

                # Optionally log to database or GCS for failure tracking
                if self.enable_memory and self.memory_manager:
                    # Could store failure information if needed
                    pass
        except Exception as e:
            verbose_logger.exception(f"❌ Error logging failure: {e}")


# Handler instance
logger_instance = ConversationMemoryLogger()


if __name__ == "__main__":
    print("=" * 80)
    print("Conversation Memory Logger with Postgres and GCS")
    print("=" * 80)
    print("\n📝 Features:")
    print("   • Three-table Postgres architecture")
    print("   • Tool-call ID ↔ Conversation ID mapping")
    print("   • Hash-based message merging (depth-limited)")
    print("   • Automatic conversation expiry")
    print("   • Periodic GCS backups (every 15 minutes)")
    print("   • Multi-format tool_call_id extraction (OpenAI & Anthropic)")
    print("\n🔧 Environment Variables:")
    print("   DATABASE_URL - Postgres connection string (required)")
    print("   GCS_CONVERSATION_BUCKET_NAME - GCS bucket for backups")
    print("   GCS_PATH_SERVICE_ACCOUNT - Service account JSON (optional)")
    print("   ENABLE_CONVERSATION_MEMORY - Enable/disable (default: true)")
    print("   ENABLE_GCS_BACKUP - Enable/disable GCS backups (default: true)")
    print("   CONVERSATION_EXPIRY_DAYS - Days before expiry (default: 7)")
    print("   MAX_HASH_DEPTH - Hash matching depth limit (default: 5)")
    print("   GCS_BACKUP_INTERVAL_SECONDS - Backup interval in seconds (default: 900)")
    print(
        "   CONVERSATION_MEMORY_ASSISTANTS - Comma-separated list of assistants to track (default: all)"
    )
    print("\n📝 Config usage:")
    print("litellm_settings:")
    print("  callbacks: logging.conversation_memory_logger.logger_instance")
    print("\n🔍 Tool Call ID Extraction:")
    print("   Supports both OpenAI and Anthropic formats:")
    print("   • OpenAI: tool_calls array in assistant messages")
    print("   • Anthropic: tool role messages with tool_call_id field")
    print("=" * 80)
