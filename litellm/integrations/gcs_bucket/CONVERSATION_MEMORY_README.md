# Conversation Memory Logger

A powerful conversation tracking system for LiteLLM that maintains conversation continuity across requests using hash-based message merging, PostgreSQL backend, and automatic GCS backups.

## Features

- ✅ **Three-table PostgreSQL architecture** for reliable storage
- 🔗 **Tool-call ID ↔ Conversation ID mapping** for seamless tracking
- 🔄 **Hash-based message merging** with depth-limited matching (default: 5 pairs)
- 🗑️ **Automatic conversation expiry** (default: 7 days)
- 📦 **Periodic GCS backups** (default: every 15 minutes)
- 🔁 **Retry logic** with exponential backoff for resilience
- 🎯 **Assistant filtering** to selectively track specific coding assistants
- 🧠 **Reasoning content extraction** (e.g., from o1 models)
- 🔐 **Multi-format tool_call_id extraction** (supports OpenAI & Anthropic)

## Database Architecture

### Table 1: `conversation_tool_calls`
Maps tool call IDs to conversation IDs for request tracking.

```sql
tool_call_id TEXT PRIMARY KEY
conversation_id UUID NOT NULL
created_at TIMESTAMP
last_used_at TIMESTAMP
metadata JSONB
```

### Table 2: `conversation_messages`
Source of truth for conversation history.

```sql
conversation_id UUID PRIMARY KEY
messages JSONB NOT NULL
created_at TIMESTAMP
last_used_at TIMESTAMP
updated_at TIMESTAMP
last_backed_up_at TIMESTAMP
message_count INTEGER
user_id TEXT
user_email TEXT
assistant_name TEXT
metadata JSONB
```

### Table 3: `conversation_hashes`
Hash mappings for efficient conversation matching.

```sql
message_hash TEXT PRIMARY KEY
conversation_id UUID NOT NULL
position INTEGER NOT NULL
message_number INTEGER NOT NULL
user_message TEXT
assistant_message TEXT
created_at TIMESTAMP
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | ✅ Yes |
| `GCS_CONVERSATION_BUCKET_NAME` | GCS bucket for backups | `GCS_SUCCESS_BUCKET_NAME` | No |
| `GCS_PATH_SERVICE_ACCOUNT` | Path to GCS service account JSON | - | No |
| `ENABLE_CONVERSATION_MEMORY` | Enable/disable conversation tracking | `true` | No |
| `ENABLE_GCS_BACKUP` | Enable/disable GCS backups | `true` | No |
| `CONVERSATION_EXPIRY_DAYS` | Days before conversation expiry | `7` | No |
| `MAX_HASH_DEPTH` | Hash matching depth limit (pairs) | `5` | No |
| `GCS_BACKUP_INTERVAL_SECONDS` | GCS backup interval in seconds | `900` (15 min) | No |
| `CONVERSATION_MEMORY_ASSISTANTS` | Comma-separated list of assistants to track | - (all) | No |

### Setup in LiteLLM Config

Add the callback to your `litellm_config.yaml`:

```yaml
litellm_settings:
  callbacks: 
    - logging.conversation_memory_logger.logger_instance

# Environment variables
environment_variables:
  DATABASE_URL: "postgresql://user:password@host:5432/database"
  GCS_CONVERSATION_BUCKET_NAME: "my-conversation-backups"
  GCS_PATH_SERVICE_ACCOUNT: "/path/to/service-account.json"
  CONVERSATION_EXPIRY_DAYS: "7"
  MAX_HASH_DEPTH: "5"
  GCS_BACKUP_INTERVAL_SECONDS: "900"
  CONVERSATION_MEMORY_ASSISTANTS: "Claude,Xyne,Roo Code"  # Optional: filter specific assistants
```

## How It Works

### 1. Hash-Based Conversation Matching

The system creates SHA256 hashes of user-assistant message pairs:

```
hash = SHA256("user_message|||assistant_message")
```

When a new request arrives:
1. Extract the last N pairs from the message history (default: 5)
2. Check hashes from most recent backwards
3. If a match is found, merge with the existing conversation
4. Reset `last_backed_up_at` to trigger a new backup

### 2. Message Merging Algorithm

```python
# Example: Existing conversation has 10 messages
# New request has 8 messages, with 3rd pair matching
# Result: Keep existing[0:8] + new[6:]
```

The merge preserves all existing messages and only appends truly new messages after the match point.

### 3. Assistant Filtering

Filter which coding assistants to track (case-insensitive):

```yaml
CONVERSATION_MEMORY_ASSISTANTS: "Claude,Xyne,Roo Code"
```

Detected assistants:
- Claude / Claude Code
- Xyne
- Roo Code
- Cline
- Kilo Code
- Cursor
- OpenCode
- Generic Coding Assistant
- Custom agents (via system prompt detection)

### 4. Retry Logic

All database and GCS operations use retry logic:
- **Attempts**: 3 retries
- **Wait**: Exponential backoff (1-10 seconds)
- **Exceptions**: PostgresError, TimeoutError, general exceptions

### 5. GCS Backup Format

Individual files per conversation:

```
conversation_backups/2026-01-05_14-30-00_{conversation_id}.json
```

Each backup includes:
```json
{
  "conversation_id": "uuid",
  "messages": [...],
  "created_at": "timestamp",
  "last_used_at": "timestamp",
  "message_count": 42,
  "user_id": "user123",
  "user_email": "user@example.com",
  "assistant_name": "Claude",
  "metadata": {...},
  "backup_timestamp": "timestamp"
}
```

## Usage Examples

### Basic Setup

```bash
# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost:5432/litellm"
export GCS_CONVERSATION_BUCKET_NAME="my-backups"
export CONVERSATION_EXPIRY_DAYS="14"
```

### Track Specific Assistants Only

```bash
# Only track Claude, Xyne, and Roo Code conversations
export CONVERSATION_MEMORY_ASSISTANTS="Claude,Xyne,Roo Code"
```

### Disable GCS Backup

```bash
export ENABLE_GCS_BACKUP="false"
```

### Adjust Hash Matching Depth

```bash
# Check last 10 message pairs instead of 5
export MAX_HASH_DEPTH="10"
```

## Tool Call ID Extraction

The system automatically extracts tool call IDs from:

### OpenAI Format
```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {...}
    }
  ]
}
```

### Anthropic Format
```json
{
  "role": "tool",
  "tool_call_id": "toolu_xyz789",
  "content": "..."
}
```

## Reasoning Content Support

Automatically extracts and stores `reasoning_content` from models like o1:

```json
{
  "role": "assistant",
  "content": "The answer is 42",
  "reasoning_content": "First, I analyzed the question..."
}
```

**Note**: `reasoning_content` is stored but **NOT** used for hash computation, ensuring conversation continuity even when reasoning changes.

## Monitoring

### Logs

The logger provides detailed logging with emojis for easy identification:

```
✅ ConversationMemoryManager initialized
🔍 Found conversation match: uuid at position 2 (depth: 1/5)
🔀 Merged messages at position 2
💾 Stored conversation uuid with 42 messages
📦 Backed up conversation uuid to GCS
🗑️ Cleaned up 5 expired conversations
⏭️ Skipping conversation memory for: Unknown Assistant
```

### Database Queries

Check conversation count:
```sql
SELECT COUNT(*) FROM conversation_messages;
```

Check hash mappings:
```sql
SELECT conversation_id, COUNT(*) as hash_count
FROM conversation_hashes
GROUP BY conversation_id
ORDER BY hash_count DESC;
```

Find conversations by email:
```sql
SELECT conversation_id, message_count, assistant_name, last_used_at
FROM conversation_messages
WHERE user_email = 'user@example.com'
ORDER BY last_used_at DESC;
```

## Performance Considerations

- **Connection Pool**: 2-10 connections (configurable in code)
- **Command Timeout**: 60 seconds
- **Hash Depth**: Limit to 5-10 pairs for optimal performance
- **Backup Interval**: Adjust based on usage (default: 15 minutes)
- **Indexes**: Created on `user_email`, `assistant_name`, `conversation_id`, `last_used_at`

## Troubleshooting

### Issue: Conversations not merging

**Solution**: Check `MAX_HASH_DEPTH` - increase if conversations are longer

### Issue: Too many backups

**Solution**: Increase `GCS_BACKUP_INTERVAL_SECONDS` or adjust `CONVERSATION_EXPIRY_DAYS`

### Issue: Assistant not tracked

**Solution**: Check `CONVERSATION_MEMORY_ASSISTANTS` - ensure assistant name matches (case-insensitive)

### Issue: Database connection errors

**Solution**: 
- Verify `DATABASE_URL` format
- Check PostgreSQL is running
- Ensure database exists and user has permissions
- Check retry logic in logs (3 attempts with exponential backoff)

### Issue: GCS upload failures

**Solution**:
- Verify `GCS_CONVERSATION_BUCKET_NAME` exists
- Check `GCS_PATH_SERVICE_ACCOUNT` path and permissions
- Ensure service account has write access to bucket
- Check retry logic in logs

## Migration from Other Systems

If migrating from a different conversation tracking system:

1. Export existing conversations to JSON format
2. Transform to match the schema (see GCS backup format above)
3. Import directly into `conversation_messages` table
4. Run hash generation script to populate `conversation_hashes`

## Dependencies

Install via `logging/requirements.txt`:

```
google-cloud-storage
asyncpg
tenacity
```

## Security Notes

- Store `DATABASE_URL` and `GCS_PATH_SERVICE_ACCOUNT` securely
- Use environment variables, not hardcoded values
- Restrict database user permissions to minimum required
- Use GCS service account with least privilege (write-only to backup bucket)
- Enable SSL for PostgreSQL connections in production

## License

Part of LiteLLM - MIT License
