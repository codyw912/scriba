# Error Handling and Retry Catalog

## Replay Event

`POST /v1/events/replay`

Replay a previously processed event.

- Required body fields: `event_id`, `target`
- Success: `202`

## Fetch Delivery Status

`GET /v1/webhooks/deliveries/{delivery_id}`

Returns status and delivery attempts.

- Includes `attempt_count`, `last_error`, `next_retry_at`

## Register Webhook

`POST /v1/webhooks`

Registers a callback URL and signing secret.

- Signature header: `X-Signature-256`
- Retries use exponential backoff: `5s`, `30s`, `120s`
- Maximum retries: `4`

## Error Codes

- `400` malformed request payload
- `409` idempotency conflict
- `429` rate limit exceeded
