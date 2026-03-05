# Projects and Events API

## List Projects

`GET /v1/projects`

Returns projects visible to the caller.

- Query params: `limit` (default `25`, max `100`), `cursor`, `status`
- Sort order: `created_at desc`

## List Project Events

`GET /v1/projects/{project_id}/events`

Returns event timeline for one project.

- Query params: `since`, `until`, `type`, `cursor`
- Success: `200` with `next_cursor`

## Export Audit Log

`GET /v1/audit/logs`

Returns immutable audit records.

- Query params: `actor_id`, `resource`, `cursor`, `limit`
- Export window limit: `31` days per request

## Pagination Rules

- Cursor tokens are opaque strings.
- Clients must not assume cursor sort semantics.
- `next_cursor` is omitted when no further page exists.
