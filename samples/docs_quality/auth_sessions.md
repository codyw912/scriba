# Auth and Sessions Guide

## Create Access Token

`POST /v1/auth/tokens`

Exchange a client credential pair for an access token.

- Required headers: `Content-Type: application/json`
- Body fields: `client_id`, `client_secret`
- Success: `201` with `access_token` and `expires_in=900`

## Refresh Access Token

`POST /v1/auth/refresh`

Refresh an expired or nearly-expired access token.

- Required headers: `Authorization: Bearer <refresh_token>`
- Success: `200` with a new `access_token`

## Revoke Session

`POST /v1/sessions/revoke`

Invalidate all active session tokens for the current subject.

- Required header: `Authorization: Bearer <access_token>`
- Idempotency supported via `Idempotency-Key`
- Success: `204`

## Security Notes

- Reject requests missing `Authorization` with `401`.
- Reject stale refresh tokens with `403`.
- Maximum clock skew tolerance is `30` seconds.
