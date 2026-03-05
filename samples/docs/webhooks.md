# Webhooks API

## Register Webhook

POST /v1/webhooks

Registers a callback URL for event delivery.

## List Webhooks

GET /v1/webhooks

Returns configured webhook endpoints.

## Delivery Events

GET /v1/webhooks/deliveries

Each event may retry up to 3 times.

Use header `X-Signature-256` to verify payload integrity.
