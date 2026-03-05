# Service Guide

## Health Check

GET /v1/health

Returns status code 200 when the service is healthy.

## Create Session

POST /v1/sessions

Creates a temporary session with `ttl_seconds=3600`.

## Rate Limits

The API allows up to 120 requests per minute.
