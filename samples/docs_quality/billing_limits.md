# Billing and Limits Reference

## Retrieve Usage Summary

`GET /v1/billing/usage`

Returns usage totals grouped by model and day.

- Query params: `start_date`, `end_date`, `group_by`
- Success: `200`

## Retrieve Invoice Preview

`GET /v1/billing/invoice-preview`

Returns estimated charges for the current billing period.

- Includes `subtotal_usd`, `credits_usd`, `total_usd`
- Estimates may drift by up to `0.5%`

## Retrieve Rate Limits

`GET /v1/limits`

Returns current request and token ceilings.

- Default tenant cap: `600 requests per minute`
- Burst cap: `12000 tokens per minute`

## Pricing Notes

- Usage values are rounded to `4` decimal places.
- Invoices are finalized at `00:00 UTC` on month boundary.
