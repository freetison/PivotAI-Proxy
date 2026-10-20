# TECHNICAL SPECIFICATION & TDD DEVELOPMENT PLAN
## AI Cascade Proxy Server with JEV Evaluator

---

## 1. Development Philosophy & Constraints

* **Methodology:** Strict TDD (Red $\rightarrow$ Green $\rightarrow$ Refactor). Write the failing unit or integration test first, then implement the minimal code needed to pass.
* **Core Principles:** Atomic, isolated, measurable, and deterministic tests.
* **Anti-Loop Protection (Circuit Breaker):** The system MUST strictly enforce a hard limit on iteration hops. Maximum 3 execution attempts. It can never fall into an infinite fallback loop.
* **Isolation Strategy:** Unit tests MUST NOT make live API calls to LLM providers or JEV. Use strict mocking and fixtures. Live tests live in `tests/integration/`, carry `@pytest.mark.integration`, and are excluded from the default run by `addopts = "-m 'not integration'"`; they additionally require `CASCADE_RUN_INTEGRATION=1`, because keys in the environment are not consent to spend money. They skip (never fail) when the configuration is incomplete.

---

## 2. System Architecture

A transparent Proxy Server compatible with the OpenAI API specification (`/v1/chat/completions`). It runs a multi-tier cascade of models evaluated by JEV (TypeSafe AI) at each step.

```text
[ Client Application ] ──> [ Proxy: /v1/chat/completions ]
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │   1. Model Level 1   │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │ 2. JEV Evaluator     │ ──(Approved)──> [ Return Response ]
                        └──────────┬───────────┘
                                   │ (Rejected)
                                   ▼
                        ┌──────────────────────┐
                        │   3. Model Level 2   │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │ 4. JEV Evaluator     │ ──(Approved)──> [ Return Response ]
                        └──────────┬───────────┘
                                   │ (Rejected)
                                   ▼
                        ┌──────────────────────┐
                        │ 5. Model Level 3     │ ──(Final) ────> [ Return Response ]
                        └──────────────────────┘
```

---

## 3. Test Plan & TDD Phases

### Phase 1: JEV Evaluator (Atomic Unit)
- [x] **Test 1.1 (`test_jev_approves_valid_response`):**  
  *Given* a valid model output, *when* JEV evaluates it, *then* it returns `True` (Score $\ge 0.8$).
- [x] **Test 1.2 (`test_jev_rejects_invalid_response`):**  
  *Given* an incomplete or inaccurate model output, *when* JEV evaluates it, *then* it returns `False` (Score $< 0.8$).
- [x] **Test 1.3 (`test_jev_timeout_is_unavailable_not_approval`):**  
  *Given* a network timeout ($> 1500\text{ms}$) or $5xx$ error from JEV API, *when* evaluating, *then* no score is produced (`jev_score = null`, `jev_status = "unavailable"`) and `jev.failure_policy` decides the outcome: `escalate` (default) treats the candidate as unapproved and tries the next level, `fail_open` returns it as approved, `fail_closed` returns HTTP 503. A missing measurement is never turned into a number, and no policy silently answers without recording that JEV was unavailable.

### Phase 2: Cascade Engine & Anti-Loop (Core Logic)
- [x] **Test 2.1 (`test_cascade_stops_at_level_1_if_jev_passes`):**  
  *Given* Model Level 1 generates an acceptable response, *when* JEV approves, *then* the pipeline halts immediately. Model Level 2 and Level 3 MUST NOT be called.
- [x] **Test 2.2 (`test_cascade_escalates_to_level_2_if_level_1_fails`):**  
  *Given* JEV rejects Model Level 1 response, *when* executing pipeline, *then* Model Level 2 is invoked and re-evaluated.
- [x] **Test 2.3 (`test_anti_loop_hard_cap`):**  
  *Given* Model Level 1 and Level 2 both fail JEV evaluation, *when* reaching Level 3 (Frontier Model), *then* the response is returned directly to the client without calling JEV, enforcing a hard iteration limit of $N=3$.
- [x] **Test 2.4 (`test_cascade_context_preservation`):**  
  *Given* an escalation step, *when* passing payload to the next model level, *then* the user's original prompt and prior failure context are correctly formatted.

### Phase 3 (extra): Provider failures and the outbound call budget
- [x] **Test F3.1 (`test_level_1_failure_escalates_and_level_2_answers`):**
  *Given* a 5xx from level 1, *when* the cascade runs, *then* the hop is consumed and level 2 answers the request; no JEV call is made for the failed level.
- [x] **Test F3.2 (`test_the_last_level_failing_raises_instead_of_returning_an_empty_answer`):**
  *Given* every level failing, *when* the cascade runs, *then* it raises a cascade error and the API returns HTTP 502 with an OpenAI-shaped body (`code: "cascade_failed"`), never a 200 and never an empty `choices` array.
- [x] **Test F3.3 (`test_a_429_is_retried_once_on_the_same_level`):**
  *Given* a 429 followed by success, *when* the cascade runs, *then* the same level is called twice with a backoff and the request succeeds; a second 429 consumes the hop instead of retrying again.
- [x] **Test F3.4 (`test_the_outbound_call_cap_aborts_even_when_hops_stay_within_three`):**
  *Given* every level returning 429, *when* the outbound call cap is reached, *then* the request aborts with `outbound_call_limit` before the (cap + 1)-th call, even though only three hops were used.

### Phase 4 (extra): Streaming
- [x] **Test F4.1 (`test_the_buffered_stream_emits_role_content_finish_and_done`):**
  *Given* an approved answer, *when* it is streamed, *then* the frames are `chat.completion.chunk` (role delta, content delta, empty delta with `finish_reason`, `data: [DONE]`) sharing one `id`/`created`, and the whole answer arrives only after the cascade finished.
- [x] **Test F4.2 (`test_passthrough_forwards_the_upstream_frames_verbatim`):**
  *Given* `STREAM_MODE=passthrough_l3` and an OpenAI-compatible final level, *when* streaming, *then* the upstream frames are forwarded byte for byte, `[DONE]` is added if the provider omitted it, and no JEV evaluation happens.
- [x] **Test F4.3 (`test_a_failing_cascade_still_returns_a_json_error_not_a_broken_stream`):**
  *Given* a failing cascade in buffered mode, *when* the client asked for a stream, *then* it receives a normal JSON error instead of a truncated stream.
- [x] **Test F4.4 (`test_passthrough_refuses_a_final_level_that_cannot_stream_in_openai_format`):**
  *Given* a final level that cannot emit OpenAI frames, *when* passthrough is requested, *then* the request fails fast with `400 streaming_unsupported` rather than being silently translated.

### Phase 5 (extra): Payload compatibility
- [x] **Test F5.1 (`test_each_unsupported_feature_bypasses_the_cascade`):**
  *Given* `tools`, `tool_choice`, `response_format`, image content or `n > 1`, *when* the request is served, *then* exactly one configured level answers it, JEV is never called, and the response carries `X-Cascade-Bypass: <reason>`.
- [x] **Test F5.2 (`test_several_reasons_are_joined_in_the_header`):**
  *Given* a payload with several unsupported features, *when* it is bypassed, *then* the reasons appear comma-joined in a stable order.
- [x] **Test F5.3 (`test_a_normal_request_has_no_bypass_header`):**
  *Given* a plain text request, *when* served, *then* no bypass header is returned and the cascade starts at level 1.
- [x] **Test F5.4 (`test_a_failed_bypass_does_not_escalate`):**
  *Given* a failing bypassed level, *when* the request fails, *then* the provider error is returned without escalating to another tier.

### Phase 6 (extra): Real economics and telemetry
- [x] **Test F6.1 (`test_cost_comes_from_reported_tokens_and_configured_prices`):**
  *Given* a request whose candidates were rejected and replaced, *when* telemetry is emitted, *then* `cost_usd` sums every call made, `baseline_cost_usd` is the cost of one direct final-level call, and `net_savings_usd = baseline − real` (negative allowed).
- [x] **Test F6.2 (`test_a_partial_table_is_refused_instead_of_mixing_sources`):**
  *Given* a `pricing.yaml` that does not price every configured level, *when* the proxy starts, *then* it exits before binding the port and names the missing key.
- [x] **Test F6.3 (`test_a_breakeven_acceptance_is_found_when_one_exists`):**
  *Given* per-level costs and an intermediate acceptance rate, *when* `eval/breakeven.py` runs, *then* it reports the minimum first-tier acceptance rate at which the cascade costs the same as answering directly, or states that the cascade always/never pays off.

### Phase 7 (extra): Shadow mode
- [x] **Test F7.1 (`test_the_answer_is_the_final_level_even_when_the_cascade_approved`):**
  *Given* `CASCADE_MODE=shadow` and a level-1 answer JEV approves, *when* the request is served, *then* the client receives a direct final-level answer and the cascade's answer is kept for comparison.
- [x] **Test F7.2 (`test_the_cascade_still_runs`):**
  *Given* `CASCADE_MODE=shadow`, *when* a request is served, *then* every configured level is still called and JEV is still consulted — the experiment must observe the real cascade, not a simulated one.
- [x] **Test F7.3 (`test_the_extra_cost_is_charged_and_visible`):**
  *Given* a shadowed request, *when* telemetry is emitted, *then* `cost_usd` includes both the cascade and the direct call, and `shadow_cost_usd` isolates the extra spend.
- [x] **Test F7.4 (`test_a_failed_cascade_does_not_hide_the_shadow_answer`):**
  *Given* a cascade that fails in shadow mode, *when* the direct final call succeeds, *then* the request still returns that answer and the failure appears only as `shadow_failed_levels`.
- [x] **Test F7.5 (`test_nothing_is_written_when_the_store_is_off`):**
  *Given* `SHADOW_STORE=false` (the default), *when* a shadowed request completes, *then* no prompt or answer reaches disk; only `SHADOW_STORE=true` appends one JSON line per request to `SHADOW_STORE_PATH`.

### Phase 10 (extra): JEV through OpenRouter, with the client's instructions
- [x] **Test F10.1 (`test_the_dialect_defaults_to_the_existing_simple_contract`):**
  *Given* a configuration that names no dialect, *then* `jev.dialect` is `simple` and the evaluator posts `{"prompt","response"}` exactly as before — the new transport is opt-in.
- [x] **Test F10.2 (`test_the_system_prompt_is_sent_to_jev`):**
  *Given* a request carrying `role: "system"` turns and `jev.dialect: typesafe_decisions`, *when* the candidate is evaluated, *then* the evaluator receives those turns joined in order as the state's `system`, so an answer that ignored them can be rejected.
- [x] **Test F10.3 (`test_no_system_prompt_is_an_empty_string_not_a_missing_field`):**
  *Given* a request without system turns, *then* the dialect sends `system: ""` rather than omitting the field; the type is stable across requests.
- [x] **Test F10.4 (`test_the_score_is_the_minimum_answer`):**
  *Given* named questions answering `0.99` and `0.1`, *then* the score is `0.1`: each question is a requirement, so adding one can only make approval harder.
- [x] **Test F10.5 (`test_typesafe_decisions_without_a_model_is_refused_at_startup`):**
  *Given* `dialect: typesafe_decisions` with no `jev.model`, *then* startup fails with a specific message instead of every request failing at the API.
- [x] **Test F10.6 (`test_an_unknown_dialect_is_refused_at_startup`):** an unknown `jev.dialect` (or `JEV_DIALECT`) is a configuration error naming the accepted values.
- [x] **Test F10.7 (`test_the_verification_cost_is_part_of_the_request_cost`):**
  *Given* a JEV reply reporting `usage.cost`, *then* `cost_usd` includes it and `jev_cost_usd` reports it separately — savings computed without the cost of deciding are flattering.
- [x] **Test F10.8 (`test_an_unavailable_jev_call_adds_no_cost`):** a verification that produced no score is never billed.
- [x] **Test F10.9 (`test_the_scorer_receives_the_system_prompt`):**
  *Given* a calibration row with a `system` field, *then* the harness scores it with that system prompt, and `_cache_key` distinguishes it from the same prompt and answer without one.
- [x] **Test F10.10 (`test_fail_closed_still_refuses_when_the_sdk_cannot_answer`):**
  *Given* the System One transport returning an error, *then* each `jev.failure_policy` behaves exactly as it does for the legacy dialect — the policy lives above the transport, not inside it.

### Phase 8 (extra): Hardening
- [x] **Test F8.1 (`test_several_keys_are_all_accepted`):**
  *Given* `API_KEYS` with several `label:key` entries, *when* a client presents one of them, *then* it is accepted, and any other value is refused with `401 authentication_error`. With no keys configured, every caller is `anonymous` and no header is required.
- [x] **Test F8.2 (`test_the_limit_is_per_key_not_global`):**
  *Given* `RATE_LIMIT_REQUESTS=5`, *when* one key sends 20 requests and a second key sends 20 more, *then* each key gets exactly 5 successful requests, and the rest receive `429` with `code: "rate_limit_exceeded"` and a `Retry-After` header. `/health` is never limited.
- [x] **Test F8.3 (`test_an_oversized_content_length_is_refused_without_buffering_it`):**
  *Given* `MAX_PAYLOAD_BYTES`, *when* a body exceeds it, *then* the request is refused with `413 payload_too_large` before the JSON is parsed, and a misleading or absent `Content-Length` cannot bypass the check.
- [x] **Test F8.4 (`test_fifty_concurrent_requests_all_succeed_with_consistent_counters`):**
  *Given* fifty simultaneous requests, *when* they all complete, *then* every one succeeds, `requests_total == 50`, and fifty distinct telemetry events are emitted — no request state is shared between them.
- [x] **Test F8.5 (`test_fifty_concurrent_requests_against_a_limit_let_exactly_the_limit_through`):**
  *Given* a limit of 10 and fifty simultaneous requests from one key, *then* exactly ten succeed, forty are refused, and `rate_limited_total == 40`.

### Phase 9 (experiment): Escalation context
Offline only (`eval/escalation_context.py`); no production behaviour changes and no new environment variable.
- [x] **Test F9.1 (`test_clean_context_sends_only_the_original_prompt`):**
  *Given* the `clean` arm, *when* an escalation is built, *then* the request is the original prompt alone.
- [x] **Test F9.2 (`test_the_failure_note_is_the_production_one_not_a_copy`):**
  *Given* the `failure_context` arm, *then* the note is `src.cascade_engine.failure_context(evaluation)` verbatim, so the experiment cannot drift from what the proxy sends.
- [x] **Test F9.3 (`test_the_two_arms_differ_only_in_the_context`):**
  *Given* the same escalated row, *when* both arms run, *then* they call the same level with the same parameters and the same first message, and differ only in the candidate and the note.
- [x] **Test F9.4 (`test_an_approved_candidate_does_not_escalate_and_costs_nothing`):**
  *Given* a candidate the evaluator already approves, *when* the experiment runs, *then* the row is excluded and no provider call is made.
- [x] **Test F9.5 (`test_an_empty_escalation_set_produces_no_numbers`):**
  *Given* a dataset where nothing escalates, *when* the report is rendered, *then* it says there is nothing to compare instead of printing percentages, and every rate is `None` rather than `0.0`.
- [x] **Test F9.6 (`test_a_failing_arm_is_recorded_and_never_counted_as_approved`):**
  *Given* a provider failure in one arm, *when* the report is rendered, *then* the failure is listed with the row id and the arm's approval count is unaffected.

### Phase 3: Proxy Server & OpenAI API Layer
- [x] **Test 3.1 (`test_openai_valid_payload_structure`):**  
  *Given* a `POST /v1/chat/completions` request, *when* processing completes, *then* it returns HTTP 200 with an OpenAI-compliant JSON payload (`id`, `object`, `created`, `model`, `choices`).
- [x] **Test 3.2 (`test_models_endpoint_derives_from_config`):**  
  *Given* a `GET /v1/models` request, *when* queried, *then* it returns HTTP 200 with the supported models list.
- [x] **Test 3.3 (`test_telemetry_and_cost_logging`):**  
  *Given* any processed request, *when* returning response, *then* a structured JSON log entry is emitted containing: `model_used`, `hops_count`, `jev_score`, `jev_status`, `total_latency_ms`, `cost_usd`, `baseline_cost_usd`, `net_savings_usd`, `outbound_calls` and `failed_levels`.

---

## 4. Proposed Project Structure

```text
ai-cascade-proxy/
├── .env.example
├── pyproject.toml / package.json
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py             # FastAPI / Web Server entrypoint
│   ├── cascade_engine.py   # Core logic & circuit breaker
│   ├── jev_evaluator.py    # JEV API client & fallback logic
│   └── telemetry.py        # Metrics & cost tracking
└── tests/
    ├── conftest.py          # Mocks & fixtures
    ├── test_jev.py          # Phase 1 tests
    ├── test_cascade.py      # Phase 2 tests
    └── test_proxy_api.py    # Phase 3 tests
```

---

## 5. Required Environment Variables (`.env.example`)

```bash
# Server Configuration
PORT=8000
HOST=0.0.0.0

# JEV Evaluator
JEV_API_KEY=your_jev_api_key_here
JEV_TIMEOUT_MS=1500

# Model Cascade Mapping
MODEL_LEVEL_1=openai/gpt-4o-mini
MODEL_LEVEL_2=anthropic/claude-3-5-haiku
MODEL_LEVEL_3=anthropic/claude-3-5-sonnet

# Provider API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

Superseded: model ids, JEV thresholds and prices moved to `config/cascade.yaml`, and the
runtime flags below were added after this section was written. `.env.example` is the
source of truth.

```bash
# Cascade Configuration
CASCADE_CONFIG_PATH=config/cascade.yaml
PRICING_PATH=config/pricing.yaml          # optional price overlay

# Runtime behavior
MAX_OUTBOUND_CALLS=6                      # hard cap, max 6
STREAM_MODE=buffered_cascade              # or passthrough_l3
UNSUPPORTED_FEATURES_LEVEL=3
JEV_FAILURE_POLICY=escalate               # fail_open | escalate | fail_closed
LOG_CONTENT=false
VALIDATE_MODELS_ON_STARTUP=false

# Client authentication and abuse limits
API_KEYS=team-search:key-one,team-batch:key-two  # overrides API_KEY; labels must be unique
RATE_LIMIT_REQUESTS=0                     # 0 disables; per key label, fixed window
RATE_LIMIT_WINDOW_SECONDS=60
MAX_PAYLOAD_BYTES=1048576                 # 413 above this, checked before parsing

# Shadow mode
CASCADE_MODE=active                       # or shadow
SHADOW_STORE=false
SHADOW_STORE_PATH=data/shadow_pairs.jsonl
```
### Test suite contract
- [x] **Test S.1 (`test_the_integration_marker_is_registered`):**
  *Given* `pyproject.toml`, *then* `integration` is a registered marker with a description, so `pytest --markers` documents it and no run emits an unknown-marker warning.
- [x] **Test S.2 (`test_integration_tests_are_excluded_by_default`):**
  *Given* the default `pytest` invocation, *then* `addopts` carries `-m 'not integration'` and `tests/integration` is deselected.
- [x] **Test S.3 (`test_integration_tests_are_skipped_rather_than_failed_without_credentials`):**
  *Given* no opt-in and no credentials, *when* the live suite is collected and run, *then* every test **skips** and none passes, so a machine that exercised nothing never reports green.
- [x] **Test S.4 (`test_the_integration_suite_is_reachable_on_demand`):**
  *Given* `-m integration --override-ini=addopts=`, *then* the live tests are collected — the exclusion is a filter, not a missing directory.
- [x] **Test S.5 (`test_every_checked_box_names_a_test_that_exists`):**
  *Given* this document, *then* every checked box names a test that exists in `tests/`, so the checklist cannot claim credit for a name nobody can find; it skips where `plans/` is absent.
