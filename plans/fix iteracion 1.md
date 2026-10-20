# ROL
Eres un ingeniero senior trabajando sobre el repo `ai-cascade-proxy` (spec: `ai_cascade_proxy_specification.md`). Vas a remediar huecos detectados en la revisión del diseño, manteniendo TDD estricto (Red → Green → Refactor).

# REGLAS INNEGOCIABLES
- Un test rojo primero, siempre. Muéstrame el fallo antes de implementar.
- Tests unitarios sin llamadas reales a LLM ni a JEV (mocks/fixtures). Los tests contra proveedores reales van en `tests/integration/` con marker `@pytest.mark.integration`, excluidos por defecto.
- No inventes datos de evaluación, precios ni resultados de métricas. Si falta un dato, déjalo como TODO explícito o pídemelo.
- Una fase = un commit (o PR) pequeño. Al terminar cada fase, PARA y resume: qué cambió, qué tests se añadieron, qué decisiones tomaste. No avances sin mi confirmación.
- Todo comportamiento nuevo debe ser configurable por variable de entorno, con valor por defecto seguro, y documentado en `.env.example` y README.
- Los IDs de modelo no se hardcodean: se leen de config y se validan al arrancar (fallo claro si no existen).

# FASE 0: Configuración centralizada de cascada

Objetivo: un único `cascade.yaml` como fuente de verdad de modelos, umbrales y precios. Las variables de entorno quedan SOLO para secretos y para sobrescrituras puntuales de despliegue.

## Esquema (`config/cascade.example.yaml`, sin valores inventados)
```yaml
cascade:
  max_hops: 3                 # tope duro, no configurable por encima de 3
  levels:
    - level: 1
      provider: TODO          # openai | anthropic
      model: TODO             # ID exacto del modelo
      params: { temperature: 0.2, max_tokens: 1024, timeout_ms: 30000 }
      pricing: { input_per_mtok_usd: TODO, output_per_mtok_usd: TODO }
      jev_threshold: 0.80     # umbral de aprobación tras este nivel
    - level: 2
      provider: TODO
      model: TODO
      params: { temperature: 0.2, max_tokens: 2048, timeout_ms: 45000 }
      pricing: { input_per_mtok_usd: TODO, output_per_mtok_usd: TODO }
      jev_threshold: 0.70
    - level: 3                # nivel final: SIN jev_threshold
      provider: TODO
      model: TODO
      params: { temperature: 0.2, max_tokens: 4096, timeout_ms: 90000 }
      pricing: { input_per_mtok_usd: TODO, output_per_mtok_usd: TODO }
jev:
  timeout_ms: 1500
  failure_policy: escalate    # fail_open | escalate | fail_closed
```

## Requisitos
1. Cargar con Pydantic (modelos tipados). Ruta por `CASCADE_CONFIG_PATH`. Los secretos (`*_API_KEY`) solo desde entorno, nunca en el YAML.
2. Sobrescritura por entorno solo para `model` y `jev_threshold` por nivel (`MODEL_LEVEL_1`, `JEV_THRESHOLD_L1`...), con precedencia: entorno > YAML.
3. Validación al arrancar, con error claro y salida inmediata:
   - exactamente 3 niveles, numerados 1..3 sin huecos
   - `jev_threshold` obligatorio en niveles 1 y 2, PROHIBIDO en el 3
   - umbrales en [0,1]
   - ningún valor `TODO` ni precio negativo
   - provider conocido y API key presente para cada provider usado
   - `max_hops` ≤ 3
4. Warning (no error) si el modelo del nivel N+1 es igual al del nivel N, o si el umbral de L2 es mayor que el de L1 (suele indicar un error de configuración).
5. Opcional, detrás de flag `VALIDATE_MODELS_ON_STARTUP=true`: comprobar con el proveedor que el ID existe. Debe poder desactivarse para arrancar sin red.
6. Sin recarga en caliente. Cambiar config = reiniciar. Menos superficie de bugs.
7. Refactoriza el resto: el motor, el cliente JEV y la telemetría leen todo de este objeto de config inyectado, nada hardcodeado. Elimina `JEV_THRESHOLD` global y `pricing.yaml` de las otras fases.
8. Endpoint `GET /v1/models` y la telemetría deben derivarse de esta config.
9. Actualiza `.env.example` y README.

## Tests (TDD, rojo primero)
- carga válida con valores de ejemplo en un fixture
- cada regla de validación del punto 3 falla con mensaje específico (un test por regla)
- precedencia entorno > YAML
- el motor usa el umbral del nivel correcto (test con JEV mockeado devolviendo 0.75: aprueba en L2 y rechaza en L1)
- ningún secreto aparece en logs ni en la representación `repr()` de la config

Para al terminar y resume decisiones. No empieces la Fase 1 sin mi confirmación.

# FASE 1: Arnés de calibración de JEV (prioridad máxima)
Objetivo: medir si JEV acierta, no asumirlo.
1. Crea `eval/` con un esquema de dataset JSONL: `{id, prompt, candidate_response, human_label: "accept"|"reject", source, notes}`.
2. Crea `eval/calibrate.py` que, dado el dataset y un cliente JEV (real o mock), calcule: falsos aceptados, falsos rechazados, precisión/recall de "reject", y la curva umbral→(FA, FR) para barridos de 0.5 a 0.95.
3. Genera un dataset de EJEMPLO sintético mínimo (≤10 filas) marcado claramente como `synthetic_example.jsonl`, solo para probar el arnés. NO lo presentes como dataset real.
4. Haz el umbral configurable (`JEV_THRESHOLD`, hoy hardcodeado a 0.8) y documenta cómo elegirlo desde la curva.
5. Tests: el cálculo de métricas con casos construidos a mano con resultado conocido.
Entregable adicional: una nota en README con el formato que debo usar para etiquetar mis ~100-300 prompts reales.

# FASE 2: Política de fallo de JEV (sustituye el comportamiento del Test 1.3)
1. Enum `JEV_FAILURE_POLICY`: `fail_open` (actual) | `escalate` | `fail_closed`. Por defecto: `escalate`.
2. Si JEV falla (timeout >JEV_TIMEOUT_MS o 5xx): aplica la política. El `jev_score` en telemetría debe ser `null` con `jev_status: "unavailable"`, NUNCA un score inventado.
3. Métrica `jev_fallback_rate` acumulada y expuesta en `/metrics` (formato Prometheus o JSON simple).
4. Tests: una por política, más verificación de que score es null y el contador incrementa.

# FASE 3: Fallos de proveedor
1. Define y documenta en el spec: un 429/5xx/timeout de un modelo de nivel N cuenta como hop consumido y escala a N+1 (reintento acotado con backoff solo para 429, máx. 1).
2. Si falla el nivel 3: devuelve error OpenAI-compatible (HTTP 502/503 con cuerpo `{"error": {...}}`), nunca una respuesta vacía ni un 200.
3. Desactiva los reintentos automáticos ocultos de los SDKs (`max_retries=0`) y gestiona los reintentos tú, de forma contada.
4. Test clave: contador de llamadas SALIENTES totales por petición (no solo hops) con un tope duro; falla si se supera aunque los hops sean ≤3.
5. Tests para: L1 falla→L2 responde; L1 y L2 fallan→L3 responde; L3 falla→error limpio; 429 con un reintento.

# FASE 4: Streaming (`stream: true`)
Decisión de diseño (impleméntala y documéntala): cuando la cascada está activa, se bufferiza la respuesta del nivel, se evalúa, y solo la aprobada se emite como SSE en formato OpenAI (`chat.completion.chunk`, terminando en `data: [DONE]`). Añade `STREAM_MODE`: `buffered_cascade` (defecto) | `passthrough_l3` (stream directo del modelo final, sin evaluación).
Tests: formato exacto de los chunks, `[DONE]`, y que el nivel 3 también se emite correctamente. Documenta el impacto en TTFT.

# FASE 5: Compatibilidad de payload
Detecta `tools`/`tool_choice`, `response_format`, imágenes en `content`, y `n>1`. Comportamiento por defecto: **passthrough directo al nivel configurado en `UNSUPPORTED_FEATURES_LEVEL` sin cascada**, con header de respuesta `X-Cascade-Bypass: <motivo>`. No intentes evaluar con JEV lo que no sabes evaluar. Tests por cada caso.

# FASE 6: Economía y telemetría real
1. Sustituye `estimated_savings_pct` por `cost_usd` real por petición, calculado desde tokens (usage de cada llamada, incluidas las de rechazados y las de JEV si tienen coste) y una tabla de precios configurable en `pricing.yaml` (déjala con valores TODO que yo rellene; no inventes precios).
2. Añade `baseline_cost_usd` (lo que habría costado ir directo a L3) y `net_savings_usd` = baseline − real (puede ser negativo).
3. Telemetría: añade `escalated`, `latency_ms_by_stage`, `outbound_calls`. Por defecto NO loguees prompts ni respuestas (`LOG_CONTENT=false`).
4. Script `eval/breakeven.py`: dada la tabla de precios y una tasa de aceptación de L1, calcula si la cascada compensa y la tasa mínima de equilibrio.

# FASE 7: Modo sombra
`CASCADE_MODE`: `active` | `shadow`. En shadow se ejecuta la cascada completa y se llama también a L3 directo, se devuelve SIEMPRE la respuesta de L3, y se registra para comparación offline (ambas respuestas guardadas en un almacén local solo si `SHADOW_STORE=true`). El coste extra debe quedar visible en telemetría. Tests: la respuesta devuelta es siempre L3; la cascada se ejecuta igualmente.

# FASE 8: Hardening
Autenticación por API key de cliente, rate limiting por clave, y límite de tamaño de payload. Test de concurrencia: 50 peticiones simultáneas sin estado compartido entre ellas (telemetría y contadores correctos).

# FASE 9 (experimento, no cambio de producción)
En `eval/`, añade un flag para comparar contexto de escalado: "limpio" (solo prompt original) vs "con contexto de fallo" (Test 2.4). No cambies el comportamiento por defecto; deja el script listo para ejecutarse sobre el dataset de la Fase 1.

# FUERA DE ALCANCE (no implementar sin pedírmelo)
Caché semántica, caché exacta, y cualquier cambio de proveedor.

# CRITERIO DE FINALIZACIÓN POR FASE
- Todos los tests nuevos y previos en verde (`pytest`, sin marker integration).
- Cobertura de las rutas nuevas ≥ 90%.
- `.env.example`, README y la spec actualizados.
- Resumen al final con: decisiones tomadas, supuestos que hiciste, y cualquier cosa que contradiga la spec original.

Empieza por la Fase 1. Antes de escribir código, dime en 5 líneas máximo tu plan y las dudas que tengas.