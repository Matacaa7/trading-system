# Retomar aquí — continuación del refactor

## Contexto para abrir nueva conversación con Claude

Este fichero es un resumen ejecutivo para que, en una nueva conversación, Claude
pueda continuar sin releer los 4 meses de contexto anterior.

---

## Qué es este proyecto

Sistema de trading algorítmico con ML sobre equities US intraday (1-min).
Proyecto **personal** (NO es un TFM). Usuario: Javier (Matacaa7).

Empezó como 3 repos separados:
- signals-historical (pipeline batch semanal)
- trading-engine (live con Alpaca paper)
- ml-sandbox (training/backtest, 6 modelos en ensemble)

Se está migrando a **monorepo único** (esta carpeta `trading-system/`).

---

## Estado al cerrar la conversación anterior

### Completado ✅

1. **Análisis de viabilidad completo** de los 3 repos: 135 issues identificados,
   documentados en `informe_viabilidad.docx` (entregado en Fase 2).

2. **7 decisiones arquitectónicas tomadas**:
   - Monorepo con `shared/` (no 3 repos separados)
   - F-39 (ciclo de vida de trades) antes del frontend
   - Alpaca como broker (Premium cuando toque)
   - Refactor completo de paridad training/live con `shared/indicators.py`
   - Frontend en Streamlit
   - Mantener los 6 modelos del ensemble
   - Informe personal, no compartido

3. **Bloque 2 — Seguridad completado**:
   - Credenciales rotadas: Supabase, Alpaca, HuggingFace
   - Credenciales eliminadas: Benzinga, EODHD (no se usaban)
   - 3 repos viejos privados + archivados en GitHub
   - Carpeta `trading-system/` creada con `.env` válido

4. **Bloque 3 — Ficheros base del monorepo generados** (este zip):
   - Estructura completa de carpetas
   - pyproject.toml, .gitignore, .env.example, README.md, LICENSE
   - shared/config.py y shared/db.py **funcionales**
   - Stubs (con TODOs marcados) de shared/indicators, guardrails, inference
   - Apps con main.py como placeholder
   - Tests básicos de smoke con pytest
   - Configs yaml base

### Pendiente

5. **Bloque 4 — Migración del código** (próximo paso):
   - Mover código real de los 3 repos viejos a apps/ del monorepo
   - Adaptar imports al nuevo esquema (shared.X en lugar de sys.path hacks)
   - Eliminar duplicaciones (db clients, create_client repetidos)

6. **Fase 3.4 — Refactor de paridad** (después del Bloque 4):
   - Implementar shared/indicators.py completo (fórmulas canónicas de silver.py)
   - Migrar silver.py y silver_rt.py para que consuman shared/indicators
   - Tests unitarios comparativos training vs live

7. **Fase 3.5 — Reconciliation.py** (resuelve F-39):
   - Implementar polling de Alpaca para detectar fills y cierres
   - Actualizar gold_trades con ts_salida, precio_salida, pnl, motivo_salida

8. **Quick wins del informe** (aprovechar para tocarlos):
   - F-19: crear tabla `config` en Supabase para circuit_breaker
   - F-28, F-41: timezone ET en contadores diarios (ya hay helper en shared/utils/time.py)
   - D-14: índices en gold_trades
   - F-42: completar campos del dict al insertar trades

9. **Fase 4**: refactor arquitectónico (versionado modelos, etc.)

10. **Fase 5**: frontend Streamlit

---

## Primeros pasos para el usuario al abrir nueva conversación

1. **Crear venv dedicado y activar**:
   ```bash
   cd C:\Users\jgrma\Desktop\APIs\trading-system
   python -m venv .venv
   .venv\Scripts\activate       # Windows
   ```

2. **Instalar el monorepo como paquete editable**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Test smoke de que todo arranca**:
   ```bash
   python -m shared.config      # imprime Config OK
   python -m shared.db          # conecta a Supabase y lee 3 symbols
   pytest tests/ -v             # tests de smoke de indicadores
   ```

4. **Inicializar git (primera vez)**:
   ```bash
   git init
   git add .
   git commit -m "Initial monorepo structure"
   # Luego crear repo privado en GitHub y push
   ```

5. **Decir a Claude en la nueva conversación**:
   > "Retomo el refactor de mi sistema de trading algorítmico. 
   > Adjunto RETOMAR_AQUI.md y el informe_viabilidad.docx.
   > Estoy listo para Bloque 4 (migración del código)."

---

## Archivos clave para adjuntar en la siguiente conversación

1. `RETOMAR_AQUI.md` (este fichero)
2. `informe_viabilidad.docx` (los 135 issues y decisiones arquitectónicas)
3. Si hace falta, los 3 repos viejos están archivados en GitHub para consulta.

---

## Comentarios importantes para el próximo Claude

- **No es un TFM**, es proyecto personal. Las decisiones se juzgan por criterios técnicos puros, no por "defensa académica".
- El usuario prefiere **paso a paso con preguntas concretas** (opciones enumeradas), no monólogos largos.
- Las respuestas en **español**.
- **Validar cada paso antes de avanzar al siguiente** (uno a uno, no todo de golpe).
- El sistema objetivo es single-user local (el usuario corre todo en su propia máquina Windows).
- **F-39 es el issue arquitectónico más grave** junto con F-88..F-91 (paridad training/live).
