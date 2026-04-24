# trading-system

Sistema de trading algorítmico con ML sobre equities US intraday (1-min).

## Estado: refactor desde 3 repos previos a monorepo (abril 2026)

## Estructura

```
trading-system/
├── shared/              # librería compartida (indicadores, db, inference, guardrails)
├── apps/
│   ├── ingestion_historical/   # pipeline batch semanal (yfinance + Alpaca News)
│   ├── ingestion_live/         # scheduler 1-min para RAW y SILVER en tiempo real
│   ├── trading_engine/         # predict → decide → execute → reconcile
│   ├── ml_sandbox/             # training, evaluate, backtest
│   └── dashboard/              # Streamlit (pendiente Fase 5)
├── config/              # yamls de live, experiments, backtests
├── data/                # tensores, modelos, logs (no en git)
├── scripts/             # utilidades one-shot (migración, etc.)
└── tests/               # pytest
```

## Setup

```bash
# 1. Crear venv dedicado
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate      # Windows

# 2. Instalar proyecto como paquete editable
pip install -e .

# 3. (Opcional) extras de desarrollo
pip install -e ".[dev,dashboard]"

# 4. Copiar .env.example a .env y rellenar credenciales
cp .env.example .env
# Editar .env con tus claves reales
```

## Uso básico

```bash
# Ingesta histórica (batch semanal)
python -m apps.ingestion_historical.main

# Live engine (scheduler continuo)
python -m apps.trading_engine.main

# Training de un modelo
python -m apps.ml_sandbox.pipeline --config config/experiments/aapl_xgboost_1m.yaml

# Backtest de un ensemble
python -m apps.ml_sandbox.backtest --config config/backtests/aapl_backtest_v1.yaml

# Dashboard (Fase 5)
streamlit run apps/dashboard/app.py
```

## Documentación de arquitectura

Ver `docs/` (pendiente). Por ahora:
- Arquitectura Medallion (RAW → SILVER → GOLD) en Supabase
- Ensemble de 6 modelos (3 sklearn + 3 PyTorch)
- Guardarraíles configurables en yaml
- Paper trading con Alpaca (bracket orders)

## Decisiones arquitectónicas tomadas

| # | Decisión | Elección |
|---|---|---|
| 7.1 | Estructura | Monorepo con `shared/` |
| 7.2 | F-39 | Antes del frontend |
| 7.3 | Broker | Alpaca (Premium cuando toque) |
| 7.4 | Paridad training/live | Refactor completo con `shared/indicators.py` |
| 7.5 | Frontend | Streamlit |
| 7.6 | Ensemble | Mantener los 6 modelos |
| 7.7 | Uso del informe | Personal, interno |

## Repos predecesores (archivados)

- `Matacaa7/signals-historical` (archivado)
- `Matacaa7/trading-engine` (archivado)
- `Matacaa7/ml-sandbox` (archivado)
