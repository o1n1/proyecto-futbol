# 01_api_football - BD Principal (Supabase)

Base de datos principal de fútbol en Supabase con 112K+ partidos y modelos ML entrenados.

## Fuente de Datos
- **API**: api-football.com (api-sports.io)
- **Almacenamiento**: Supabase (PostgreSQL cloud)
- **URL**: https://ykqaplnfrhvjqkvejudg.supabase.co

## Estadísticas
| Métrica | Valor |
|---------|-------|
| Total fixtures | 112,164 |
| Fixtures con features | 112,164 (100%) |
| Partidos terminados | 110,561 |
| Partidos futuros | 1,603 |
| Período | 2020-01-01 → 2026-01-22 |

## Estructura

```
01_api_football/
├── README.md           # Este archivo
├── scripts/            # Scripts ejecutables (en orden)
│   ├── 01_migrate_fixtures.py      # Migrar Excel → Supabase
│   ├── 02_setup_features_table.py  # Crear tabla de features
│   ├── 03_calculate_features.py    # Calcular ~85 features
│   ├── 04_sync_daily.py            # Sincronizar con API
│   ├── 05_update_draw_features.py  # Features de empates
│   ├── 06_train_model.py           # Entrenar modelos
│   ├── 07_predict_batch.py         # Predicciones en lote
│   ├── 08_analyze_calibration.py   # Análisis de calibración
│   ├── 09_select_strategies.py     # Seleccionar estrategias
│   ├── 10_validate_evaluation.py   # Validar estrategias
│   └── 11_simulate_bankroll.py     # Simular bankroll
├── data/
│   └── full_database_export.xlsx   # Datos iniciales (108K partidos)
├── models/             # Modelos entrenados
│   ├── xgb_result_model.pkl        # XGBoost resultado (H/D/A)
│   ├── lgb_result_model.pkl        # LightGBM resultado
│   ├── xgb_over25_model.pkl        # XGBoost Over 2.5
│   ├── lgb_over25_model.pkl        # LightGBM Over 2.5
│   ├── xgb_btts_model.pkl          # XGBoost BTTS
│   ├── lgb_btts_model.pkl          # LightGBM BTTS
│   ├── imputer_*.pkl               # Imputadores
│   ├── scaler_*.pkl                # Escaladores
│   ├── label_encoder_result.pkl    # Encoder de clases
│   ├── feature_cols_*.json         # Lista de features
│   └── metrics_*.json              # Métricas de evaluación
└── outputs/            # Resultados de análisis
    ├── sync_results.json           # Estado de sincronización
    ├── predictions.xlsx            # Predicciones
    ├── calibration_results.xlsx    # Resultados calibración
    ├── selected_strategies.xlsx    # Estrategias seleccionadas
    ├── validated_strategies.xlsx   # Estrategias validadas
    ├── simulation_daily.xlsx       # Bankroll diario
    └── simulation_metrics.json     # Métricas de simulación
```

## Orden de Ejecución

### Setup Inicial (una vez)
```bash
# 1. Migrar datos de Excel a Supabase
python scripts/01_migrate_fixtures.py

# 2. Crear tabla de features
python scripts/02_setup_features_table.py

# 3. Calcular features para todos los partidos
python scripts/03_calculate_features.py
```

### Actualización Diaria
```bash
# 4. Sincronizar nuevos partidos de la API
python scripts/04_sync_daily.py

# 5. Actualizar features de empates (si es necesario)
python scripts/05_update_draw_features.py

# 6. Calcular features de nuevos partidos
python scripts/03_calculate_features.py
```

### Entrenamiento de Modelos (mensual)
```bash
# 6. Entrenar modelos con datos actualizados
python scripts/06_train_model.py
```

### Predicciones
```bash
# 7. Generar predicciones para partidos futuros
python scripts/07_predict_batch.py --future-only

# 7b. Generar predicciones desde una fecha
python scripts/07_predict_batch.py --min-date 2025-06-01
```

### Análisis de Estrategias
```bash
# 8. Analizar combinaciones liga/umbral/tipo
python scripts/08_analyze_calibration.py

# 9. Seleccionar estrategias rentables
python scripts/09_select_strategies.py

# 10. Validar en datos ciegos
python scripts/10_validate_evaluation.py

# 11. Simular bankroll
python scripts/11_simulate_bankroll.py
```

## Métricas del Modelo (v2.2)

### Resultado (H/D/A)
| Métrica | Test |
|---------|------|
| Accuracy | 46.57% |
| Balanced Accuracy | 44.68% |
| Macro F1 | 0.447 |
| Draw Recall | 32.7% |

### Over 2.5
| Modelo | Test Accuracy |
|--------|---------------|
| XGBoost | 56.84% |
| LightGBM | 57.43% |

### BTTS
| Modelo | Test Accuracy |
|--------|---------------|
| XGBoost | 54.72% |
| LightGBM | 54.38% |

## Credenciales

```python
SUPABASE_URL = "https://ykqaplnfrhvjqkvejudg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # Ver CLAUDE.md
```

## API de Datos

```python
# API key de api-football.com
FOOTBALL_API_KEY = "e09583304de2b04f4a046c31bdff0a75"
# Plan: Free (100 requests/día)
```
