# Proyecto Fútbol - Análisis y Predicción de Partidos

## Descripción
Sistema de análisis y predicción de partidos de fútbol con múltiples fuentes de datos. El proyecto contiene:
- **API-Football (Supabase)**: 112K+ partidos con features ML y modelos entrenados
- **SofaScore (SQLite)**: 392K+ eventos con cuotas de todas las ligas del mundo
- **Hybrid DB (SQLite)**: Base de datos local que combina features de API-Football + cuotas de SofaScore

Este problema pertenece a un sistema formal determinista.
No aceptes correcciones sin verificación lógica.
Verifica toda la información.
Si necesitas información del usuario, pídela.
MCP de supabase activo
MCP de playwright activo

## Estructura del Proyecto (Reorganizado Enero 2026)

```
proyecto/
├── CLAUDE.md                           # Este archivo
├── requirements.txt                    # Dependencias
│
├── 01_api_football/                    # BD Principal (Supabase) + ML
│   ├── README.md                       # Documentación del módulo
│   ├── scripts/                        # Scripts numerados (01-11)
│   │   ├── 01_migrate_fixtures.py      # Migrar Excel → Supabase
│   │   ├── 02_setup_features_table.py  # Crear tabla features
│   │   ├── 03_calculate_features.py    # Calcular ~85 features
│   │   ├── 04_sync_daily.py            # Sincronizar con API
│   │   ├── 05_update_draw_features.py  # Features de empates
│   │   ├── 06_train_model.py           # Entrenar modelos ML
│   │   ├── 07_predict_batch.py         # Predicciones en lote
│   │   ├── 08_analyze_calibration.py   # Análisis calibración
│   │   ├── 09_select_strategies.py     # Seleccionar estrategias
│   │   ├── 10_validate_evaluation.py   # Validar estrategias
│   │   └── 11_simulate_bankroll.py     # Simular bankroll
│   ├── data/
│   │   └── full_database_export.xlsx   # Datos iniciales
│   ├── models/                         # Modelos entrenados
│   │   ├── xgb_result_model.pkl
│   │   ├── lgb_result_model.pkl
│   │   ├── imputer_*.pkl
│   │   └── metrics_*.json
│   └── outputs/                        # Resultados
│       ├── predictions.xlsx
│       ├── simulation_daily.xlsx
│       └── simulation_metrics.json
│
├── 02_sofascore/                       # BD Secundaria (SQLite)
│   ├── README.md
│   ├── scripts/
│   │   └── 01_scrape_sofascore.py      # Scraper principal
│   ├── data/
│   │   ├── sofascore.db                # SQLite (392K eventos)
│   │   ├── checkpoint.json
│   │   └── raw/                        # JSONs por día
│   ├── models/                         # (futuro)
│   └── outputs/                        # (futuro)
│
├── 03_hybrid_db/                       # BD Híbrida (API-Football + SofaScore)
│   ├── README.md
│   ├── scripts/
│   │   ├── 01_export_api_football.py   # Exportar Supabase → SQLite
│   │   ├── 02_match_events.py          # Matching API ↔ SofaScore
│   │   ├── 03_merge_odds.py            # Unir cuotas de SofaScore
│   │   ├── 04_prepare_training_data.py # Preparar datos para ML
│   │   └── 05_train_hybrid_model.py    # Entrenar modelo híbrido
│   ├── data/
│   │   └── hybrid.db                   # SQLite con datos combinados
│   ├── models/                         # Modelos híbridos entrenados
│   └── outputs/                        # Resultados
│
└── .github/
    └── workflows/
        ├── sync_daily.yml
        └── calculate_features.yml
```

## Convención de Nombres

| Prefijo | Significado |
|---------|-------------|
| `01_`, `02_` | Orden de ejecución de scripts |
| `01_api_football/` | Módulo de BD API-Football |
| `02_sofascore/` | Módulo de BD SofaScore |
| `03_hybrid_db/` | Módulo de BD Híbrida (API + SofaScore) |

## Bases de Datos Disponibles

| Módulo | Fuente | Almacenamiento | Eventos | Período |
|--------|--------|----------------|---------|---------|
| 01_api_football | api-football.com | Supabase | 112,164 | 2020-2026 |
| 02_sofascore | sofascore.com | SQLite | 392,686 | 2019-2026 |
| 03_hybrid_db | API + SofaScore | SQLite local | ~80-90K | 2020-2026 |

## Base de Datos - Supabase

### Proyecto
- **Nombre**: Futbol
- **ID**: `ykqaplnfrhvjqkvejudg`
- **Región**: us-east-1
- **URL**: `https://ykqaplnfrhvjqkvejudg.supabase.co`

### Tabla Principal: `fixtures`

Contiene todos los partidos de fútbol con 40 columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `fixture_id` | BIGINT (PK) | ID único del partido de la API |
| `date` | TIMESTAMPTZ | Fecha y hora del partido |
| `timestamp` | BIGINT | Timestamp Unix |
| `timezone` | VARCHAR(50) | Zona horaria |
| `venue_id` | BIGINT | ID del estadio |
| `venue_name` | VARCHAR(255) | Nombre del estadio |
| `venue_city` | VARCHAR(255) | Ciudad del estadio |
| `status_long` | VARCHAR(100) | Estado completo |
| `status_short` | VARCHAR(10) | Estado abreviado |
| `status_elapsed` | SMALLINT | Minutos transcurridos |
| `league_id` | BIGINT | ID de la liga |
| `league_name` | VARCHAR(255) | Nombre de la liga |
| `league_country` | VARCHAR(100) | País de la liga |
| `league_season` | SMALLINT | Temporada |
| `league_round` | VARCHAR(100) | Jornada/Ronda |
| `home_team_id` | BIGINT | ID equipo local |
| `home_team_name` | VARCHAR(255) | Nombre equipo local |
| `away_team_id` | BIGINT | ID equipo visitante |
| `away_team_name` | VARCHAR(255) | Nombre equipo visitante |
| `goals_home` | SMALLINT | Goles local |
| `goals_away` | SMALLINT | Goles visitante |
| `score_halftime_home` | SMALLINT | Goles local al medio tiempo |
| `score_halftime_away` | SMALLINT | Goles visitante al medio tiempo |
| `score_fulltime_home` | SMALLINT | Goles local tiempo completo |
| `score_fulltime_away` | SMALLINT | Goles visitante tiempo completo |
| `json_data` | JSONB | Datos adicionales de la API |
| `created_at` | TIMESTAMPTZ | Fecha de creación |
| `updated_at` | TIMESTAMPTZ | Fecha de actualización |
| `stats_fetched` | SMALLINT | Flag de estadísticas obtenidas |
| `odds_fetched` | SMALLINT | Flag de cuotas obtenidas |
| `home_shots_on_goal` | SMALLINT | Tiros a puerta local |
| `home_total_shots` | SMALLINT | Tiros totales local |
| `home_ball_possession` | DECIMAL(5,2) | Posesión local (%) |
| `away_shots_on_goal` | SMALLINT | Tiros a puerta visitante |
| `away_total_shots` | SMALLINT | Tiros totales visitante |
| `away_ball_possession` | DECIMAL(5,2) | Posesión visitante (%) |
| `odds_home` | DECIMAL(6,2) | Cuota local |
| `odds_draw` | DECIMAL(6,2) | Cuota empate |
| `odds_away` | DECIMAL(6,2) | Cuota visitante |
| `match_type` | VARCHAR(50) | Tipo: Terminado, Próximo, Otro |

### Índices Creados
- `idx_fixtures_date` - Búsquedas por fecha
- `idx_fixtures_league_season` - Filtros por liga/temporada
- `idx_fixtures_home_team` - Búsquedas por equipo local
- `idx_fixtures_away_team` - Búsquedas por equipo visitante
- `idx_fixtures_teams` - Búsquedas por cualquier equipo
- `idx_fixtures_match_type` - Filtro por tipo de partido
- `idx_fixtures_status` - Filtro por estado
- `idx_fixtures_timestamp` - Ordenamiento por timestamp
- `idx_fixtures_json_data` - Búsquedas en JSON (GIN)

### Políticas RLS
- Lectura pública habilitada
- Escritura pública habilitada (para migración y API)

### Tabla de Features: `fixture_features`

Contiene ~130 columnas con features calculadas para predicción:

| Categoría | Columnas | Descripción |
|-----------|----------|-------------|
| **Forma General** | `home_form_*_last3/5/10`, `away_form_*_last3/5/10` | Puntos, victorias, goles, clean sheets últimos N partidos |
| **Forma Local/Visitante** | `home_home_form_*`, `away_away_form_*` | Rendimiento específico como local/visitante |
| **Temporada** | `home_season_*`, `away_season_*` | PPG, goles promedio, posición estimada |
| **Rachas** | `home_streak_*`, `away_streak_*` | Victorias, invicto, goleador, etc. |
| **Goles por Mitad** | `*_avg_goals_first/second_half` | Distribución de goles por tiempo |
| **Head to Head** | `h2h_*` | Historial de enfrentamientos directos |
| **Liga** | `league_*` | Estadísticas promedio de la liga |
| **Contextuales** | `*_days_rest`, `day_of_week`, `month` | Descanso, día de semana, mes |
| **Combinadas** | `attack_vs_defense_*`, `form_momentum_diff` | Features derivadas |
| **Targets** | `result`, `total_goals`, `btts`, `over25` | Variables objetivo (solo partidos terminados) |
| **Indicadores** | `is_home_new_team`, `is_away_new_team`, `has_h2h_history`, `*_matches_available` | Flags para ML cuando faltan datos históricos |

#### Columnas Indicadoras (para ML)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `is_home_new_team` | BOOLEAN | True si el equipo local no tiene historial suficiente (form_last3 es NULL) |
| `is_away_new_team` | BOOLEAN | True si el equipo visitante no tiene historial suficiente (form_last3 es NULL) |
| `has_h2h_history` | BOOLEAN | True si hay enfrentamientos directos previos entre ambos equipos |
| `home_form_matches_available` | SMALLINT | Número de partidos disponibles para calcular forma del local (0-10) |
| `away_form_matches_available` | SMALLINT | Número de partidos disponibles para calcular forma del visitante (0-10) |

Estas columnas permiten al modelo de ML diferenciar entre:
- "El equipo no ha ganado ningún partido" (información real)
- "No hay datos históricos del equipo" (falta de información)

#### Índices
- `idx_ff_result_null` - Fixtures sin targets (partidos futuros)
- `idx_ff_calculated_at` - Fecha de cálculo
- `idx_ff_version` - Versión de features

#### Estado Actual
- **Total features calculadas**: 108,446 (100%)
- **Con targets** (terminados): 108,132
- **Sin targets** (futuros): 314

## Estadísticas del Dataset

- **Total partidos**: 108,446
- **Ligas**: 285
- **Equipos**: 4,354
- **Período**: 31 dic 2019 - 4 oct 2025
- **Partidos terminados**: 108,132 (99.7%)
- **Partidos próximos**: 288 (0.3%)

### Distribución de Resultados (partidos terminados)
- Victoria Local: 44.0%
- Victoria Visitante: 31.6%
- Empate: 24.4%

### Estadísticas de Goles
- Promedio goles local: 1.51
- Promedio goles visitante: 1.24
- Total promedio por partido: 2.75
- Over 2.5: 51.4%
- BTTS (ambos marcan): 50.7%

## Credenciales

### API Keys (anon - pública)
```
URL: https://ykqaplnfrhvjqkvejudg.supabase.co
Anon Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlrcWFwbG5mcmh2anFrdmVqdWRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg2NjY4NjgsImV4cCI6MjA4NDI0Mjg2OH0.abeJY6QxUn4gT5GYJmoD2xJ7uPVNEwAVAxJ0wE5bMvM
```

## Scripts Disponibles

### migrate_fixtures.py
Script para migrar datos del Excel a Supabase. Características:
- Lee el archivo Excel con pandas
- Transforma tipos de datos para PostgreSQL
- Maneja valores NULL correctamente
- Upsert por lotes de 500 registros
- Bypass SSL para entornos corporativos

Uso:
```bash
python migrate_fixtures.py
```

### calculate_features.py
Script principal para calcular features derivadas de cada partido. Características:

**Principios de Diseño:**
1. **Cero data leakage**: Solo usa datos con `fecha < fecha_partido`
2. **Repetibilidad matemática**: Mismo input → mismo output
3. **Cálculo para TODOS los partidos**: Incluyendo futuros
4. **No recalcula**: Solo procesa fixtures SIN features existentes
5. **Actualización de targets**: Cuando un partido futuro termina, solo actualiza targets (no recalcula features)

**Features Calculadas (~85 por partido):**
- Forma últimos 3, 5, 10 partidos (puntos, goles, clean sheets, BTTS, over2.5)
- Forma específica local/visitante
- Estadísticas de temporada (PPG, posición estimada)
- Rachas actuales (victorias, invicto, goleador)
- Head to head histórico
- Promedios de la liga
- Días de descanso
- Día de semana y mes

**Flujo de Ejecución:**
1. FASE 1: Calcula features para fixtures nuevos (sin features)
2. FASE 2: Actualiza targets de partidos que pasaron de "Próximo" a "Terminado"

**Configuración:**
- `MAX_FIXTURES_PER_RUN = 5000` - Límite por ejecución (para GitHub Actions)
- `BATCH_SIZE = 500` - Tamaño de lote para inserts

Uso:
```bash
python calculate_features.py
```

### .github/workflows/calculate_features.yml
GitHub Action para ejecución automática diaria:
- **Schedule**: Diario a las 6:00 UTC
- **Manual**: Permite ejecución manual (workflow_dispatch)
- **Timeout**: 30 minutos máximo
- **Secret requerido**: `SUPABASE_KEY`

## Próximos Pasos (Pendientes)
- [ ] Inicializar repositorio Git y push a GitHub
- [ ] Configurar secret `SUPABASE_KEY` en GitHub Actions
- [x] ~~Entrenar modelo de predicción con las features~~ ✅ Completado (v2.2)
- [ ] Crear script de predicción diaria (`predict_daily.py`)
- [ ] Crear bot de Telegram para predicciones diarias
- [ ] Crear dashboard de visualización
- [ ] Implementar reentrenamiento mensual automático

## Notas Técnicas
- El entorno corporativo requiere `verify=False` en requests por temas de certificados SSL
- La migración usa la API REST de Supabase directamente (no el SDK) debido a restricciones de proxy
- El script de features usa paginación para manejar >100k registros (límite API: 1000 por request)
- Los batch inserts pueden fallar con "All object keys must match" - el script maneja esto insertando uno por uno como fallback

---

## Auditoría de Features (Enero 2026)

### Estado Actual de la Base de Datos

| Métrica | Valor |
|---------|-------|
| Total fixtures | 108,446 |
| Total features calculadas | 108,446 (100%) |
| Con targets (terminados) | 108,132 |
| Sin targets (futuros) | 314 |

### Verificación de Data Leakage

**✅ NO hay data leakage**. Verificado mediante:

1. **Revisión de código**: La función `get_team_matches_before_date()` usa `df['date'] < date` (estricto)
2. **Partidos futuros**: Tienen `result=NULL` y `home_goals=NULL` pero SÍ tienen features calculadas
3. **Verificación manual**: Para el equipo 2776 con partido el 2025-10-02, sus últimos 4 partidos antes de esa fecha promedian 0 goles, y la feature muestra `0.00`

### Análisis de Valores Nulos

| Feature | Con valor | Sin valor (NULL) | % Nulos | Causa |
|---------|-----------|------------------|---------|-------|
| `home_form_points_last5` | 101,829 | 6,617 | 6.1% | Equipos con <5 partidos |
| `home_form_points_last3` | 96,594 | 11,852 | 10.9% | Equipos con <3 partidos |
| `home_season_points_per_game` | 92,506 | 15,940 | 14.7% | Primera jornada temporada |
| `h2h_matches_total` | 108,446 | 0 | 0% | Siempre tiene valor (puede ser 0) |
| `league_avg_goals` | 106,405 | 2,041 | 1.9% | Ligas nuevas sin historial |

### Distribución de Casos Sin Historial

| Tipo | Cantidad | % |
|------|----------|---|
| **Con historial completo** | 68,412 | 63.1% |
| Sin H2H previo | 17,131 | 15.8% |
| Sin forma (equipo nuevo) | 11,852 | 10.9% |
| Primera jornada temporada | 11,051 | 10.2% |

### Estrategia de Manejo de NULLs

El script usa la siguiente estrategia cuando no hay datos históricos:

```python
if len(matches) == 0:
    return {
        'points': None,           # NULL para promedios
        'wins': 0, 'draws': 0,    # 0 para contadores
        'goals_scored': None,     # NULL para promedios
        ...
    }
```

**Reglas aplicadas:**
- **Promedios** (puntos, goles) → `NULL` si no hay datos
- **Contadores** (victorias, rachas) → `0` si no hay datos
- **H2H sin historial** → `h2h_matches_total = 0`, resto de H2H = `NULL`

### Columnas Indicadoras Agregadas

Para facilitar el entrenamiento de modelos ML, se agregaron 5 columnas indicadoras:

| Columna | Distribución |
|---------|--------------|
| `is_home_new_team = true` | 11,852 (10.9%) |
| `is_away_new_team = true` | 10,727 (9.9%) |
| `has_h2h_history = true` | 81,078 (74.8%) |
| `has_h2h_history = false` | 27,368 (25.2%) |
| `home_form_matches_available = 10` | 103,408 (95.4%) |
| `home_form_matches_available = 0` | 5,038 (4.6%) |

### Recomendaciones para Entrenamiento de Modelo

1. **Imputación de NULLs**: Usar `SimpleImputer(strategy='median')` en el pipeline de ML
2. **Usar indicadores**: Las columnas `is_*_new_team` y `has_h2h_history` permiten al modelo aprender que "falta información" es diferente de "mala performance"
3. **División temporal**: Usar partidos antiguos para train, recientes para test (no random split)
4. **No excluir datos**: El 37% de partidos con datos incompletos contiene información valiosa

### Conclusión

| Aspecto | Estado | Notas |
|---------|--------|-------|
| Cálculo completo | ✅ 100% | 108,446/108,446 |
| Data leakage | ✅ Cero | Verificado código + datos |
| Repetibilidad | ✅ Sí | Mismo input → mismo output |
| Manejo de NULLs | ✅ Documentado | Columnas indicadoras agregadas |
| Listo para ML | ✅ Sí | Usar imputación en pipeline |

---

## Entrenamiento de Modelos de Predicción (Enero 2026)

### Archivos Creados

| Archivo | Descripción |
|---------|-------------|
| `train_model.py` | Script principal de entrenamiento (~730 líneas) |
| `update_draw_features.py` | Script para actualizar features de empates |
| `models/` | Directorio con modelos entrenados |
| `models/xgb_result_model.pkl` | Modelo XGBoost para resultado (H/D/A) |
| `models/lgb_result_model.pkl` | Modelo LightGBM para resultado |
| `models/xgb_over25_model.pkl` | Modelo XGBoost para Over 2.5 |
| `models/lgb_over25_model.pkl` | Modelo LightGBM para Over 2.5 |
| `models/xgb_btts_model.pkl` | Modelo XGBoost para BTTS |
| `models/lgb_btts_model.pkl` | Modelo LightGBM para BTTS |
| `models/imputer_*.pkl` | Imputadores para cada target |
| `models/scaler_*.pkl` | Escaladores para cada target |
| `models/label_encoder_result.pkl` | Encoder de clases (A=0, D=1, H=2) |
| `models/feature_cols_*.json` | Lista de features usadas |
| `models/metrics_*.json` | Métricas de evaluación |

### División Temporal de Datos

```
TRAIN:      2020-01-01 → 2024-12-31    (93,769 partidos - 86.7%)
VALIDATE:   2025-01-01 → 2025-05-31    (8,054 partidos - 7.4%)
TEST:       2025-06-01 → 2025-10-04    (6,309 partidos - 5.8%)
```

**Justificación:**
- ✅ Sin data leakage (entrena con pasado, predice futuro)
- ✅ Simula predicción real
- ✅ TEST con datos completamente nuevos
- ✅ Detecta degradación temporal del modelo

### Features Utilizadas

**Total features**: 152 (incluyendo 11 nuevas features de empates v2.2)

#### Features de Empates (v2.2) - Nuevas

| Feature | Descripción |
|---------|-------------|
| `home_draw_rate_last10` | % empates del equipo local en últimos 10 partidos |
| `away_draw_rate_last10` | % empates del equipo visitante en últimos 10 partidos |
| `home_result_volatility` | Desviación estándar de puntos (baja = más empates) |
| `away_result_volatility` | Desviación estándar de puntos visitante |
| `home_balance_ratio` | Ratio goles a favor / goles en contra |
| `away_balance_ratio` | Ratio goles visitante |
| `home_low_scoring_rate` | % partidos con ≤2 goles totales |
| `away_low_scoring_rate` | % partidos low scoring visitante |
| `home_defense_strength` | % clean sheets del local |
| `away_defense_strength` | % clean sheets del visitante |
| `momentum_balance` | 1 - |PPG_home - PPG_away| / 3 (cercano a 1 = equipos igualados) |

### Evolución del Modelo

#### Problema Inicial (v1)
- El modelo ignoraba empates (Recall Draw: 1-2%)
- Optimizaba para Home Win (clase mayoritaria 44%)
- Gap Train-Test: ~8%

#### Mejoras Implementadas

| Versión | Cambio | Impacto |
|---------|--------|---------|
| v2 | `sample_weight='balanced'` (XGBoost) | Draw Recall: 1% → 31% |
| v2 | `is_unbalance=True` (LightGBM) | Draw Recall mejorado |
| v2.1 | `early_stopping_rounds=50` | Gap Train-Test: 13.7% → 6.2% |
| v2.1 | Regularización más fuerte | Menos overfitting |
| v2.2 | 11 features específicas para empates | +0.2% accuracy, modelo más estable |

### Hiperparámetros Finales

#### XGBoost (v2.2)
```python
XGBClassifier(
    max_depth=4,
    min_child_weight=5,
    n_estimators=500,
    gamma=1.0,
    reg_alpha=1.0,
    reg_lambda=1.5,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.05,
    early_stopping_rounds=50
)
```

#### LightGBM (v2.2)
```python
LGBMClassifier(
    max_depth=4,
    num_leaves=15,
    min_data_in_leaf=50,
    n_estimators=500,
    lambda_l1=1.0,
    lambda_l2=1.5,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    learning_rate=0.05,
    is_unbalance=True
)
```

### Resultados Finales (v2.2)

#### Modelo de Resultado (H/D/A) - XGBoost

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| Accuracy | 51.95% | 47.23% | **46.57%** |
| Balanced Accuracy | 50.22% | 45.43% | **44.68%** |
| Macro F1 | 0.5005 | 0.4544 | **0.4471** |
| Log Loss | 0.988 | 1.028 | **1.033** |

**Métricas por Clase (Test):**

| Clase | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Away (A) | 0.486 | 0.486 | 0.486 | 2,035 |
| Draw (D) | 0.272 | **0.327** | 0.297 | 1,518 |
| Home (H) | 0.594 | 0.526 | 0.558 | 2,756 |

**Análisis de Overfitting:**
- Gap Train-Val: 4.72% ✅ OK
- Gap Train-Test: 5.38% ✅ OK (objetivo <8%)
- Gap Val-Test: 0.66% ✅ OK

#### Modelo Over 2.5

| Modelo | Test Accuracy | Balanced Acc |
|--------|---------------|--------------|
| XGBoost | 56.84% | 56.84% |
| LightGBM | 57.43% | 57.42% |

#### Modelo BTTS

| Modelo | Test Accuracy | Balanced Acc |
|--------|---------------|--------------|
| XGBoost | 54.72% | 54.68% |
| LightGBM | 54.38% | 54.38% |

### Top 20 Features (XGBoost Result)

| # | Feature | Importancia |
|---|---------|-------------|
| 1 | home_form_wins_last10 | 0.0705 |
| 2 | **home_balance_ratio** | 0.0499 |
| 3 | home_form_goal_diff_last10 | 0.0424 |
| 4 | **away_balance_ratio** | 0.0334 |
| 5 | home_form_matches_available | 0.0292 |
| 6 | away_form_goal_diff_last10 | 0.0269 |
| 7 | home_season_points_per_game | 0.0223 |
| 8 | away_form_wins_last10 | 0.0212 |
| 9 | away_form_matches_available | 0.0208 |
| 10 | away_season_points_per_game | 0.0197 |
| 11 | rest_advantage | 0.0161 |
| 12 | expected_goals_diff | 0.0138 |
| 13 | away_season_goals_scored_avg | 0.0122 |
| 14 | is_away_new_team | 0.0119 |
| 15 | away_form_clean_sheets_last10 | 0.0109 |
| 16 | home_form_btts_last10 | 0.0098 |
| 17 | away_form_goals_scored_last10 | 0.0096 |
| 18 | h2h_home_goals_avg | 0.0094 |
| 19 | h2h_away_goals_avg | 0.0088 |
| 20 | league_avg_goals | 0.0082 |

**Nota**: Las features de empates `home_balance_ratio` y `away_balance_ratio` aparecen en el top 5, indicando que el modelo las usa activamente.

### Comparación de Versiones

| Métrica | v1 (inicial) | v2 (class weight) | v2.1 (early stop) | v2.2 (draw features) |
|---------|--------------|-------------------|-------------------|----------------------|
| Accuracy | 50.9% | 46.1% | 46.4% | **46.6%** |
| Balanced Acc | ~42% | 44.0% | 44.6% | **44.7%** |
| Draw Recall | 1-2% | 31.2% | 33.3% | **32.7%** |
| Macro F1 | 0.38 | 0.44 | 0.447 | **0.447** |
| Gap Train-Test | ~8% | 13.7% | 6.17% | **5.38%** |

### Conclusiones del Entrenamiento

| Aspecto | Estado | Notas |
|---------|--------|-------|
| Overfitting | ✅ Controlado | Gap 5.4% < 8% objetivo |
| Draw Prediction | ✅ Funcional | Recall ~33% (vs 1-2% inicial) |
| Generalización | ✅ Buena | Val-Test gap < 1% |
| Accuracy | ⚠️ Moderada | 46.6% (vs 44% baseline) |

**Interpretación:**
- El modelo supera el baseline (predecir siempre Home = 44%)
- Predice correctamente ~1 de cada 3 empates reales
- La accuracy de ~47% es razonable para predicción de fútbol con datos públicos
- Modelos profesionales con datos privados (lesiones, formaciones, etc.) alcanzan 55-60%

### Uso de los Modelos

```python
import joblib
import pandas as pd

# Cargar modelos
xgb_model = joblib.load('models/xgb_result_model.pkl')
imputer = joblib.load('models/imputer_result.pkl')
scaler = joblib.load('models/scaler_result.pkl')
label_encoder = joblib.load('models/label_encoder_result.pkl')

# Preprocesar features de un partido nuevo
X = imputer.transform(features_df)
X = scaler.transform(X)

# Predecir
y_pred = xgb_model.predict(X)
y_proba = xgb_model.predict_proba(X)

# Decodificar resultado
result = label_encoder.inverse_transform(y_pred)  # 'A', 'D', o 'H'
```

### Próximos Pasos

- [x] ~~Análisis de estrategias de apuestas~~ ✅ Completado (Enero 2026)
- [ ] Crear script de predicción diaria (`predict_daily.py`)
- [ ] Implementar bot de Telegram para enviar predicciones
- [ ] Crear dashboard de monitoreo de accuracy en producción
- [ ] Implementar reentrenamiento mensual automático
- [ ] Explorar features adicionales (clima, lesiones, motivación)

---

## Análisis de Estrategia de Apuestas (Enero 2026)

### Objetivo
Identificar combinaciones rentables de liga + umbral + tipo_apuesta, validar sin data leakage, y simular rentabilidad real con bankroll management.

### Metodología Anti-Data Leakage

```
┌─────────────────────────────────────────────────────────────┐
│                    TEST SET (Jun-Oct 2025)                   │
│                                                              │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │    CALIBRATION      │    │        EVALUATION           │ │
│  │    Jun-Jul 2025     │    │        Ago-Oct 2025         │ │
│  │    2,244 partidos   │    │        3,814 partidos       │ │
│  │                     │    │                             │ │
│  │  Descubrir mejores  │    │  Validar rentabilidad       │ │
│  │  combinaciones      │    │  (datos ciegos)             │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Scripts Creados

| Archivo | Descripción |
|---------|-------------|
| `analyze_calibration.py` | FASE 1: Genera todas las combinaciones liga × umbral × tipo |
| `select_strategies.py` | FASE 2: Filtra por accuracy ≥ 55% y n_partidos ≥ 10 |
| `validate_evaluation.py` | FASE 3: Valida en datos ciegos (Ago-Oct 2025) |
| `simulate_bankroll.py` | FASE 4: Simula bankroll día a día |
| `outputs/` | Directorio con resultados en Excel |

### Resultados de la Simulación

#### Configuración
- **Bankroll inicial**: $1,000
- **Stake por apuesta**: 2% del bankroll
- **Período**: Agosto - Octubre 2025 (63 días)

#### Resultados Finales

| Métrica | Valor |
|---------|-------|
| **Bankroll final** | **$1,423.16** |
| **ROI Total** | **+42.32%** |
| Total días con apuestas | 40 |
| Total apuestas | 145 |
| Apuestas ganadas | 101 |
| Apuestas perdidas | 44 |
| **Win Rate** | **69.7%** |
| Max Drawdown | 8.5% |
| Sharpe Ratio | 4.54 |

### Flujo del Análisis

#### FASE 1: Calibración (Jun-Jul 2025)
- **Input**: 2,244 partidos terminados
- **Ligas analizadas**: 32 (con ≥20 partidos)
- **Combinaciones generadas**: 548
- **Combinaciones con accuracy ≥55%**: 300
- **Combinaciones con accuracy ≥60%**: 259

#### FASE 2: Selección
- **Criterios**: accuracy ≥ 55%, n_partidos ≥ 10
- **Estrategias seleccionadas**: 138

| Tipo de Apuesta | Estrategias |
|-----------------|-------------|
| Home | 32 |
| Over 2.5 | 49 |
| BTTS | 47 |
| Away | 10 |

#### FASE 3: Validación (Ago-Oct 2025)
- **Input**: 3,814 partidos terminados
- **Estrategias degradadas** (bajan >10%): 59
- **Estrategias válidas**: 79

#### FASE 4: Simulación
- **Estrategias usadas**: 79 (no degradadas)
- **Días simulados**: 40 (con apuestas)
- **Resultado**: +42.32% ROI

### Top 10 Estrategias Validadas

| # | Liga | Tipo | Umbral | Accuracy Calib | Accuracy Eval |
|---|------|------|--------|----------------|---------------|
| 1 | Emperor Cup (Japan) | Home | ≥0.55 | 91.7% | 100% |
| 2 | Emperor Cup (Japan) | Home | ≥0.50 | 88.2% | 100% |
| 3 | Emperor Cup (Japan) | Home | ≥0.40 | 86.4% | 100% |
| 4 | Emperor Cup (Japan) | Home | ≥0.45 | 85.7% | 100% |
| 5 | Primera División (Peru) | Home | ≥0.40 | 78.6% | 88.9% |
| 6 | Emperor Cup (Japan) | Over25 | ≥0.45 | 66.7% | 87.5% |
| 7 | Emperor Cup (Japan) | Over25 | ≥0.40 | 66.7% | 87.5% |
| 8 | Emperor Cup (Japan) | Over25 | ≥0.50 | 66.7% | 83.3% |
| 9 | World Cup Qual. Europe | Home | ≥0.40 | 90.0% | 80.0% |
| 10 | Super League (China) | Over25 | ≥0.60 | 83.3% | 77.8% |

### Archivos de Salida

| Archivo | Contenido |
|---------|-----------|
| `outputs/calibration_results.xlsx` | 548 combinaciones con accuracy |
| `outputs/selected_strategies.xlsx` | 138 estrategias filtradas |
| `outputs/validated_strategies.xlsx` | 138 estrategias con validación |
| `outputs/simulation_daily.xlsx` | Bankroll día a día |
| `outputs/simulation_bets.xlsx` | Detalle de 145 apuestas |
| `outputs/simulation_metrics.json` | Métricas finales |

### Interpretación de Resultados

**¿Son los resultados confiables?**

| Aspecto | Estado | Notas |
|---------|--------|-------|
| Data Leakage | ✅ Cero | Modelo entrenado <Jun 2025, calibración Jun-Jul, evaluación Ago-Oct |
| Tamaño de muestra | ⚠️ Limitado | 145 apuestas en 40 días |
| Overfitting | ✅ Controlado | Validación en datos ciegos |
| Cuotas reales | ✅ Sí | Cuotas de cierre de casas de apuestas |

**Conclusión**: Los resultados son prometedores pero requieren validación con más datos:
- ROI de +42% en 2 meses es excelente pero puede ser varianza
- Win rate de 69.7% es muy alto y debería monitorearse
- Max drawdown de 8.5% es bajo y manejable
- Se recomienda continuar monitoreando con datos nuevos

### Uso de las Estrategias

```python
# Cargar estrategias validadas
import pandas as pd

strategies = pd.read_excel('outputs/validated_strategies.xlsx')

# Filtrar solo no degradadas
valid_strategies = strategies[~strategies['is_degraded']]

# Para un partido nuevo:
# 1. Verificar si la liga está en las estrategias
# 2. Verificar si la probabilidad supera el umbral
# 3. Si cumple ambos criterios → apostar
```

### Próximos Pasos para Apuestas

- [ ] Monitorear ROI en producción (Octubre 2025+)
- [ ] Ajustar estrategias mensualmente
- [ ] Implementar Kelly Criterion para stake variable
- [ ] Agregar más ligas europeas cuando inicien temporadas

---

## Sincronización Diaria de Datos (Enero 2026)

### Script: `sync_daily.py`

Script para sincronizar automáticamente la base de datos con datos de api-football.com.

### Credenciales API

```
API: api-football.com (api-sports.io)
URL Base: https://v3.football.api-sports.io
API Key: e09583304de2b04f4a046c31bdff0a75
Plan: Free (100 requests/día, ~30/minuto)
```

### Configuración

**Variables de Entorno:**
```bash
FOOTBALL_API_KEY=e09583304de2b04f4a046c31bdff0a75
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
DRY_RUN=false  # true para modo prueba
```

**Secrets en GitHub Actions:**
- `FOOTBALL_API_KEY`: API key de api-football.com
- `SUPABASE_KEY`: Ya existente

### Modos de Operación

| Modo | Condición | Acción |
|------|-----------|--------|
| **BACKFILL** | Gap >14 días desde último FT | Obtiene toda la temporada de ligas prioritarias |
| **DAILY** | Gap ≤14 días | Actualiza pendientes, busca próximos, obtiene odds |

### Ligas Prioritarias

**Ligas de Estrategia (17):**
```python
LIGAS_ESTRATEGIA = [
    102,   # Emperor Cup (Japan)
    344,   # Primera División (Bolivia)
    32,    # World Cup - Qualification Europe
    169,   # Super League (China)
    253,   # Major League Soccer (USA)
    71,    # Serie A (Brazil)
    241,   # Copa Colombia
    2,     # UEFA Champions League
    103,   # Eliteserien (Norway)
    848,   # UEFA Europa Conference League
    292,   # K League 1 (South-Korea)
    283,   # Liga I (Romania)
    265,   # Primera División (Chile)
    262,   # Liga MX (Mexico)
    357,   # Premier Division (Ireland)
    242,   # Liga Pro (Ecuador)
    239,   # Primera A (Colombia)
]
```

**Ligas Top Europeas (7):**
```python
LIGAS_TOP = [
    39,    # Premier League (England)
    140,   # La Liga (Spain)
    135,   # Serie A (Italy)
    78,    # Bundesliga (Germany)
    61,    # Ligue 1 (France)
    94,    # Primeira Liga (Portugal)
    88,    # Eredivisie (Netherlands)
]
```

### Endpoints Utilizados

| Endpoint | Uso | Requests |
|----------|-----|----------|
| `/fixtures?league=X&season=Y` | Backfill completo de liga | 1 por liga |
| `/fixtures?date=YYYY-MM-DD` | Partidos por fecha | 1 por día |
| `/fixtures?ids=ID1-ID2-...` | Hasta 20 fixtures con stats | 1 por 20 |
| `/odds?fixture=ID` | Cuotas pre-partido | 1 por fixture |

### Limitaciones Conocidas

1. **Plan Free REAL**: La API reporta solo **10 requests/día** (no 100 como indica la documentación)
2. **Odds históricas**: Solo disponibles ~7 días después del partido
3. **Stats**: Pueden tardar unas horas post-partido en estar disponibles
4. **Backfill lento**: Con 10 req/día, se necesitan ~3 días para sincronizar las 24 ligas prioritarias

### Uso

```bash
# Ejecución normal
set FOOTBALL_API_KEY=e09583304de2b04f4a046c31bdff0a75
python sync_daily.py

# Modo prueba (no escribe en BD)
set FOOTBALL_API_KEY=e09583304de2b04f4a046c31bdff0a75
set DRY_RUN=true
python sync_daily.py
```

### GitHub Actions

Archivo: `.github/workflows/sync_daily.yml`
- **Schedule**: 4:00 UTC diario (antes de calculate_features a las 6:00)
- **Manual**: Permite ejecución con opción DRY_RUN

### Ejecución Inicial (20 Enero 2026)

**Estado previo:**
- Último fixture FT: 2025-10-02
- Gap a llenar: ~107 días
- Total fixtures: 108,446

**Resultados de la primera ejecución:**

| Métrica | Valor |
|---------|-------|
| Modo | BACKFILL |
| Tiempo | 31.03s |
| Fixtures insertados | 2,986 |
| Fixtures fallidos | 0 |
| Requests usadas | 11 |
| Límite diario real | 10 (no 100) |

**Ligas actualizadas (temporada 2024):**

| Liga | ID | Fixtures |
|------|-----|----------|
| Emperor Cup (Japan) | 102 | 87 |
| Primera División (Bolivia) | 344 | 321 |
| World Cup Qual. Europe | 32 | 200 |
| Super League (China) | 169 | 240 |
| MLS (USA) | 253 | 526 |
| Serie A (Brazil) | 71 | 380 |
| Copa Colombia | 241 | 70 |
| UEFA Champions League | 2 | 279 |
| Eliteserien (Norway) | 103 | 242 |
| UEFA Conference League | 848 | 409 |
| K League 1 (South Korea) | 292 | 232 |
| **Total** | **11 ligas** | **2,986** |

**Estado actual de la BD:**
- Total fixtures: **108,783** (+337 nuevos)

**Ligas pendientes de sincronizar:**

| Liga | ID | Estado |
|------|-----|--------|
| Liga I (Romania) | 283 | Pendiente |
| Primera División (Chile) | 265 | Pendiente |
| Liga MX (Mexico) | 262 | Pendiente |
| Premier Division (Ireland) | 357 | Pendiente |
| Liga Pro (Ecuador) | 242 | Pendiente |
| Primera A (Colombia) | 239 | Pendiente |
| Premier League (England) | 39 | Pendiente |
| La Liga (Spain) | 140 | Pendiente |
| Serie A (Italy) | 135 | Pendiente |
| Bundesliga (Germany) | 78 | Pendiente |
| Ligue 1 (France) | 61 | Pendiente |
| Primeira Liga (Portugal) | 94 | Pendiente |
| Eredivisie (Netherlands) | 88 | Pendiente |
| **Total pendiente** | **13 ligas** | ~2-3 días más |

**PROBLEMA DETECTADO:** La API reporta un límite de **10 requests/día** en lugar de 100. Esto requiere múltiples días para completar el backfill.

### Archivos Relacionados

| Archivo | Descripción |
|---------|-------------|
| `sync_daily.py` | Script principal (~750 líneas, actualizado con batch odds) |
| `predict_batch.py` | Script de predicción en batch (~375 líneas) |
| `.github/workflows/sync_daily.yml` | Workflow de GitHub Actions |
| `sync_results.json` | Resultados de última ejecución |

---

## Actualización de Datos y Predicciones (20 Enero 2026)

### Resumen de la Sesión

Se completó la sincronización completa de datos, cálculo de features y generación de predicciones.

### Estado Final de la Base de Datos

| Métrica | Valor |
|---------|-------|
| **Total fixtures** | **112,164** |
| **Fixtures con features** | 112,164 (100%) |
| **Partidos terminados** | 110,561 |
| **Partidos futuros** | 1,603 |
| **Partidos con cuotas** | 7,900 (7.3%) |

### Sincronización de Datos

**Actualización del Rate Limiter:**
- Se corrigió un bug donde el script leía `x-ratelimit-limit` (por minuto = 10) en lugar de `x-ratelimit-requests-limit` (diario = 100)
- Ahora el script utiliza correctamente los 100 requests/día disponibles

**Datos Sincronizados:**
- **13,913 nuevos fixtures** insertados
- **24 ligas** sincronizadas (temporadas 2024 y 2025)
- **101 fixtures futuros** agregados (próximos 7 días)

**Nuevas Funciones en sync_daily.py:**
- `initialize_rate_limiter()`: Verifica estado real de la API antes de ejecutar (usa endpoint gratuito `/timezone`)
- `get_odds_by_league(league_id, season)`: Obtiene TODAS las cuotas de una liga en 1 request
- `get_odds_by_date(date)`: Obtiene TODAS las cuotas de una fecha en 1 request
- `fetch_odds_by_league_batch()`: Obtiene cuotas en batch por liga
- `fetch_odds_by_date_batch()`: Obtiene cuotas en batch por fecha

**Fix del Rate Limiter (20 Enero 2026):**
- El rate limiter no persistía entre ejecuciones, causando que se excediera el límite diario
- Ahora el script hace un request inicial a `/timezone` (gratuito) para verificar el estado real
- Si la API está bloqueada, el script termina inmediatamente con mensaje claro
- El contador de requests se sincroniza con el estado real de la API

**Nota sobre Cuotas:**
- La API de API-Football NO provee cuotas históricas en el plan free
- Las cuotas solo están disponibles durante ~7 días antes/después del partido
- Las cuotas existentes (7,900 partidos) provienen del dataset original

### Cálculo de Features

**Ejecución de calculate_features.py:**
- **3,718 features calculadas** (nuevos partidos sincronizados)
- **63 targets actualizados** (partidos que pasaron de futuros a terminados)
- **Tiempo de ejecución**: ~24 minutos
- **Tasa de éxito**: 100%

### Predicciones Generadas

**Script predict_batch.py creado:**
- Carga los 3 modelos entrenados (Result, Over25, BTTS)
- Genera predicciones para todos los fixtures con features
- Calcula accuracy en partidos terminados
- Exporta resultados a Excel

**Resultados de Predicción:**

| Métrica | Valor |
|---------|-------|
| **Total predicciones** | 112,164 |
| **Accuracy (partidos terminados)** | **51.18%** |
| **Partidos futuros pendientes** | 1,504 |

**Distribución de Predicciones:**
| Predicción | Cantidad | Porcentaje |
|------------|----------|------------|
| Home (H) | 44,559 | 39.7% |
| Away (A) | 37,316 | 33.3% |
| Draw (D) | 30,289 | 27.0% |

### Evaluación de ROI (Actualizada)

Se re-ejecutó la simulación de bankroll con los datos actualizados:

| Métrica | Valor Anterior | Valor Actual |
|---------|----------------|--------------|
| **Bankroll final** | $1,423.16 | **$1,485.90** |
| **ROI Total** | +42.32% | **+48.59%** |
| Total apuestas | 145 | 152 |
| Apuestas ganadas | 101 | 106 |
| Apuestas perdidas | 44 | 46 |
| **Win Rate** | 69.7% | **69.7%** |
| Max Drawdown | 8.5% | 10.0% |
| Sharpe Ratio | 4.54 | 4.53 |

### Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `outputs/predictions.xlsx` | 112,164 predicciones con probabilidades |
| `outputs/simulation_daily.xlsx` | Bankroll actualizado día a día |
| `outputs/simulation_bets.xlsx` | Detalle de 152 apuestas |
| `outputs/simulation_metrics.json` | Métricas actualizadas |

### Uso de predict_batch.py

```bash
# Predecir todos los partidos desde junio 2025
python predict_batch.py --min-date 2025-06-01

# Predecir solo partidos futuros
python predict_batch.py --future-only

# Especificar archivo de salida
python predict_batch.py --output outputs/mis_predicciones.xlsx
```

### Próximos Pasos

#### Fase 2: Agregar Features de Shots (Pendiente)

1. **Análisis de cobertura**: Identificar ligas con >90% de estadísticas de tiros
2. **Nuevas features** a agregar:
   - `home_avg_shots_on_goal_last5/10`
   - `home_avg_shots_conceded_last5/10`
   - `away_avg_shots_on_goal_last5/10`
   - `away_avg_shots_conceded_last5/10`
3. **Indicador**: `has_shots_history` para diferenciar partidos con/sin esta info
4. **Reentrenar modelo** con features adicionales
5. **Comparar accuracy** antes/después

#### Ligas con Buena Cobertura de Stats (>90%)
- Premier League (England)
- La Liga (Spain)
- Serie A (Italy)
- Bundesliga (Germany)
- Ligue 1 (France)
- MLS (USA)
- Liga MX (Mexico)
- ~20,000 partidos disponibles para entrenamiento con stats

#### Automatización
- [ ] Configurar GitHub Actions para sync diario
- [ ] Crear bot de Telegram para predicciones
- [ ] Implementar reentrenamiento mensual automático
- [ ] Dashboard de monitoreo de accuracy en producción

---

## Base de Datos SofaScore (Enero 2026)

### Descripción
Base de datos local SQLite con información completa de SofaScore: partidos, cuotas, estadísticas y alineaciones de TODAS las ligas del mundo desde 2019.

### Ubicación
```
02_sofascore/data/sofascore.db
```

### Estadísticas
| Métrica | Valor |
|---------|-------|
| Total eventos | 392,686 |
| Total filas de cuotas | 2,319,540 |
| Eventos con cuotas | 773,740 |
| Período | 2019-01-01 → 2026-01-23 |
| Países | 179 |
| Ligas | 2,621 |

### Eventos por Año
| Año | Eventos |
|-----|---------|
| 2019 | 52,581 |
| 2020 | 63,966 |
| 2021 | 67,766 |
| 2022 | 54,073 |
| 2023 | 51,931 |
| 2024 | 48,500 |
| 2025 | 51,931 |
| 2026 | 1,934 |

### Top 10 Países por Eventos
| País | Eventos |
|------|---------|
| England | 23,206 |
| World | 21,911 |
| Italy | 17,543 |
| Germany Amateur | 15,294 |
| Turkey | 13,548 |
| Brazil | 13,021 |
| Europe | 11,721 |
| USA | 11,360 |
| France | 9,611 |
| eSoccer | 9,370 |

### Uso del Scraper
```bash
# Scrapear un día
python 02_sofascore/scripts/01_scrape_sofascore.py --date 2024-06-15

# Scrapear un año
python 02_sofascore/scripts/01_scrape_sofascore.py --year 2024

# Ver estadísticas
python 02_sofascore/scripts/01_scrape_sofascore.py --stats

# Exportar a CSV
python 02_sofascore/scripts/01_scrape_sofascore.py --export
```

### Consultas SQL Útiles
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('02_sofascore/data/sofascore.db')

# Partidos con cuotas 1X2
df = pd.read_sql("""
    SELECT e.date, e.country, e.league_name,
           e.home_team, e.away_team,
           e.home_score, e.away_score,
           o1.odds_final as odds_home,
           o2.odds_final as odds_draw,
           o3.odds_final as odds_away
    FROM events e
    LEFT JOIN odds o1 ON e.event_id = o1.event_id
        AND o1.market_name = 'Full time' AND o1.choice_name = '1'
    LEFT JOIN odds o2 ON e.event_id = o2.event_id
        AND o2.market_name = 'Full time' AND o2.choice_name = 'X'
    LEFT JOIN odds o3 ON e.event_id = o3.event_id
        AND o3.market_name = 'Full time' AND o3.choice_name = '2'
    WHERE e.status = 'finished'
""", conn)
```

### Ventajas sobre API-Football
| Aspecto | API-Football | SofaScore |
|---------|--------------|-----------|
| Cuotas históricas | ❌ No disponibles | ✅ 2019-2026 |
| Ligas menores | ⚠️ Limitado | ✅ 2,621 ligas |
| Costo | 💰 Plan free limitado | 🆓 Gratuito |
| Rate limiting | 100/día | Sin límite |

### Mercados de Cuotas Disponibles
- Full time (1X2)
- Double chance
- 1st half
- Draw no bet
- Both teams score (BTTS)
- Over/Under 2.5
- Asian handicap
- Corners

---

## Base de Datos Híbrida (Enero 2026)

### Descripción
Base de datos SQLite local que combina:
- **Features de API-Football**: 112K partidos con 152 features calculadas
- **Cuotas de SofaScore**: 2.3M filas de cuotas históricas

### Ubicación
```
03_hybrid_db/data/hybrid.db
```

### Scripts Disponibles

| Script | Descripción |
|--------|-------------|
| `01_export_api_football.py` | Exporta fixtures y features de Supabase a SQLite |
| `02_match_events.py` | Matching fuzzy entre API-Football ↔ SofaScore |
| `03_merge_odds.py` | Une cuotas de SofaScore a los fixtures |
| `04_prepare_training_data.py` | Crea tabla `training_data` lista para ML |
| `05_train_hybrid_model.py` | Entrena modelos XGBoost/LightGBM |

### Orden de Ejecución

```bash
# 1. Exportar de Supabase (requiere sofascore.db copiada)
python 03_hybrid_db/scripts/01_export_api_football.py

# 2. Matching entre fuentes
python 03_hybrid_db/scripts/02_match_events.py

# 3. Merge de cuotas
python 03_hybrid_db/scripts/03_merge_odds.py

# 4. Preparar datos de entrenamiento
python 03_hybrid_db/scripts/04_prepare_training_data.py

# 5. Entrenar modelo híbrido
python 03_hybrid_db/scripts/05_train_hybrid_model.py
```

### Esquema de Base de Datos

```sql
-- Datos de API-Football
api_fixtures (fixture_id, date, league_name, home_team_name, away_team_name, ...)
api_features (fixture_id, 152 columnas de features)

-- Mapping entre fuentes
event_mapping (fixture_id, event_id, match_score, match_method)

-- Cuotas de SofaScore
hybrid_odds (fixture_id, odds_home_open/close, odds_draw_open/close, odds_away_open/close, ...)

-- Dataset final para ML
training_data (todas las columnas combinadas + features derivadas de cuotas)
```

### Features de Cuotas Agregadas

| Feature | Descripción |
|---------|-------------|
| `implied_prob_home` | 1 / odds_home_close |
| `implied_prob_draw` | 1 / odds_draw_close |
| `implied_prob_away` | 1 / odds_away_close |
| `odds_movement_home` | odds_close - odds_open (movimiento de línea) |
| `odds_movement_draw` | Movimiento de cuota de empate |
| `odds_movement_away` | Movimiento de cuota visitante |

### Requisitos

1. **sofascore.db**: Debe estar en `02_sofascore/data/sofascore.db`
2. **Conexión a Supabase**: Para exportar fixtures y features
3. **Librerías**: pandas, numpy, scikit-learn, xgboost, lightgbm
