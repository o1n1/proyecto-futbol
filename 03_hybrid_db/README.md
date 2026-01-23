# 03_hybrid_db - Base de Datos Híbrida

## Descripción

Este módulo combina datos de dos fuentes:
- **API-Football (Supabase)**: 112K partidos con 152 features calculadas
- **SofaScore (SQLite)**: 2.3M filas de cuotas históricas

El objetivo es crear un dataset enriquecido con features de rendimiento + cuotas para entrenar mejores modelos de predicción.

## Estructura

```
03_hybrid_db/
├── README.md                           # Este archivo
├── scripts/
│   ├── 01_export_api_football.py       # Exportar Supabase → SQLite
│   ├── 02_match_events.py              # Matching API ↔ SofaScore
│   ├── 03_merge_odds.py                # Unir cuotas de SofaScore
│   ├── 04_prepare_training_data.py     # Preparar datos para ML
│   └── 05_train_hybrid_model.py        # Entrenar modelo híbrido
├── data/
│   └── hybrid.db                       # SQLite con datos combinados
├── models/                             # Modelos entrenados
│   ├── xgb_hybrid_model.pkl
│   ├── lgb_hybrid_model.pkl
│   ├── imputer_hybrid.pkl
│   ├── scaler_hybrid.pkl
│   └── feature_cols_hybrid.json
└── outputs/                            # Resultados
```

## Requisitos Previos

1. **sofascore.db** debe existir en `02_sofascore/data/sofascore.db`
2. Acceso a Supabase (las credenciales están en el código)
3. Dependencias:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm requests
   ```

## Orden de Ejecución

### Paso 1: Exportar de Supabase
```bash
python scripts/01_export_api_football.py
```
- Exporta ~112K fixtures de Supabase
- Exporta ~112K registros de features
- Crea `hybrid.db` con tablas `api_fixtures` y `api_features`
- Tiempo estimado: ~5 minutos

### Paso 2: Matching entre fuentes
```bash
python scripts/02_match_events.py
```
- Compara cada fixture de API-Football con eventos de SofaScore
- Usa fuzzy matching por fecha + equipos + liga
- Crea tabla `event_mapping` con los matches
- Tiempo estimado: ~10-15 minutos

### Paso 3: Merge de cuotas
```bash
python scripts/03_merge_odds.py
```
- Para cada fixture con match, obtiene cuotas de SofaScore
- Extrae cuotas 1X2, Over/Under 2.5, BTTS
- Crea tabla `hybrid_odds`
- Tiempo estimado: ~5 minutos

### Paso 4: Preparar datos de entrenamiento
```bash
python scripts/04_prepare_training_data.py
```
- Combina features + cuotas en una sola tabla
- Agrega features derivadas de cuotas (probabilidades implícitas, movimientos)
- Crea tabla `training_data` lista para ML
- Tiempo estimado: ~2 minutos

### Paso 5: Entrenar modelo
```bash
python scripts/05_train_hybrid_model.py
```
- Divide datos temporalmente (train/val/test)
- Entrena XGBoost y LightGBM
- Guarda modelos y métricas
- Tiempo estimado: ~10 minutos

## Esquema de Base de Datos

### api_fixtures
Datos básicos de partidos exportados de Supabase.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| fixture_id | INTEGER PK | ID único |
| date | TEXT | Fecha del partido |
| league_name | TEXT | Nombre de la liga |
| home_team_name | TEXT | Equipo local |
| away_team_name | TEXT | Equipo visitante |
| goals_home | INTEGER | Goles local |
| goals_away | INTEGER | Goles visitante |

### api_features
152 features calculadas para predicción.

| Categoría | Ejemplos |
|-----------|----------|
| Forma | `home_form_points_last5`, `away_form_wins_last10` |
| Temporada | `home_season_ppg`, `away_season_goals_avg` |
| H2H | `h2h_matches_total`, `h2h_home_wins` |
| Rachas | `home_streak_wins`, `away_streak_unbeaten` |
| Target | `result`, `total_goals`, `btts`, `over25` |

### event_mapping
Vinculación entre API-Football y SofaScore.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| fixture_id | INTEGER PK | ID de API-Football |
| event_id | INTEGER | ID de SofaScore |
| match_score | REAL | Confianza del match (0-1) |
| match_method | TEXT | "exact" o "fuzzy" |

### hybrid_odds
Cuotas extraídas de SofaScore.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| fixture_id | INTEGER PK | ID del fixture |
| odds_home_open | REAL | Cuota apertura local |
| odds_home_close | REAL | Cuota cierre local |
| odds_draw_open | REAL | Cuota apertura empate |
| odds_draw_close | REAL | Cuota cierre empate |
| odds_away_open | REAL | Cuota apertura visitante |
| odds_away_close | REAL | Cuota cierre visitante |
| odds_over25_close | REAL | Cuota Over 2.5 |
| odds_btts_yes_close | REAL | Cuota BTTS Yes |

### training_data
Dataset final combinado para ML.

Incluye todas las columnas de `api_features` + `hybrid_odds` + features derivadas:
- `implied_prob_home`: Probabilidad implícita del local (1/odds)
- `implied_prob_draw`: Probabilidad implícita del empate
- `implied_prob_away`: Probabilidad implícita del visitante
- `odds_movement_home`: Movimiento de línea (cierre - apertura)

## Algoritmo de Matching

El script `02_match_events.py` usa el siguiente algoritmo:

1. Para cada fixture de API-Football:
2. Buscar eventos de SofaScore en ±1 día
3. Para cada candidato:
   - Normalizar nombres de equipos (remover FC, United, etc.)
   - Calcular similitud de nombres (SequenceMatcher)
   - Calcular similitud de liga
   - Score final = 80% equipos + 20% liga
4. Si score >= 0.85 → match válido
5. Guardar el mejor match

## Cobertura Esperada

| Métrica | Valor Esperado |
|---------|----------------|
| Fixtures totales | ~112K |
| Fixtures con match | ~80-90K (70-80%) |
| Con cuotas 1X2 | ~70-80K |
| Con cuotas Over/Under | ~50-60K |
| Con cuotas BTTS | ~40-50K |

La cobertura depende de:
- Overlap de ligas entre las dos fuentes
- Similitud de nombres de equipos
- Disponibilidad de cuotas en SofaScore

## Ventajas del Modelo Híbrido

1. **Más features**: Features de rendimiento + probabilidades implícitas
2. **Movimientos de línea**: Indica dónde va el "dinero inteligente"
3. **Calibración**: Comparar predicciones vs probabilidades del mercado
4. **Mejor generalización**: Más información para aprender patrones
