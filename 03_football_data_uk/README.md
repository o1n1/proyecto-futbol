# 03_football_data_uk - Cuotas de Casas de Apuestas (CSV)

Datos históricos de cuotas de Football-Data.co.uk con cuotas de múltiples casas de apuestas incluyendo Pinnacle y Bet365.

## Fuente de Datos
- **Web**: https://www.football-data.co.uk/data.php
- **Formato**: CSV
- **Almacenamiento**: Local

## Ligas Disponibles

| Código | Liga | País | League ID |
|--------|------|------|-----------|
| E0 | Premier League | England | 39 |
| E1 | Championship | England | 40 |
| D1 | Bundesliga | Germany | 78 |
| I1 | Serie A | Italy | 135 |
| SP1 | La Liga | Spain | 140 |
| F1 | Ligue 1 | France | 61 |
| N1 | Eredivisie | Netherlands | 88 |
| P1 | Primeira Liga | Portugal | 94 |
| B1 | Jupiler League | Belgium | 144 |
| T1 | Super Lig | Turkey | 203 |

## Temporadas Disponibles
- 2019-20 (código: 1920)
- 2020-21 (código: 2021)
- 2021-22 (código: 2122)
- 2022-23 (código: 2223)
- 2023-24 (código: 2324)
- 2024-25 (código: 2425)

## Estructura

```
03_football_data_uk/
├── README.md           # Este archivo
├── scripts/
│   └── 01_download_data.py  # Descargar CSVs
├── data/
│   ├── E0_1920.csv         # Premier League 2019-20
│   ├── E0_2021.csv         # Premier League 2020-21
│   ├── ...
│   └── all_leagues_combined.csv  # Todos los datos combinados
├── models/                 # Modelos entrenados (futuro)
└── outputs/                # Resultados de análisis (futuro)
```

## Uso

```bash
# Descargar todos los CSVs
python scripts/01_download_data.py
```

## Columnas de Cuotas

### Cuotas 1X2
| Columna | Descripción |
|---------|-------------|
| B365H | Bet365 - Home |
| B365D | Bet365 - Draw |
| B365A | Bet365 - Away |
| PSH | Pinnacle - Home |
| PSD | Pinnacle - Draw |
| PSA | Pinnacle - Away |
| MaxH | Máxima del mercado - Home |
| MaxD | Máxima del mercado - Draw |
| MaxA | Máxima del mercado - Away |
| AvgH | Promedio del mercado - Home |
| AvgD | Promedio del mercado - Draw |
| AvgA | Promedio del mercado - Away |

### Over/Under 2.5
| Columna | Descripción |
|---------|-------------|
| BbMx>2.5 | Máxima Over 2.5 |
| BbAv>2.5 | Promedio Over 2.5 |
| BbMx<2.5 | Máxima Under 2.5 |
| BbAv<2.5 | Promedio Under 2.5 |

### Resultado del Partido
| Columna | Descripción |
|---------|-------------|
| FTHG | Full Time Home Goals |
| FTAG | Full Time Away Goals |
| FTR | Full Time Result (H/D/A) |
| HTHG | Half Time Home Goals |
| HTAG | Half Time Away Goals |
| HTR | Half Time Result |

## Columnas Renombradas (después de procesamiento)

| Original | Renombrado |
|----------|------------|
| FTHG | goals_home |
| FTAG | goals_away |
| FTR | result |
| B365H | odds_b365_home |
| B365D | odds_b365_draw |
| B365A | odds_b365_away |
| PSH | odds_pinnacle_home |
| PSD | odds_pinnacle_draw |
| PSA | odds_pinnacle_away |
| MaxH | odds_max_home |
| MaxD | odds_max_draw |
| MaxA | odds_max_away |
| AvgH | odds_avg_home |
| AvgD | odds_avg_draw |
| AvgA | odds_avg_away |

## Ventajas de Esta Fuente

1. **Cuotas Pinnacle**: Consideradas las más "sharp" del mercado
2. **Cuotas máximas**: Útiles para calcular mejor valor esperado
3. **Histórico amplio**: Desde 2019
4. **Formato simple**: CSV fácil de procesar

## Notas

- Los archivos se descargan sin autenticación
- Puede requerir `verify=False` en entornos corporativos (SSL)
- Los datos se actualizan semanalmente durante la temporada
