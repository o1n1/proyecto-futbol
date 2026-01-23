# 02_sofascore - BD de Cuotas Históricas (SQLite Local)

Base de datos local con información completa de SofaScore: partidos, cuotas, estadísticas y alineaciones de TODAS las ligas del mundo.

## Fuente de Datos
- **API**: REST API de SofaScore (gratuita, sin autenticación)
- **URL Base**: https://www.sofascore.com/api/v1
- **Almacenamiento**: SQLite local + JSON crudos

## Estadísticas Generales

| Métrica | Valor |
|---------|-------|
| Total eventos | 392,686 |
| Eventos terminados | 359,567 |
| Eventos cancelados/pospuestos | 33,108 |
| Eventos futuros | 11 |
| Total filas de cuotas | 2,319,540 |
| **Eventos con cuotas** | **313,689 (87.2%)** |
| Período | 2019-01-01 -> 2026-01-23 |
| Países | 179 |
| Ligas | 1,372 |

### Cobertura de Cuotas por Año

| Año | Terminados | Con Cuotas | Cobertura |
|-----|------------|------------|-----------|
| 2019 | 51,500 | 45,358 | 88.1% |
| 2020 | 47,391 | 33,060 | 69.8% * |
| 2021 | 60,227 | 51,652 | 85.8% |
| 2022 | 51,494 | 45,713 | 88.8% |
| 2023 | 49,718 | 44,955 | 90.4% |
| 2024 | 46,793 | 44,061 | 94.2% |
| 2025 | 50,653 | 47,185 | 93.2% |
| 2026 | 1,787 | 1,701 | 95.2% |

*2020 tiene menor cobertura probablemente por la pandemia COVID-19.

### Eventos por Status

| Status | Cantidad |
|--------|----------|
| finished | 359,567 |
| canceled | 25,650 |
| postponed | 7,450 |
| notstarted | 11 |
| otros | 8 |

## Estructura

```
02_sofascore/
├── README.md           # Este archivo
├── scripts/
│   ├── 01_scrape_sofascore.py  # Scraper principal
│   └── 02_export_to_csv.py     # Exportar a CSV (futuro)
├── data/
│   ├── sofascore.db            # SQLite (~270MB)
│   ├── checkpoint.json         # Estado del scraping
│   ├── raw/                    # JSONs crudos por día
│   │   ├── 2019/
│   │   ├── 2020/
│   │   ├── 2021/
│   │   ├── 2022/
│   │   ├── 2023/
│   │   ├── 2024/
│   │   ├── 2025/
│   │   └── 2026/
│   └── details/                # Stats y lineups (futuro)
├── models/                     # Modelos entrenados (futuro)
└── outputs/                    # Resultados de análisis (futuro)
```

## Uso del Scraper

```bash
# Scrapear un día específico
python scripts/01_scrape_sofascore.py --date 2024-06-15

# Scrapear un año completo
python scripts/01_scrape_sofascore.py --year 2024

# Scrapear rango de años
python scripts/01_scrape_sofascore.py --range 2019-2026

# Ver estadísticas de la BD
python scripts/01_scrape_sofascore.py --stats

# Exportar a CSV
python scripts/01_scrape_sofascore.py --export

# Continuar desde checkpoint (resume automático)
python scripts/01_scrape_sofascore.py
```

## Estructura de la Base de Datos

### Tabla: events
```sql
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY,
    date TEXT,
    time TEXT,
    timestamp INTEGER,
    country TEXT,
    league_name TEXT,
    league_id INTEGER,
    season TEXT,
    round INTEGER,
    home_team TEXT,
    home_team_id INTEGER,
    away_team TEXT,
    away_team_id INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    home_score_ht INTEGER,
    away_score_ht INTEGER,
    status TEXT,
    winner_code INTEGER,
    slug TEXT,
    scraped_at TEXT
);
```

### Tabla: odds
```sql
CREATE TABLE odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    market_name TEXT,      -- "Full time", "BTTS", "Over/Under", etc.
    market_id INTEGER,
    choice_name TEXT,      -- "1", "X", "2", "Yes", "No", etc.
    odds_initial REAL,
    odds_final REAL,
    winning INTEGER,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
```

### Tabla: statistics (para ligas mayores)
```sql
CREATE TABLE statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    period TEXT,           -- "ALL", "1ST", "2ND"
    stat_name TEXT,
    stat_key TEXT,
    home_value REAL,
    away_value REAL,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
```

### Tabla: lineups (para ligas mayores)
```sql
CREATE TABLE lineups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    team_type TEXT,        -- "home", "away"
    team_id INTEGER,
    player_id INTEGER,
    player_name TEXT,
    position TEXT,
    jersey_number INTEGER,
    is_starter INTEGER,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
```

## Ejemplos de Consultas

### Partidos con cuotas 1X2
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/sofascore.db')

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
    AND e.date >= '2024-01-01'
""", conn)
```

### Partidos de Bolivia
```python
df_bolivia = pd.read_sql("""
    SELECT * FROM events
    WHERE country = 'Bolivia'
    ORDER BY date DESC
""", conn)
```

### Mercados de cuotas disponibles
```python
df_markets = pd.read_sql("""
    SELECT market_name, COUNT(DISTINCT event_id) as events
    FROM odds
    GROUP BY market_name
    ORDER BY events DESC
""", conn)
```

## Mercados de Cuotas Disponibles

| Mercado | Descripción |
|---------|-------------|
| Full time | 1X2 (resultado final) - **único mercado scrapeado** |

**Nota**: Solo se scrapeó el mercado Full time (1X2). Cada evento tiene 3 filas de cuotas: 1 (local), X (empate), 2 (visitante).

---

## Diferencias: Partidos Terminados vs Futuros

| Campo | Terminado | Futuro |
|-------|-----------|--------|
| `home_score` | Valor (ej: 2) | `NULL` |
| `away_score` | Valor (ej: 1) | `NULL` |
| `home_score_ht` | Valor (ej: 1) | `NULL` |
| `away_score_ht` | Valor (ej: 1) | `NULL` |
| `winner_code` | 1=Local, 2=Visitante, 3=Empate | `NULL` |
| `status` | "finished" | "notstarted" |
| **Cuotas** | `winning=1` en la ganadora | `winning=0` en todas |
| `odds_initial` | Cuota de apertura | Cuota de apertura |
| `odds_final` | Cuota de cierre | Cuota actual |

**Ejemplo Partido Terminado:**
```
Lincoln City 2-1 Burton Albion
Cuotas: 1=1.49 (winning=1), X=3.9 (winning=0), 2=6.5 (winning=0)
```

**Ejemplo Partido Futuro:**
```
Geylang International vs Tampines Rovers
Scores: NULL
Cuotas: 1=3.6, X=4.2, 2=1.67 (todas winning=0)
```

---

## Diferencias Entre Ligas

### Ligas con Alta Cobertura de Cuotas (~95%+)
- Premier League (England) - 8,211 partidos
- LaLiga (Spain) - 8,082 partidos
- Serie A (Italy) - 8,074 partidos
- Bundesliga (Germany) - 6,513 partidos
- Ligue 1 (France) - 7,305 partidos
- Liga Portugal - 6,570 partidos
- Eredivisie (Netherlands) - 6,706 partidos

### Ligas con Baja/Nula Cobertura de Cuotas

| Tipo | Ejemplos | Razón |
|------|----------|-------|
| **Amistosos** | Club Friendly Games (5,227 sin cuotas) | Casas de apuestas no cubren |
| **eSports** | Liga Pro FIFA20/21 | Mercado pequeño |
| **Ligas menores** | Rusia 1ª Liga, Irán, Belarus | Poca demanda |
| **Juveniles** | U17 World Cup | Sin mercado de apuestas |
| **Ligas oscuras** | 3ª división Turquía, Croacia | Cobertura limitada |

### Top 10 Ligas por Cantidad de Partidos

| Liga | País | Partidos | Con Cuotas |
|------|------|----------|------------|
| Club Friendly Games | World | 40,297 | 28.9% |
| Primera B Nacional | Argentina | 12,140 | 32.6% |
| League Two | England | 12,101 | 32.7% |
| Championship | England | 12,058 | 33.2% |
| National League | England | 11,961 | 32.4% |
| League One | England | 11,836 | 32.8% |
| MLS | USA | 9,973 | 33.2% |
| LaLiga 2 | Spain | 9,964 | 33.2% |
| J.League 2 | Japan | 9,309 | 33.3% |

---

## Notas Técnicas

- **Paralelismo**: 20 workers, ~16 requests/segundo
- **Sin rate limiting**: La API no tiene límite detectable
- **Tiempo de scraping**: ~7 minutos para 2019-2026
- **Checkpoint**: Permite resume automático si se interrumpe
- **Stats/Lineups**: Tablas vacías (no se scrapearon detalles de eventos)
- **JSONs raw**: 5,158 archivos de respaldo (uno por día scrapeado)

---

## Fase 2: Obtener Estadísticas y Lineups (PENDIENTE)

### Estado Actual
- **Eventos terminados**: 359,567
- **Eventos con stats**: 0
- **Eventos con lineups**: 0
- **Requests necesarios**: ~719,134 (2 por evento)

### Problema: Bloqueo de Cloudflare (22 Enero 2026)

La API de SofaScore activó protección Cloudflare con captcha después de ~100 requests paralelos. Todos los endpoints devuelven:
```json
{"error": {"code": 403, "reason": "challenge"}}
```

**IP bloqueada**: La IP de Pirelli está bloqueada. Se necesita ejecutar desde otra red/computadora.

### Instrucciones para Ejecutar desde Otra PC

#### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/proyecto-futbol.git
cd proyecto-futbol
```

#### Paso 2: Instalar dependencias
```bash
pip install requests pandas tqdm
```

#### Paso 3: Probar si la API funciona
```python
# test_sofascore.py
import requests
import urllib3
urllib3.disable_warnings()

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.sofascore.com/',
})

url = 'https://www.sofascore.com/api/v1/sport/football/scheduled-events/2026-01-23'
r = session.get(url, timeout=30, verify=False)
print(f'Status: {r.status_code}')
if r.status_code == 200:
    print(f'SUCCESS! {len(r.json().get("events", []))} eventos')
else:
    print(f'BLOCKED: {r.text[:200]}')
```

**Si devuelve 200**: ¡La API funciona! Continúa al paso 4.
**Si devuelve 403**: La IP también está bloqueada, intenta desde otra red.

#### Paso 4: Descargar la base de datos

La base de datos SQLite (~270MB) NO está en el repositorio. Necesitas:

**Opción A**: Copiar `02_sofascore/data/sofascore.db` desde la PC original (USB, OneDrive, etc.)

**Opción B**: Re-scrapear todo (toma ~7 minutos):
```bash
cd 02_sofascore
python scripts/01_scrape_sofascore.py --range 2019-2026
```

#### Paso 5: Ejecutar scraping de detalles
```bash
cd 02_sofascore
python scripts/01_scrape_sofascore.py --details
```

**Configuración conservadora del scraper:**
- 3 workers paralelos (no 20)
- 0.5 segundos de delay entre requests
- Auto-pausa si detecta muchos errores 403
- Tiempo estimado: ~2.5-3 horas para 359K eventos

#### Paso 6: Verificar resultados
```bash
python scripts/01_scrape_sofascore.py --stats
```

Debería mostrar:
- Eventos con stats: >0
- Eventos con lineups: >0

### Notas Importantes

1. **No usar paralelismo agresivo**: El scraper ya está configurado para ser conservador
2. **La BD es grande**: ~270MB, no incluida en git
3. **Checkpoint automático**: Si se interrumpe, se puede continuar
4. **Ligas menores**: Muchas no tienen stats/lineups disponibles
