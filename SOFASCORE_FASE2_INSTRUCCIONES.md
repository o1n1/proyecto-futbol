# SofaScore Fase 2: Scraping de Estadísticas y Lineups

## Contexto del Proyecto

Este proyecto tiene 3 fuentes de datos de fútbol:
1. **API-Football (Supabase)**: 112K partidos con modelos ML entrenados
2. **SofaScore (SQLite)**: 392K eventos con cuotas históricas - **ESTA ES LA BD QUE ESTAMOS COMPLETANDO**
3. **Football-Data.co.uk (CSV)**: Cuotas de Pinnacle/Bet365

## Estado Actual de SofaScore (22 Enero 2026)

### Datos YA Scrapeados (Fase 1 - COMPLETADA)
| Métrica | Valor |
|---------|-------|
| Total eventos | 392,686 |
| Eventos terminados | 359,567 |
| Total filas de cuotas | 2,319,540 |
| Eventos con cuotas | 313,689 (87.2%) |
| Período | 2019-01-01 → 2026-01-23 |
| Países | 179 |
| Ligas | 1,372 |

### Datos PENDIENTES (Fase 2 - LO QUE HAY QUE HACER)
| Métrica | Valor |
|---------|-------|
| Eventos con statistics | **0** |
| Eventos con lineups | **0** |
| Requests necesarios | ~719,134 (2 por evento: stats + lineups) |

## ¿Qué Son Statistics y Lineups?

### Statistics (Estadísticas del Partido)
Datos del rendimiento de cada equipo durante el partido:

```json
// Ejemplo de respuesta de /event/{id}/statistics
{
  "statistics": [
    {
      "period": "ALL",  // También "1ST", "2ND"
      "groups": [
        {
          "groupName": "Match overview",
          "statisticsItems": [
            {"name": "Ball possession", "home": "55%", "away": "45%"},
            {"name": "Total shots", "home": "15", "away": "8"},
            {"name": "Shots on target", "home": "6", "away": "3"},
            {"name": "Corner kicks", "home": "7", "away": "4"},
            {"name": "Fouls", "home": "12", "away": "14"},
            {"name": "Yellow cards", "home": "2", "away": "3"},
            {"name": "Passes", "home": "450", "away": "380"},
            {"name": "Pass accuracy", "home": "85%", "away": "78%"}
          ]
        }
      ]
    }
  ]
}
```

**Se guarda en tabla `statistics`:**
```sql
CREATE TABLE statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    period TEXT,           -- "ALL", "1ST", "2ND"
    stat_name TEXT,        -- "Ball possession", "Total shots", etc.
    stat_key TEXT,         -- Clave normalizada
    home_value REAL,
    away_value REAL,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
```

### Lineups (Alineaciones)
Jugadores titulares y suplentes de cada equipo:

```json
// Ejemplo de respuesta de /event/{id}/lineups
{
  "home": {
    "players": [
      {"player": {"id": 123, "name": "Lionel Messi"}, "position": "F", "jerseyNumber": 10, "substitute": false},
      {"player": {"id": 456, "name": "Sergio Ramos"}, "position": "D", "jerseyNumber": 4, "substitute": false}
    ]
  },
  "away": {
    "players": [...]
  }
}
```

**Se guarda en tabla `lineups`:**
```sql
CREATE TABLE lineups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    team_type TEXT,        -- "home", "away"
    team_id INTEGER,
    player_id INTEGER,
    player_name TEXT,
    position TEXT,         -- "G", "D", "M", "F"
    jersey_number INTEGER,
    is_starter INTEGER,    -- 1=titular, 0=suplente
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
```

## ¿Por Qué Configuración "Conservadora"?

### Lo Que Pasó (22 Enero 2026, ~21:30)

1. **Fase 1 exitosa**: Scrapeamos 392K eventos + 2.3M cuotas en ~7 minutos usando 20 workers paralelos (~16 req/s)

2. **Inicio de Fase 2**: Intentamos scrapear stats/lineups con la misma configuración agresiva

3. **Bloqueo después de ~100 requests**: SofaScore activó protección Cloudflare
   ```json
   {"error": {"code": 403, "reason": "challenge"}}
   ```

4. **Intentos fallidos**:
   - VPN (diferentes servidores) → Bloqueado
   - Cloudscraper library → SSL issues con proxy corporativo
   - Cookies del navegador → No funcionó
   - Playwright automation → También bloqueado
   - Navegador real → Muestra captcha

5. **Conclusión**: La IP de Pirelli está bloqueada. Necesitamos otra red/PC.

### Configuración Conservadora Implementada

El script `02_sofascore/scripts/01_scrape_sofascore.py` fue modificado:

```python
# ANTES (agresivo - causó el bloqueo)
MAX_WORKERS = 20           # 20 workers paralelos
MAX_WORKERS_DETAILS = 20   # 20 para stats/lineups
REQUEST_DELAY = 0          # Sin delay

# DESPUÉS (conservador - para evitar bloqueos)
MAX_WORKERS = 5            # 5 workers para eventos
MAX_WORKERS_DETAILS = 3    # Solo 3 para stats/lineups
REQUEST_DELAY = 0.5        # 0.5 segundos entre requests
```

**Además se agregó detección de errores:**
```python
if errors_count >= 10:
    logger.warning("Muchos errores 403, posible bloqueo. Pausando 60s...")
    time.sleep(60)
    errors_count = 0
```

## Endpoints de la API

### Base URL
```
https://www.sofascore.com/api/v1
```

### Endpoints Usados

| Endpoint | Descripción | Ejemplo |
|----------|-------------|---------|
| `/sport/football/scheduled-events/{date}` | Eventos por fecha | `/scheduled-events/2024-06-15` |
| `/event/{id}` | Detalle de evento | `/event/12345678` |
| `/event/{id}/odds` | Cuotas del evento | `/event/12345678/odds` |
| `/event/{id}/statistics` | **Stats del partido** | `/event/12345678/statistics` |
| `/event/{id}/lineups` | **Alineaciones** | `/event/12345678/lineups` |

### Headers Requeridos
```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.sofascore.com/',
}
```

## Plan de Ejecución

### Paso 1: Verificar que la API funciona
```python
import requests
import urllib3
urllib3.disable_warnings()

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.sofascore.com/',
})

# Test con evento de hoy
url = 'https://www.sofascore.com/api/v1/sport/football/scheduled-events/2026-01-23'
r = session.get(url, timeout=30, verify=False)
print(f'Status: {r.status_code}')

if r.status_code == 200:
    events = r.json().get('events', [])
    print(f'SUCCESS! {len(events)} eventos encontrados')

    # Test de stats de un evento terminado
    if events:
        event_id = events[0]['id']
        stats_url = f'https://www.sofascore.com/api/v1/event/{event_id}/statistics'
        r2 = session.get(stats_url, timeout=30, verify=False)
        print(f'Stats status: {r2.status_code}')
else:
    print(f'BLOCKED: {r.text[:200]}')
```

**Si devuelve 200**: ¡Continúa!
**Si devuelve 403**: La IP está bloqueada, intenta desde otra red.

### Paso 2: Asegurar que tienes la BD
```bash
# Verificar que existe la BD
ls -la 02_sofascore/data/sofascore.db

# Si no existe, tienes 2 opciones:
# A) Copiarla desde la PC original (~270MB)
# B) Re-scrapear todo (toma ~7 min si la API funciona):
python 02_sofascore/scripts/01_scrape_sofascore.py --range 2019-2026
```

### Paso 3: Ver estadísticas actuales
```bash
python 02_sofascore/scripts/01_scrape_sofascore.py --stats
```

Debería mostrar:
```
=== Estadísticas de la Base de Datos ===
Total eventos: 392,686
Eventos terminados: 359,567
...
Eventos con statistics: 0    <-- ESTO ES LO QUE VAMOS A LLENAR
Eventos con lineups: 0       <-- ESTO TAMBIÉN
```

### Paso 4: Ejecutar scraping de detalles
```bash
cd 02_sofascore
python scripts/01_scrape_sofascore.py --details
```

**Qué hace este comando:**
1. Obtiene todos los `event_id` de eventos terminados que NO tienen stats/lineups
2. Para cada evento, hace 2 requests:
   - `GET /event/{id}/statistics`
   - `GET /event/{id}/lineups`
3. Guarda los datos en las tablas `statistics` y `lineups`
4. Guarda JSONs de respaldo en `02_sofascore/data/details/`

**Estimación de tiempo:**
- 359,567 eventos × 2 requests = ~719,134 requests
- Con 3 workers y 0.5s delay: ~77 req/s teóricos
- **Tiempo estimado: 2.5-3 horas**

### Paso 5: Monitorear progreso
El script muestra progreso cada 50 eventos:
```
Progreso detalles: 50/359567 (45 stats, 38 lineups)
Progreso detalles: 100/359567 (92 stats, 78 lineups)
...
```

**Si ves muchos errores 403:**
- El script pausará automáticamente 60 segundos
- Si continúan, Ctrl+C y espera unas horas antes de reintentar

### Paso 6: Verificar resultados
```bash
python scripts/01_scrape_sofascore.py --stats
```

Ahora debería mostrar:
```
Eventos con statistics: >0
Eventos con lineups: >0
```

## Notas Importantes

### Ligas Sin Stats/Lineups
Muchas ligas menores NO tienen estadísticas ni alineaciones disponibles:
- Amistosos
- Ligas amateur
- eSports
- Juveniles

**Esto es normal**. El scraper simplemente no guardará nada para esos eventos.

### Checkpoint Automático
Si el scraping se interrumpe (Ctrl+C, error de red, etc.):
- Los datos ya guardados persisten en SQLite
- Al re-ejecutar `--details`, solo procesa eventos SIN stats/lineups
- No hay que empezar de cero

### JSONs de Respaldo
Cada respuesta se guarda en:
```
02_sofascore/data/details/{event_id}_stats.json
02_sofascore/data/details/{event_id}_lineups.json
```

Esto permite:
- Debugging si algo falla
- Re-procesar sin hacer nuevos requests
- Backup de los datos crudos

## Estructura de Archivos Relevantes

```
02_sofascore/
├── scripts/
│   └── 01_scrape_sofascore.py    # Script principal (ya modificado)
├── data/
│   ├── sofascore.db              # SQLite (~270MB) - NECESITAS COPIAR ESTO
│   ├── checkpoint.json           # Estado del scraping
│   ├── raw/                      # JSONs de eventos por día (Fase 1)
│   └── details/                  # JSONs de stats/lineups (Fase 2) - SE CREARÁ
└── README.md                     # Documentación del módulo
```

## Resumen Ejecutivo

| Qué | Valor |
|-----|-------|
| **Objetivo** | Obtener statistics y lineups para 359,567 eventos |
| **Comando** | `python 02_sofascore/scripts/01_scrape_sofascore.py --details` |
| **Requisito** | Tener `sofascore.db` (~270MB) en `02_sofascore/data/` |
| **Tiempo estimado** | 2.5-3 horas |
| **Configuración** | 3 workers, 0.5s delay (conservador) |
| **Si falla con 403** | IP bloqueada, intentar desde otra red |

## Después de Completar Fase 2

Una vez que tengamos stats y lineups, podremos:
1. Calcular features adicionales para los modelos ML (tiros a puerta, posesión, etc.)
2. Analizar correlación entre stats y resultados
3. Mejorar las predicciones con datos más ricos

---

*Documento creado: 23 Enero 2026*
*Última actualización: 23 Enero 2026*
