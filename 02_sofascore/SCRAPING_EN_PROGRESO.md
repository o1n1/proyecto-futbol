# Scraping de Stats y Lineups - EN PROGRESO

## Estado Actual (23 Enero 2026, ~00:40)

| Métrica | Valor |
|---------|-------|
| **Total eventos a procesar** | 172,696 |
| **Velocidad** | ~3 eventos/segundo |
| **Tiempo estimado total** | ~14-15 horas |
| **Configuración** | 3 contextos, 0.3s delay |

## Problema Resuelto

SofaScore bloqueó los requests de Python/curl mediante **TLS fingerprinting** el 23 de enero 2026.
La solución fue usar **Playwright** (navegador Chromium real) que bypasea esta protección.

## Script Creado

```
02_sofascore/scripts/02_scrape_details_playwright.py
```

### Comandos

```powershell
# Ejecutar scraping (continúa desde donde se quedó)
cd "C:\Users\Héctor\Documents\Nueva carpeta\proyecto-futbol"
python 02_sofascore/scripts/02_scrape_details_playwright.py

# Ver estadísticas sin ejecutar
python 02_sofascore/scripts/02_scrape_details_playwright.py --stats

# Ejecutar con más contextos (más rápido, más riesgo)
python 02_sofascore/scripts/02_scrape_details_playwright.py --contexts 5

# Ejecutar con menos delay (más rápido, más riesgo)
python 02_sofascore/scripts/02_scrape_details_playwright.py --delay 0.2

# Modo test (solo 10 eventos)
python 02_sofascore/scripts/02_scrape_details_playwright.py --test
```

## Archivos de Progreso

| Archivo | Descripción |
|---------|-------------|
| `02_sofascore/checkpoint_details.json` | Checkpoint con eventos completados |
| `02_sofascore/sofascore.db` | Base de datos SQLite (tablas: statistics, lineups) |
| `02_sofascore/data/details/*.json` | JSONs de respaldo por evento |

## Cómo Retomar

1. El script **guarda checkpoint cada 100 eventos**
2. Si se interrumpe (Ctrl+C, sin internet, etc.), el progreso se guarda
3. Al ejecutar de nuevo, **continúa automáticamente** desde donde se quedó
4. No hay que hacer nada especial, solo ejecutar el mismo comando

## Verificar Progreso

```powershell
# Ver cuántos eventos tienen stats en la BD
python -c "import sqlite3; c=sqlite3.connect('02_sofascore/sofascore.db').cursor(); c.execute('SELECT COUNT(DISTINCT event_id) FROM statistics'); print(f'Stats: {c.fetchone()[0]}')"

# Ver cuántos archivos JSON se han guardado
(ls 02_sofascore/data/details/).Count
```

## Requisitos

- **Hotspot móvil**: La IP de la red fija (Pirelli/casa) está bloqueada por SofaScore
- **Playwright instalado**: `pip install playwright && python -m playwright install chromium`

## Log de Ejemplo

```
2026-01-23 00:35:57 - INFO - Eventos a procesar: 172696
2026-01-23 00:35:57 - INFO - Configuración: 3 contextos, 0.3s delay
2026-01-23 00:35:57 - INFO - Iniciando scraping de 172696 eventos con 3 contextos
2026-01-23 00:35:57 - INFO - Navegador Chromium iniciado
2026-01-23 00:36:00 - INFO - Contexto 1 creado
2026-01-23 00:36:03 - INFO - Contexto 2 creado
2026-01-23 00:36:09 - INFO - Contexto 3 creado
2026-01-23 00:37:41 - INFO - Progreso: 300/172696 eventos (273 stats, 265 lineups)
2026-01-23 00:39:20 - INFO - Progreso: 600/172696 eventos (550 stats, 534 lineups)
```

## Notas Importantes

1. **NO usar la red fija** - SofaScore bloqueó esa IP
2. **El script es robusto** - Guarda progreso automáticamente
3. **Si hay errores 403** - El script pausa 60s y continúa
4. **~91% de eventos tienen stats** - Es normal que algunos no tengan (ligas menores)
5. **~88% de eventos tienen lineups** - También normal

## Después de Completar

Una vez que termine el scraping:

1. Verificar estadísticas: `python 02_sofascore/scripts/02_scrape_details_playwright.py --stats`
2. La BD tendrá ~150K+ eventos con stats y lineups
3. Se pueden calcular nuevas features para el modelo ML
4. Copiar la BD actualizada a la otra PC si es necesario

---

*Documento creado: 23 Enero 2026*
*Última actualización: 23 Enero 2026 00:45*
