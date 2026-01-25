# Contexto de Scraping SofaScore - Enero 2026

## Estado Actual

### Problema
- **IP bloqueada** por Cloudflare/SofaScore (error 403 "challenge")
- IP actual: `136.226.1.115`
- Bloqueo afecta tanto requests directos como Playwright

### Lo que ya tenemos
- **sofascore.db**: 392,686 eventos con cuotas 1X2 (2019-2026)
- **Script funcional**: `02_scrape_full_details.py` listo para obtener datos completos
- **Datos faltantes**: BTTS, Over/Under, estadísticas, alineaciones, incidentes

## Qué hacer cuando se desbloquee

### 1. Verificar que la API funciona
```bash
python -c "import requests; r = requests.get('https://api.sofascore.com/api/v1/event/14572796/odds/1/all', headers={'User-Agent': 'Mozilla/5.0'}, timeout=15, verify=False); print(f'Status: {r.status_code}')"
```
- Si devuelve **200**: API desbloqueada ✅
- Si devuelve **403**: Sigue bloqueada ❌

### 2. Ejecutar scraping de detalles completos
```bash
cd c:\Users\peralhe001\Documents\proyecto

# Ver estadísticas actuales
python 02_sofascore/scripts/02_scrape_full_details.py --stats

# Test con 100 eventos
python 02_sofascore/scripts/02_scrape_full_details.py --test 100

# Procesar todos los eventos (~395K, ~6 horas)
python 02_sofascore/scripts/02_scrape_full_details.py --all
```

### 3. El script obtiene por cada evento:
- `/event/{id}/odds/1/all` → TODAS las cuotas (1X2, BTTS, Over/Under, Asian, etc.)
- `/event/{id}/statistics` → Estadísticas (posesión, tiros, xG)
- `/event/{id}/lineups` → Alineaciones
- `/event/{id}/incidents` → Goles, tarjetas, sustituciones

### 4. Configuración del scraper
- **200 eventos en paralelo** × 4 endpoints = 800 requests simultáneos
- **Velocidad**: ~73 eventos/segundo
- **Pausa preventiva**: 15 min cada 2,500 eventos
- **Checkpoint**: Resume automático si se interrumpe

## Opciones para desbloquear

### Con VPN (recomendado)
1. Instalar VPN (ProtonVPN gratuito, Windscribe, etc.)
2. Conectar a servidor en otro país
3. Verificar nueva IP: `curl httpbin.org/ip`
4. Ejecutar scraper

### Sin VPN
- **Google Colab**: Ejecutar desde notebook en la nube
- **Esperar 24-48h**: Los bloqueos de Cloudflare suelen expirar

## Archivos importantes

| Archivo | Descripción |
|---------|-------------|
| `02_sofascore/scripts/01_scrape_sofascore.py` | Scraper básico (eventos + cuotas 1X2 por fecha) |
| `02_sofascore/scripts/02_scrape_full_details.py` | Scraper completo (todos los detalles por evento) |
| `02_sofascore/data/sofascore.db` | Base de datos SQLite (392K eventos) |
| `02_sofascore/data/checkpoint_details.json` | Progreso del scraping detallado |
| `02_sofascore/data/raw_details/` | JSONs crudos por evento |

## Después del scraping

1. **Procesar JSONs** → Insertar en tablas SQLite (crear script `03_process_raw_details.py`)
2. **Actualizar hybrid_db** → Re-ejecutar merge con cuotas completas
3. **Reentrenar modelo** → Con BTTS, Over/Under como features adicionales

## Notas técnicas

- La API de SofaScore NO requiere autenticación
- Headers mínimos requeridos: `User-Agent`, `Accept: application/json`, `Referer: https://www.sofascore.com/`
- Rate limiting: ~10 req/seg sin bloqueo, pero Cloudflare puede bloquear por IP
- SSL: Requiere `verify=False` en entornos corporativos