"""
SofaScore Full Details Scraper
Obtiene TODA la información de cada evento: odds completas, estadísticas, alineaciones e incidentes.

Usa 200 eventos en paralelo, cada uno con 4 endpoints en paralelo = 800 requests simultáneos.
Velocidad: ~73 eventos/segundo (~1.5 horas para 395K eventos).

Uso:
    python 02_scrape_full_details.py                    # Continuar desde checkpoint
    python 02_scrape_full_details.py --year 2025       # Scrapear año específico
    python 02_scrape_full_details.py --date 2024-06-15 # Scrapear día específico
    python 02_scrape_full_details.py --stats           # Ver estadísticas
    python 02_scrape_full_details.py --test 20         # Test con N eventos

Endpoints por evento:
    - /event/{id}/odds/1/all     → Todas las cuotas (1X2, BTTS, Over/Under, etc.)
    - /event/{id}/statistics     → Estadísticas (posesión, tiros, xG, etc.)
    - /event/{id}/lineups        → Alineaciones (titulares, suplentes)
    - /event/{id}/incidents      → Incidentes (goles, tarjetas, sustituciones)
"""

import os
import json
import sqlite3
import logging
import argparse
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import urllib3

# Deshabilitar warnings de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

BASE_URL = "https://api.sofascore.com/api/v1"

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_details")
DB_FILE = os.path.join(DATA_DIR, "sofascore.db")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint_details.json")

# Paralelismo: 100 eventos en paralelo, cada uno con 4 endpoints en paralelo = 400 requests simultáneos
# Sin delay entre batches para máxima velocidad (la pausa larga controla el rate limiting)
MAX_WORKERS = 100  # Eventos en paralelo (cada uno hace 4 requests internos)
DELAY_BETWEEN_BATCHES = 0  # Sin delay, la pausa larga de 10 min cada 2500 eventos controla el ritmo
EVENTS_PER_BATCH = 500  # Eventos por batch antes de guardar checkpoint
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3

# Pausa larga para evitar bloqueo de IP
# 2500 eventos × 4 endpoints = 10,000 requests antes de pausa
EVENTS_BEFORE_LONG_PAUSE = 2500
LONG_PAUSE_DURATION = 900  # 15 minutos en segundos

# Detección y recuperación de bloqueos
BLOCK_CHECK_INTERVAL = 300  # 5 minutos entre reintentos cuando bloqueado
BLOCK_LOG_FILE = os.path.join(DATA_DIR, "block_log.json")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.sofascore.com/',
}


# ============================================================================
# BLOCK TRACKER - Logging de bloqueos para análisis
# ============================================================================

class BlockTracker:
    """Registra y analiza bloqueos de la API para determinar límites."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.data = self._load()
        self.current_block_start = None
        self.current_cycle_start = None
        self.current_cycle_number = 0

    def _load(self) -> dict:
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'blocks': [],
            'cycles': [],
            'summary': {
                'total_blocks': 0,
                'total_cycles': 0,
                'avg_block_duration_minutes': 0,
                'avg_requests_before_block': 0,
                'total_events_processed': 0,
                'total_requests_made': 0
            }
        }

    def save(self):
        # Actualizar summary
        if self.data['blocks']:
            durations = [b['duration_minutes'] for b in self.data['blocks'] if b.get('duration_minutes')]
            requests = [b['requests_before_block'] for b in self.data['blocks']]
            self.data['summary']['total_blocks'] = len(self.data['blocks'])
            self.data['summary']['avg_block_duration_minutes'] = sum(durations) / len(durations) if durations else 0
            self.data['summary']['avg_requests_before_block'] = sum(requests) / len(requests) if requests else 0

        if self.data['cycles']:
            self.data['summary']['total_cycles'] = len(self.data['cycles'])
            self.data['summary']['total_events_processed'] = sum(c['events_processed'] for c in self.data['cycles'])
            self.data['summary']['total_requests_made'] = sum(c['requests_made'] for c in self.data['cycles'])

        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, default=str)

    def start_cycle(self, cycle_number: int):
        """Inicia un nuevo ciclo de scraping."""
        self.current_cycle_number = cycle_number
        self.current_cycle_start = datetime.now()
        logger.info(f"[CYCLE {cycle_number}] Iniciando ciclo")

    def log_cycle_complete(self, events_processed: int, requests_made: int, was_blocked: bool):
        """Registra un ciclo completado."""
        duration = (datetime.now() - self.current_cycle_start).total_seconds() if self.current_cycle_start else 0

        cycle_data = {
            'cycle_number': self.current_cycle_number,
            'start_time': self.current_cycle_start.isoformat() if self.current_cycle_start else None,
            'end_time': datetime.now().isoformat(),
            'events_processed': events_processed,
            'requests_made': requests_made,
            'duration_seconds': round(duration, 1),
            'events_per_second': round(events_processed / duration, 2) if duration > 0 else 0,
            'was_blocked': was_blocked
        }

        self.data['cycles'].append(cycle_data)
        self.save()

        logger.info(f"[CYCLE {self.current_cycle_number}] Completado: {events_processed} eventos, "
                   f"{requests_made:,} requests, {duration:.0f}s, blocked={was_blocked}")

    def log_block_start(self, cycle_number: int, events_processed: int, requests_made: int):
        """Registra el inicio de un bloqueo."""
        self.current_block_start = datetime.now()

        block_data = {
            'cycle_number': cycle_number,
            'start_time': self.current_block_start.isoformat(),
            'end_time': None,
            'duration_minutes': None,
            'events_before_block': events_processed,
            'requests_before_block': requests_made,
            'check_attempts': 0
        }

        self.data['blocks'].append(block_data)
        self.save()

        logger.warning(f"[BLOCK] Bloqueo detectado en ciclo {cycle_number} después de "
                      f"{events_processed} eventos ({requests_made:,} requests)")

    def log_block_check(self, attempt: int, status_code: int, is_unblocked: bool):
        """Registra un intento de verificación durante bloqueo."""
        if self.data['blocks']:
            self.data['blocks'][-1]['check_attempts'] = attempt

        logger.info(f"[BLOCK] Intento {attempt}: status={status_code}, unblocked={is_unblocked}")

    def log_block_end(self):
        """Registra el fin de un bloqueo."""
        if not self.data['blocks'] or not self.current_block_start:
            return

        end_time = datetime.now()
        duration = (end_time - self.current_block_start).total_seconds() / 60

        self.data['blocks'][-1]['end_time'] = end_time.isoformat()
        self.data['blocks'][-1]['duration_minutes'] = round(duration, 2)
        self.save()

        logger.info(f"[BLOCK] Desbloqueado después de {duration:.1f} minutos")
        self.current_block_start = None

    def get_total_blocks(self) -> int:
        return len(self.data['blocks'])

    def get_stats(self) -> dict:
        return self.data['summary']


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Maneja el progreso del scraping para permitir resume."""

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'completed_events': [],
            'failed_events': [],
            'total_processed': 0,
            'total_odds': 0,
            'total_stats': 0,
            'total_lineups': 0,
            'total_incidents': 0,
            'started_at': datetime.now().isoformat(),
            'last_updated': None
        }

    def save(self):
        self.data['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, default=str)

    def mark_completed(self, event_id: int, has_odds: bool, has_stats: bool,
                       has_lineups: bool, has_incidents: bool):
        if event_id not in self.data['completed_events']:
            self.data['completed_events'].append(event_id)
        self.data['total_processed'] += 1
        if has_odds:
            self.data['total_odds'] += 1
        if has_stats:
            self.data['total_stats'] += 1
        if has_lineups:
            self.data['total_lineups'] += 1
        if has_incidents:
            self.data['total_incidents'] += 1

    def mark_failed(self, event_id: int, error: str):
        self.data['failed_events'].append({
            'event_id': event_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

    def is_completed(self, event_id: int) -> bool:
        return event_id in self.data['completed_events']

    def get_stats(self) -> dict:
        return {
            'completed': len(self.data['completed_events']),
            'failed': len(self.data['failed_events']),
            'total_processed': self.data['total_processed'],
            'total_odds': self.data['total_odds'],
            'total_stats': self.data['total_stats'],
            'total_lineups': self.data['total_lineups'],
            'total_incidents': self.data['total_incidents'],
            'started_at': self.data['started_at'],
            'last_updated': self.data['last_updated']
        }


# ============================================================================
# SOFASCORE CLIENT
# ============================================================================

class SofaScoreClient:
    """Cliente para la API de SofaScore."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        # Aumentar el pool de conexiones para soportar 200 workers
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=250,
            pool_maxsize=250,
            max_retries=3
        )
        self.session.mount('https://', adapter)

    def _make_request(self, url: str, retries: int = MAX_RETRIES) -> Optional[dict]:
        """
        Hace un request a la API.
        Si detecta 403/429, retorna dict especial con flag 'blocked'.
        """
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT, verify=False)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return None  # No data available
                elif response.status_code == 429 or response.status_code == 403:
                    # Rate limited/blocked - retornar dict especial para detección
                    if attempt == 0:
                        logger.warning(f"HTTP {response.status_code} - blocked/rate limited")
                    # Retornar marcador de bloqueo en lugar de None
                    return {'__blocked__': True, '__status__': response.status_code}
                else:
                    if attempt == retries - 1:
                        logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.exceptions.Timeout:
                if attempt == retries - 1:
                    logger.warning(f"Timeout: {url}")
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Error: {e}")

            if attempt < retries - 1:
                time.sleep(1)

        return None

    def get_event_odds(self, event_id: int) -> Optional[dict]:
        """Obtiene TODAS las cuotas de un evento."""
        url = f"{BASE_URL}/event/{event_id}/odds/1/all"
        return self._make_request(url)

    def get_event_statistics(self, event_id: int) -> Optional[dict]:
        """Obtiene estadísticas de un evento."""
        url = f"{BASE_URL}/event/{event_id}/statistics"
        return self._make_request(url)

    def get_event_lineups(self, event_id: int) -> Optional[dict]:
        """Obtiene alineaciones de un evento."""
        url = f"{BASE_URL}/event/{event_id}/lineups"
        return self._make_request(url)

    def get_event_incidents(self, event_id: int) -> Optional[dict]:
        """Obtiene incidentes de un evento."""
        url = f"{BASE_URL}/event/{event_id}/incidents"
        return self._make_request(url)

    def get_scheduled_events(self, date: str) -> Optional[dict]:
        """Obtiene todos los eventos de una fecha desde la API."""
        url = f"{BASE_URL}/sport/football/scheduled-events/{date}"
        return self._make_request(url)

    def check_api_status(self) -> tuple:
        """
        Verifica si la API está accesible con un request de prueba.
        Returns: (is_accessible: bool, status_code: int)
        """
        url = f"{BASE_URL}/event/14572796/odds/1/all"  # Evento conocido
        try:
            response = self.session.get(url, timeout=15, verify=False)
            return (response.status_code == 200, response.status_code)
        except Exception as e:
            logger.error(f"Error verificando API: {e}")
            return (False, 0)


# ============================================================================
# SCRAPER
# ============================================================================

class FullDetailsScraper:
    """Scraper para obtener todos los detalles de eventos."""

    def __init__(self):
        os.makedirs(RAW_DIR, exist_ok=True)
        self.client = SofaScoreClient()
        self.checkpoint = CheckpointManager(CHECKPOINT_FILE)
        self.block_tracker = BlockTracker(BLOCK_LOG_FILE)
        self.db_conn = sqlite3.connect(DB_FILE)
        self.is_blocked = False  # Flag para detectar bloqueo

    def wait_for_unblock(self, cycle_number: int, events_processed: int, requests_made: int):
        """
        Espera hasta que la API esté desbloqueada, verificando cada 5 minutos.
        Registra todo en el block_tracker para análisis.
        """
        self.block_tracker.log_block_start(cycle_number, events_processed, requests_made)
        self.checkpoint.save()

        attempt = 1
        while True:
            logger.info(f"[BLOCK] Esperando {BLOCK_CHECK_INTERVAL // 60} minutos antes de verificar (intento {attempt})...")

            # Cuenta regresiva cada minuto
            for remaining in range(BLOCK_CHECK_INTERVAL // 60, 0, -1):
                logger.info(f"[BLOCK]   Verificando en {remaining} minutos...")
                time.sleep(60)

            # Verificar si está desbloqueado
            is_accessible, status_code = self.client.check_api_status()
            self.block_tracker.log_block_check(attempt, status_code, is_accessible)

            if is_accessible:
                self.block_tracker.log_block_end()
                self.is_blocked = False
                logger.info(f"[BLOCK] API desbloqueada después de {attempt * BLOCK_CHECK_INTERVAL // 60} minutos")
                return

            attempt += 1

    def get_events_from_api_for_date(self, date: str) -> List[Tuple[int, str]]:
        """Obtiene eventos de la API para una fecha específica."""
        data = self.client.get_scheduled_events(date)
        if not data:
            return []

        events = data.get('events', [])
        results = []
        for event in events:
            event_id = event.get('id')
            if event_id:
                results.append((event_id, date))
        return results

    def get_events_from_api_range(self, start_date: str, end_date: str) -> List[Tuple[int, str]]:
        """Obtiene eventos de la API para un rango de fechas (de más reciente a más antiguo)."""
        all_events = []
        current = datetime.strptime(end_date, '%Y-%m-%d')
        end = datetime.strptime(start_date, '%Y-%m-%d')

        while current >= end:
            date_str = current.strftime('%Y-%m-%d')
            logger.info(f"Obteniendo eventos de API para {date_str}...")
            events = self.get_events_from_api_for_date(date_str)
            all_events.extend(events)
            logger.info(f"  -> {len(events)} eventos")
            current -= timedelta(days=1)

        return all_events

    def get_all_events(self, year: int = None, date: str = None,
                       limit: int = None) -> List[Tuple[int, str]]:
        """
        Obtiene event_ids combinando API (fechas futuras) + DB (históricas).
        Ordena de más reciente a más antiguo (31/01/2026 -> 2018).
        """
        cursor = self.db_conn.cursor()

        if date:
            # Fecha específica: primero intentar API, luego DB
            events_api = self.get_events_from_api_for_date(date)
            if events_api:
                return events_api[:limit] if limit else events_api

            query = "SELECT event_id, date FROM events WHERE date = ? ORDER BY date DESC"
            cursor.execute(query, (date,))
            results = cursor.fetchall()
            return results[:limit] if limit else results

        elif year:
            query = """
                SELECT event_id, date FROM events
                WHERE date LIKE ?
                ORDER BY date DESC
            """
            cursor.execute(query, (f"{year}%",))
            results = cursor.fetchall()
            return results[:limit] if limit else results

        else:
            # Obtener TODOS: API (31/01 -> 24/01) + DB (23/01 -> 2018)
            all_events = []

            # 1. Obtener eventos futuros de API (31/01/2026 hasta día después del último en DB)
            cursor.execute("SELECT MAX(date) FROM events")
            max_date_db = cursor.fetchone()[0]  # ej: 2026-01-23

            if max_date_db:
                # Día siguiente al último en DB
                next_day = (datetime.strptime(max_date_db, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

                # Obtener eventos desde API para fechas que no están en DB
                if next_day <= '2026-01-31':
                    logger.info(f"Obteniendo eventos de API: {next_day} -> 2026-01-31")
                    api_events = self.get_events_from_api_range(next_day, '2026-01-31')
                    all_events.extend(api_events)
                    logger.info(f"Total eventos de API: {len(api_events)}")

            # 2. Obtener eventos de DB (ordenados DESC)
            query = """
                SELECT event_id, date FROM events
                ORDER BY date DESC
            """
            cursor.execute(query)
            db_events = cursor.fetchall()
            all_events.extend(db_events)

            logger.info(f"Total eventos combinados: {len(all_events)}")

            return all_events[:limit] if limit else all_events

    def fetch_event_details(self, event_id: int) -> Dict:
        """
        Obtiene todos los detalles de un evento (4 endpoints en paralelo).
        Retorna dict con datos y flag de bloqueo.
        """
        result = {
            'event_id': event_id,
            'odds': None,
            'statistics': None,
            'lineups': None,
            'incidents': None,
            'scraped_at': datetime.now().isoformat(),
            'blocked': False,  # Flag para detección de bloqueo
            'block_count': 0   # Cuántos endpoints retornaron 403
        }

        block_count = 0

        # Hacer los 4 requests en paralelo para cada evento
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.client.get_event_odds, event_id): 'odds',
                executor.submit(self.client.get_event_statistics, event_id): 'statistics',
                executor.submit(self.client.get_event_lineups, event_id): 'lineups',
                executor.submit(self.client.get_event_incidents, event_id): 'incidents',
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    data = future.result()
                    # Verificar si es respuesta de bloqueo
                    if isinstance(data, dict) and data.get('__blocked__'):
                        block_count += 1
                        result[key] = None
                    else:
                        result[key] = data
                except Exception:
                    result[key] = None

        # Marcar como bloqueado si algún endpoint retornó 403
        result['block_count'] = block_count
        result['blocked'] = block_count > 0

        return result

    def save_to_raw(self, event_id: int, date: str, data: Dict):
        """Guarda los datos en archivos JSON en raw_details."""
        if not date:
            date = "unknown"

        year = date[:4] if len(date) >= 4 else "unknown"
        year_dir = os.path.join(RAW_DIR, year)
        os.makedirs(year_dir, exist_ok=True)

        # Guardar archivo por evento
        filename = f"{date}_{event_id}.json"
        filepath = os.path.join(year_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def process_events(self, events: List[Tuple[int, str]], show_progress: bool = True):
        """Procesa una lista de eventos en paralelo con rate limiting y detección de bloqueo."""
        # Filtrar eventos ya completados
        pending_events = [
            (eid, date) for eid, date in events
            if not self.checkpoint.is_completed(eid)
        ]

        if not pending_events:
            logger.info("No hay eventos pendientes de procesar")
            return

        total = len(pending_events)
        logger.info(f"Eventos a procesar: {total}")
        logger.info(f"Eventos ya completados: {len(events) - total}")
        logger.info(f"Config: {MAX_WORKERS} eventos en paralelo, {DELAY_BETWEEN_BATCHES}s delay entre batches")
        logger.info(f"Pausa larga: {LONG_PAUSE_DURATION//60} minutos cada {EVENTS_BEFORE_LONG_PAUSE} eventos ({EVENTS_BEFORE_LONG_PAUSE * 4:,} requests)")
        logger.info(f"Detección de bloqueo: check cada {BLOCK_CHECK_INTERVAL//60} minutos si bloqueado")

        processed = 0
        processed_since_pause = 0  # Contador para pausa larga
        requests_since_pause = 0   # Contador de requests por ciclo
        cycle_number = 1
        cycle_was_blocked = False
        start_time = time.time()

        self.block_tracker.start_cycle(cycle_number)

        # Procesar en mini-batches de MAX_WORKERS eventos con delay entre ellos
        for batch_start in range(0, total, EVENTS_PER_BATCH):
            checkpoint_batch = pending_events[batch_start:batch_start + EVENTS_PER_BATCH]

            # Dividir el checkpoint_batch en mini-batches de MAX_WORKERS
            for mini_batch_start in range(0, len(checkpoint_batch), MAX_WORKERS):
                mini_batch = checkpoint_batch[mini_batch_start:mini_batch_start + MAX_WORKERS]

                # Almacenar resultados del mini-batch para detectar bloqueo
                batch_results = []
                blocked_count = 0

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Crear diccionario de futures con (event_id, date)
                    future_to_event = {
                        executor.submit(self.fetch_event_details, eid): (eid, date)
                        for eid, date in mini_batch
                    }

                    for future in as_completed(future_to_event):
                        event_id, date = future_to_event[future]

                        try:
                            result = future.result()
                            batch_results.append(result)

                            # Contar eventos bloqueados
                            if result.get('blocked'):
                                blocked_count += 1

                            # Guardar en raw (incluso si está parcialmente bloqueado)
                            self.save_to_raw(event_id, date, result)

                            # Actualizar checkpoint
                            self.checkpoint.mark_completed(
                                event_id,
                                has_odds=result['odds'] is not None,
                                has_stats=result['statistics'] is not None,
                                has_lineups=result['lineups'] is not None,
                                has_incidents=result['incidents'] is not None
                            )

                            processed += 1
                            processed_since_pause += 1
                            requests_since_pause += 4  # 4 endpoints por evento

                            # Mostrar progreso cada 100 eventos
                            if show_progress and processed % 100 == 0:
                                elapsed = time.time() - start_time
                                rate = processed / elapsed
                                eta = (total - processed) / rate if rate > 0 else 0
                                logger.info(
                                    f"Progreso: {processed}/{total} ({processed/total*100:.1f}%) | "
                                    f"Velocidad: {rate:.1f} eventos/s | "
                                    f"ETA: {eta/3600:.1f}h | "
                                    f"Ciclo {cycle_number}: {processed_since_pause}/{EVENTS_BEFORE_LONG_PAUSE} | "
                                    f"Requests: {requests_since_pause:,} | "
                                    f"Bloqueos: {self.block_tracker.get_total_blocks()}"
                                )

                        except Exception as e:
                            self.checkpoint.mark_failed(event_id, str(e))
                            logger.error(f"Error procesando evento {event_id}: {e}")

                # Detectar bloqueo: si >50% del batch tiene endpoints bloqueados
                if blocked_count > len(mini_batch) * 0.5:
                    logger.warning(f"[BLOCK] Detectado bloqueo en ciclo {cycle_number}: "
                                  f"{blocked_count}/{len(mini_batch)} eventos con 403")
                    cycle_was_blocked = True

                    # Guardar ciclo actual (incompleto, con bloqueo)
                    self.block_tracker.log_cycle_complete(
                        processed_since_pause, requests_since_pause, was_blocked=True
                    )

                    # Esperar hasta desbloqueo
                    self.wait_for_unblock(cycle_number, processed_since_pause, requests_since_pause)

                    # Iniciar nuevo ciclo
                    cycle_number += 1
                    processed_since_pause = 0
                    requests_since_pause = 0
                    cycle_was_blocked = False
                    self.block_tracker.start_cycle(cycle_number)

                # Delay entre mini-batches para evitar rate limiting
                time.sleep(DELAY_BETWEEN_BATCHES)

                # Verificar si necesitamos pausa larga preventiva
                if processed_since_pause >= EVENTS_BEFORE_LONG_PAUSE:
                    # Registrar ciclo completado exitosamente
                    self.block_tracker.log_cycle_complete(
                        processed_since_pause, requests_since_pause, was_blocked=False
                    )

                    self.checkpoint.save()
                    logger.info("=" * 60)
                    logger.info(f"[CYCLE {cycle_number}] COMPLETADO: {processed_since_pause} eventos, {requests_since_pause:,} requests")
                    logger.info(f"PAUSA PREVENTIVA: {LONG_PAUSE_DURATION // 60} minutos")
                    logger.info(f"Progreso total: {processed}/{total} ({processed/total*100:.1f}%)")
                    logger.info(f"Bloqueos totales hasta ahora: {self.block_tracker.get_total_blocks()}")
                    logger.info("=" * 60)

                    # Mostrar cuenta regresiva cada minuto
                    for remaining in range(LONG_PAUSE_DURATION // 60, 0, -1):
                        logger.info(f"  Reanudando en {remaining} minutos...")
                        time.sleep(60)

                    # Iniciar nuevo ciclo
                    cycle_number += 1
                    processed_since_pause = 0
                    requests_since_pause = 0
                    cycle_was_blocked = False
                    self.block_tracker.start_cycle(cycle_number)

                    logger.info("=" * 60)
                    logger.info(f"[CYCLE {cycle_number}] INICIANDO...")
                    logger.info("=" * 60)

            # Guardar checkpoint después de cada batch
            self.checkpoint.save()
            logger.info(f"Checkpoint guardado ({processed}/{total})")

        # Registrar último ciclo si quedó incompleto
        if processed_since_pause > 0:
            self.block_tracker.log_cycle_complete(
                processed_since_pause, requests_since_pause, was_blocked=cycle_was_blocked
            )

        # Resumen final
        elapsed = time.time() - start_time
        block_stats = self.block_tracker.get_stats()

        logger.info("=" * 60)
        logger.info("SCRAPING COMPLETADO")
        logger.info("=" * 60)
        logger.info(f"Eventos procesados: {processed} en {elapsed/3600:.2f} horas")
        logger.info(f"Velocidad promedio: {processed/elapsed:.1f} eventos/segundo")
        logger.info("-" * 60)
        logger.info("ESTADISTICAS DE BLOQUEOS:")
        logger.info(f"  Total ciclos: {block_stats['total_cycles']}")
        logger.info(f"  Total bloqueos: {block_stats['total_blocks']}")
        logger.info(f"  Duración promedio bloqueo: {block_stats['avg_block_duration_minutes']:.1f} minutos")
        logger.info(f"  Requests promedio antes de bloqueo: {block_stats['avg_requests_before_block']:,.0f}")
        logger.info("-" * 60)

        cp_stats = self.checkpoint.get_stats()
        logger.info("DATOS OBTENIDOS:")
        logger.info(f"  Con odds: {cp_stats['total_odds']}")
        logger.info(f"  Con stats: {cp_stats['total_stats']}")
        logger.info(f"  Con lineups: {cp_stats['total_lineups']}")
        logger.info(f"  Con incidents: {cp_stats['total_incidents']}")
        logger.info("=" * 60)

    def print_stats(self):
        """Imprime estadísticas actuales."""
        # Stats del checkpoint
        cp_stats = self.checkpoint.get_stats()

        # Stats de la DB
        cursor = self.db_conn.cursor()

        # Total eventos hasta 31/01/2026 (lo que vamos a procesar)
        cursor.execute("SELECT COUNT(*) FROM events WHERE date <= '2026-01-31'")
        total_to_process = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM events WHERE status = 'finished'")
        total_finished = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM events")
        total_events = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(date), MAX(date) FROM events WHERE date <= '2026-01-31'")
        date_range = cursor.fetchone()

        # Contar archivos en raw_details
        raw_files = 0
        if os.path.exists(RAW_DIR):
            for year_dir in os.listdir(RAW_DIR):
                year_path = os.path.join(RAW_DIR, year_dir)
                if os.path.isdir(year_path):
                    raw_files += len([f for f in os.listdir(year_path) if f.endswith('.json')])

        print("\n" + "=" * 60)
        print("ESTADISTICAS DE SCRAPING DE DETALLES")
        print("=" * 60)
        print(f"Total eventos en DB:      {total_events:,}")
        print(f"Eventos a procesar:       {total_to_process:,} (hasta 31/01/2026)")
        print(f"Eventos terminados:       {total_finished:,}")
        print(f"Rango de fechas:          {date_range[0]} -> {date_range[1]}")
        print("-" * 60)
        print("PROGRESO")
        print("-" * 60)
        print(f"Eventos procesados:       {cp_stats['completed']:,}")
        print(f"Eventos fallidos:         {cp_stats['failed']:,}")
        print(f"Pendientes:               {total_to_process - cp_stats['completed']:,}")
        print(f"Progreso:                 {cp_stats['completed']/total_to_process*100:.1f}%")
        print("-" * 60)
        print("DATOS OBTENIDOS")
        print("-" * 60)
        print(f"Con odds completas:       {cp_stats['total_odds']:,}")
        print(f"Con estadísticas:         {cp_stats['total_stats']:,}")
        print(f"Con alineaciones:         {cp_stats['total_lineups']:,}")
        print(f"Con incidentes:           {cp_stats['total_incidents']:,}")
        print(f"Archivos en raw_details:  {raw_files:,}")
        print("-" * 60)
        print(f"Iniciado:                 {cp_stats['started_at']}")
        print(f"Última actualización:     {cp_stats['last_updated']}")
        print("=" * 60)

    def close(self):
        self.db_conn.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SofaScore Full Details Scraper')
    parser.add_argument('--year', type=int, help='Scrapear año específico')
    parser.add_argument('--date', type=str, help='Scrapear día específico (YYYY-MM-DD)')
    parser.add_argument('--stats', action='store_true', help='Mostrar estadísticas')
    parser.add_argument('--test', type=int, help='Test con N eventos')
    parser.add_argument('--all', action='store_true', help='Procesar todos los eventos')

    args = parser.parse_args()

    scraper = FullDetailsScraper()

    try:
        if args.stats:
            scraper.print_stats()

        elif args.test:
            logger.info(f"=== TEST: Procesando {args.test} eventos ===")
            events = scraper.get_all_events(limit=args.test)
            scraper.process_events(events)

        elif args.date:
            logger.info(f"=== Procesando fecha: {args.date} ===")
            events = scraper.get_all_events(date=args.date)
            scraper.process_events(events)

        elif args.year:
            logger.info(f"=== Procesando año: {args.year} ===")
            events = scraper.get_all_events(year=args.year)
            scraper.process_events(events)

        elif args.all:
            logger.info("=== Procesando TODOS los eventos (31/01/2026 -> 2018) ===")
            events = scraper.get_all_events()
            scraper.process_events(events)

        else:
            # Por defecto mostrar stats
            scraper.print_stats()
            print("\nUso:")
            print("  --all          Procesar todos (API 31/01->24/01 + DB)")
            print("  --year 2025    Procesar ano especifico")
            print("  --date 2025-01-20  Procesar fecha especifica")
            print("  --test 100     Test con N eventos")
            print("  --stats        Mostrar estadisticas")

    finally:
        scraper.close()


if __name__ == "__main__":
    main()
