"""
SofaScore Stats & Lineups Scraper usando Playwright

Este script usa Playwright (navegador Chromium real) para evitar el bloqueo
de TLS fingerprinting que SofaScore implementó.

Uso:
    python 02_scrape_details_playwright.py              # Scrapear ligas prioritarias
    python 02_scrape_details_playwright.py --all        # Scrapear todos los eventos
    python 02_scrape_details_playwright.py --test       # Probar con 10 eventos
    python 02_scrape_details_playwright.py --stats      # Ver estadísticas

Configuración:
    - 3 contextos paralelos (pestañas independientes)
    - 0.3s delay entre requests
    - Checkpoint cada 100 eventos para poder resumir
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import time
import asyncio
import signal
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

# Variable global para manejo de interrupción
INTERRUPTED = False

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DB_FILE = os.path.join(BASE_DIR, "sofascore.db")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint_details.json")
DETAILS_DIR = os.path.join(BASE_DIR, "data", "details")

# Configuración de scraping
NUM_CONTEXTS = 3           # Número de contextos paralelos
REQUEST_DELAY = 0.3        # Delay entre requests (segundos)
BATCH_SIZE = 100           # Guardar cada N eventos
MAX_ERRORS = 5             # Errores seguidos antes de pausar
PAUSE_ON_ERROR = 30        # Segundos de pausa por error
PAUSE_ON_BLOCK = 60        # Segundos de pausa si detectamos bloqueo

# API URLs
BASE_URL = "https://www.sofascore.com/api/v1"

# Ligas prioritarias (IDs de SofaScore - tournament.uniqueTournament.id)
# Estas son las ligas del modelo ML de predicción
LIGAS_PRIORITARIAS_NOMBRES = [
    # Ligas de estrategia
    "Emperor Cup",              # Japan
    "Primera División",         # Bolivia (344 en api-football)
    "World Cup - Qualification Europe",
    "Super League",             # China
    "Major League Soccer",      # USA
    "Serie A",                  # Brazil (diferente de Italia)
    "Copa Colombia",
    "UEFA Champions League",
    "Eliteserien",              # Norway
    "UEFA Europa Conference League",
    "K League 1",               # South Korea
    "Liga I",                   # Romania
    "Primera División",         # Chile
    "Liga MX",                  # Mexico
    "Premier Division",         # Ireland
    "Liga Pro",                 # Ecuador
    "Primera A",                # Colombia
    # Ligas top europeas
    "Premier League",           # England
    "LaLiga",                   # Spain
    "Serie A",                  # Italy
    "Bundesliga",               # Germany
    "Ligue 1",                  # France
    "Primeira Liga",            # Portugal
    "Eredivisie",               # Netherlands
]

# Países de las ligas prioritarias para filtrar
PAISES_PRIORITARIOS = [
    "Japan", "Bolivia", "World", "China", "USA", "Brazil", "Colombia",
    "Europe", "Norway", "South Korea", "Romania", "Chile", "Mexico",
    "Ireland", "Ecuador", "England", "Spain", "Italy", "Germany",
    "France", "Portugal", "Netherlands"
]


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
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            'completed_events': [],
            'failed_events': [],
            'stats_count': 0,
            'lineups_count': 0,
            'started_at': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat()
        }

    def save(self):
        self.data['last_update'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)

    def mark_completed(self, event_id: int, has_stats: bool, has_lineups: bool):
        if event_id not in self.data['completed_events']:
            self.data['completed_events'].append(event_id)
        if has_stats:
            self.data['stats_count'] += 1
        if has_lineups:
            self.data['lineups_count'] += 1

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
            'stats_scraped': self.data['stats_count'],
            'lineups_scraped': self.data['lineups_count']
        }


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Maneja la base de datos SQLite."""

    def __init__(self, db_file: str):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.conn.execute("PRAGMA journal_mode=WAL")

    def get_events_without_stats(self, limit: int = None, prioritarias_only: bool = True) -> List[int]:
        """Obtiene eventos terminados sin estadísticas."""
        cursor = self.conn.cursor()

        query = """
            SELECT e.event_id
            FROM events e
            LEFT JOIN statistics s ON e.event_id = s.event_id
            WHERE s.event_id IS NULL
            AND e.status = 'finished'
        """

        if prioritarias_only:
            # Filtrar por países prioritarios
            placeholders = ','.join('?' * len(PAISES_PRIORITARIOS))
            query += f" AND e.country IN ({placeholders})"
            params = PAISES_PRIORITARIOS
        else:
            params = []

        query += " ORDER BY e.date DESC"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]

    def insert_statistics(self, event_id: int, stats_data: dict) -> int:
        """Inserta estadísticas de un evento."""
        count = 0
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM statistics WHERE event_id = ?", (event_id,))

            for period_data in stats_data.get('statistics', []):
                period = period_data.get('period', 'ALL')

                for group in period_data.get('groups', []):
                    for item in group.get('statisticsItems', []):
                        cursor.execute("""
                            INSERT INTO statistics (event_id, period, stat_name, stat_key,
                                                   home_value, away_value)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            event_id,
                            period,
                            item.get('name'),
                            item.get('key'),
                            item.get('homeValue'),
                            item.get('awayValue')
                        ))
                        count += 1

            return count
        except Exception as e:
            logger.error(f"Error inserting statistics for event {event_id}: {e}")
            return 0

    def insert_lineups(self, event_id: int, lineups_data: dict) -> int:
        """Inserta alineaciones de un evento."""
        count = 0
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM lineups WHERE event_id = ?", (event_id,))

            for team_type in ['home', 'away']:
                team_data = lineups_data.get(team_type, {})
                team_id = team_data.get('team', {}).get('id')

                for player_type in ['players', 'substitutes']:
                    is_starter = 1 if player_type == 'players' else 0

                    for player_info in team_data.get(player_type, []):
                        player = player_info.get('player', {})

                        cursor.execute("""
                            INSERT INTO lineups (event_id, team_type, team_id, player_id,
                                                player_name, position, jersey_number, is_starter)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            event_id,
                            team_type,
                            team_id,
                            player.get('id'),
                            player.get('name'),
                            player_info.get('position'),
                            player_info.get('jerseyNumber'),
                            is_starter
                        ))
                        count += 1

            return count
        except Exception as e:
            logger.error(f"Error inserting lineups for event {event_id}: {e}")
            return 0

    def commit(self):
        self.conn.commit()

    def get_db_stats(self) -> dict:
        """Obtiene estadísticas de la BD."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM events")
        total_events = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM events WHERE status = 'finished'")
        finished_events = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT event_id) FROM statistics")
        events_with_stats = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT event_id) FROM lineups")
        events_with_lineups = cursor.fetchone()[0]

        # Eventos prioritarios
        placeholders = ','.join('?' * len(PAISES_PRIORITARIOS))
        cursor.execute(f"""
            SELECT COUNT(*) FROM events
            WHERE status = 'finished' AND country IN ({placeholders})
        """, PAISES_PRIORITARIOS)
        prioritarios_finished = cursor.fetchone()[0]

        cursor.execute(f"""
            SELECT COUNT(DISTINCT s.event_id)
            FROM statistics s
            JOIN events e ON s.event_id = e.event_id
            WHERE e.country IN ({placeholders})
        """, PAISES_PRIORITARIOS)
        prioritarios_with_stats = cursor.fetchone()[0]

        return {
            'total_events': total_events,
            'finished_events': finished_events,
            'events_with_stats': events_with_stats,
            'events_with_lineups': events_with_lineups,
            'prioritarios_finished': prioritarios_finished,
            'prioritarios_with_stats': prioritarios_with_stats,
            'prioritarios_pending': prioritarios_finished - prioritarios_with_stats
        }

    def close(self):
        self.conn.close()


# ============================================================================
# PLAYWRIGHT SCRAPER
# ============================================================================

async def scrape_with_playwright(
    event_ids: List[int],
    db: DatabaseManager,
    checkpoint: CheckpointManager,
    num_contexts: int = NUM_CONTEXTS,
    request_delay: float = REQUEST_DELAY
):
    """
    Scraping principal usando Playwright con múltiples contextos.
    """
    from playwright.async_api import async_playwright

    total_events = len(event_ids)
    logger.info(f"Iniciando scraping de {total_events} eventos con {num_contexts} contextos")

    # Crear directorio para JSONs de respaldo
    os.makedirs(DETAILS_DIR, exist_ok=True)

    stats_total = 0
    lineups_total = 0
    errors_count = 0

    async with async_playwright() as p:
        # Lanzar navegador
        browser = await p.chromium.launch(headless=True)
        logger.info("Navegador Chromium iniciado")

        # Crear contextos (pestañas independientes)
        contexts = []
        pages = []
        for i in range(num_contexts):
            ctx = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36'
            )
            page = await ctx.new_page()
            # Navegar a SofaScore para obtener cookies/tokens
            await page.goto('https://www.sofascore.com/', timeout=30000)
            await page.wait_for_timeout(1000)  # Esperar 1s para que cargue JS
            contexts.append(ctx)
            pages.append(page)
            logger.info(f"Contexto {i+1} creado")

        # Función para scrapear un evento
        async def scrape_event(page, event_id: int) -> Tuple[int, Optional[dict], Optional[dict], str]:
            """Scrapea stats y lineups de un evento."""
            stats_data = None
            lineups_data = None
            error = ""

            try:
                # Obtener statistics
                stats_url = f"{BASE_URL}/event/{event_id}/statistics"
                stats_response = await page.request.get(stats_url)

                if stats_response.status == 200:
                    stats_data = await stats_response.json()
                    # Guardar JSON de respaldo
                    stats_file = os.path.join(DETAILS_DIR, f"{event_id}_stats.json")
                    with open(stats_file, 'w', encoding='utf-8') as f:
                        json.dump(stats_data, f, ensure_ascii=False)
                elif stats_response.status == 403:
                    error = "403_stats"
                elif stats_response.status != 404:
                    error = f"stats_{stats_response.status}"

                # Delay entre requests
                await asyncio.sleep(request_delay)

                # Obtener lineups
                lineups_url = f"{BASE_URL}/event/{event_id}/lineups"
                lineups_response = await page.request.get(lineups_url)

                if lineups_response.status == 200:
                    lineups_data = await lineups_response.json()
                    # Guardar JSON de respaldo
                    lineups_file = os.path.join(DETAILS_DIR, f"{event_id}_lineups.json")
                    with open(lineups_file, 'w', encoding='utf-8') as f:
                        json.dump(lineups_data, f, ensure_ascii=False)
                elif lineups_response.status == 403:
                    if error:
                        error += "_403_lineups"
                    else:
                        error = "403_lineups"
                elif lineups_response.status != 404:
                    if error:
                        error += f"_lineups_{lineups_response.status}"
                    else:
                        error = f"lineups_{lineups_response.status}"

            except Exception as e:
                error = str(e)

            return event_id, stats_data, lineups_data, error

        # Procesar eventos distribuyendo entre contextos
        processed = 0
        batch_stats = 0
        batch_lineups = 0

        # Crear cola de eventos
        event_queue = list(event_ids)

        while event_queue and not INTERRUPTED:
            # Tomar un batch de eventos (uno por contexto)
            batch = []
            for i in range(min(num_contexts, len(event_queue))):
                event_id = event_queue.pop(0)
                if not checkpoint.is_completed(event_id):
                    batch.append((pages[i], event_id))

            if not batch:
                continue

            # Ejecutar requests en paralelo
            tasks = [scrape_event(page, eid) for page, eid in batch]
            results = await asyncio.gather(*tasks)

            # Procesar resultados
            for event_id, stats_data, lineups_data, error in results:
                has_stats = False
                has_lineups = False

                if error and "403" in error:
                    errors_count += 1
                    if errors_count >= MAX_ERRORS:
                        logger.warning(f"Muchos errores 403, pausando {PAUSE_ON_BLOCK}s...")
                        await asyncio.sleep(PAUSE_ON_BLOCK)
                        errors_count = 0
                else:
                    errors_count = 0

                # Insertar en BD
                if stats_data and stats_data.get('statistics'):
                    count = db.insert_statistics(event_id, stats_data)
                    if count > 0:
                        stats_total += 1
                        batch_stats += 1
                        has_stats = True

                if lineups_data and (lineups_data.get('home') or lineups_data.get('away')):
                    count = db.insert_lineups(event_id, lineups_data)
                    if count > 0:
                        lineups_total += 1
                        batch_lineups += 1
                        has_lineups = True

                # Marcar como completado
                checkpoint.mark_completed(event_id, has_stats, has_lineups)
                if error and "403" in error:
                    checkpoint.mark_failed(event_id, error)

                processed += 1

            # Commit y checkpoint cada BATCH_SIZE eventos
            if processed % BATCH_SIZE == 0:
                db.commit()
                checkpoint.save()
                logger.info(f"Progreso: {processed}/{total_events} eventos ({stats_total} stats, {lineups_total} lineups)")
                batch_stats = 0
                batch_lineups = 0

            # Delay entre batches
            await asyncio.sleep(request_delay)

        # Commit final (siempre guardar, incluso si fue interrumpido)
        db.commit()
        checkpoint.save()
        logger.info("Checkpoint guardado")

        # Cerrar contextos y navegador
        for ctx in contexts:
            await ctx.close()
        await browser.close()

    if INTERRUPTED:
        logger.info(f"=== SCRAPING INTERRUMPIDO (Ctrl+C) ===")
        logger.info(f"Progreso guardado. Puedes continuar ejecutando el script de nuevo.")
    else:
        logger.info(f"=== SCRAPING COMPLETADO ===")

    logger.info(f"Eventos procesados: {processed}")
    logger.info(f"Eventos con stats: {stats_total}")
    logger.info(f"Eventos con lineups: {lineups_total}")

    return stats_total, lineups_total


def signal_handler(signum, frame):
    """Handler para Ctrl+C - guarda progreso antes de salir."""
    global INTERRUPTED
    INTERRUPTED = True
    logger.warning("\n¡Interrupción detectada! Guardando progreso...")


# ============================================================================
# CLI
# ============================================================================

def print_stats(db: DatabaseManager, checkpoint: CheckpointManager):
    """Imprime estadísticas."""
    db_stats = db.get_db_stats()
    cp_stats = checkpoint.get_stats()

    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DE LA BASE DE DATOS")
    print("=" * 60)
    print(f"Total eventos:           {db_stats['total_events']:,}")
    print(f"Eventos terminados:      {db_stats['finished_events']:,}")
    print(f"Eventos con stats:       {db_stats['events_with_stats']:,}")
    print(f"Eventos con lineups:     {db_stats['events_with_lineups']:,}")
    print("-" * 60)
    print("LIGAS PRIORITARIAS")
    print("-" * 60)
    print(f"Eventos terminados:      {db_stats['prioritarios_finished']:,}")
    print(f"Con stats:               {db_stats['prioritarios_with_stats']:,}")
    print(f"Pendientes:              {db_stats['prioritarios_pending']:,}")
    print("-" * 60)
    print("CHECKPOINT (sesión actual)")
    print("-" * 60)
    print(f"Completados:             {cp_stats['completed']:,}")
    print(f"Fallidos:                {cp_stats['failed']:,}")
    print(f"Stats scrapeados:        {cp_stats['stats_scraped']:,}")
    print(f"Lineups scrapeados:      {cp_stats['lineups_scraped']:,}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='SofaScore Stats & Lineups Scraper (Playwright)')
    parser.add_argument('--all', action='store_true', help='Scrapear todos los eventos (no solo prioritarios)')
    parser.add_argument('--test', action='store_true', help='Modo test: solo 10 eventos')
    parser.add_argument('--stats', action='store_true', help='Mostrar estadísticas')
    parser.add_argument('--limit', type=int, help='Límite de eventos a procesar')
    parser.add_argument('--contexts', type=int, default=NUM_CONTEXTS, help='Número de contextos paralelos')
    parser.add_argument('--delay', type=float, default=REQUEST_DELAY, help='Delay entre requests (segundos)')

    args = parser.parse_args()

    # Verificar que existe la BD
    if not os.path.exists(DB_FILE):
        logger.error(f"Base de datos no encontrada: {DB_FILE}")
        logger.error("Copia sofascore.db al directorio 02_sofascore/")
        sys.exit(1)

    # Configurar handler para Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Inicializar
    db = DatabaseManager(DB_FILE)
    checkpoint = CheckpointManager(CHECKPOINT_FILE)

    try:
        if args.stats:
            print_stats(db, checkpoint)
            return

        # Determinar límite
        if args.test:
            limit = 10
        elif args.limit:
            limit = args.limit
        else:
            limit = None

        # Obtener eventos a procesar
        prioritarias_only = not args.all
        event_ids = db.get_events_without_stats(limit=limit, prioritarias_only=prioritarias_only)

        # Filtrar eventos ya completados en checkpoint
        event_ids = [eid for eid in event_ids if not checkpoint.is_completed(eid)]

        if not event_ids:
            logger.info("No hay eventos pendientes de procesar")
            print_stats(db, checkpoint)
            return

        logger.info(f"Eventos a procesar: {len(event_ids)}")
        logger.info(f"Configuración: {args.contexts} contextos, {args.delay}s delay")

        # Ejecutar scraping
        asyncio.run(scrape_with_playwright(
            event_ids=event_ids,
            db=db,
            checkpoint=checkpoint,
            num_contexts=args.contexts,
            request_delay=args.delay
        ))

        # Mostrar estadísticas finales
        print_stats(db, checkpoint)

    finally:
        db.close()


if __name__ == "__main__":
    main()
