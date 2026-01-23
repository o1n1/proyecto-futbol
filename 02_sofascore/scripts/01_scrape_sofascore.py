"""
SofaScore Full Database Scraper
Extrae TODA la información disponible de SofaScore: partidos, cuotas, estadísticas, alineaciones.

Uso:
    python scrape_sofascore_full.py --year 2026      # Scrapear año específico
    python scrape_sofascore_full.py --date 2024-06-15  # Scrapear día específico
    python scrape_sofascore_full.py                    # Continuar desde checkpoint
    python scrape_sofascore_full.py --details          # Scrapear stats/lineups
    python scrape_sofascore_full.py --stats            # Ver estadísticas de la BD
    python scrape_sofascore_full.py --export           # Exportar a CSV

Fuente: API REST de SofaScore (gratuita, sin autenticación)
"""

import os
import json
import sqlite3
import logging
import argparse
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
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

BASE_URL = "https://www.sofascore.com/api/v1"

# Rutas relativas al directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = DATA_DIR  # Para compatibilidad
RAW_DIR = os.path.join(DATA_DIR, "raw")
DETAILS_DIR = os.path.join(DATA_DIR, "details")
DB_FILE = os.path.join(DATA_DIR, "sofascore.db")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint.json")

# Paralelismo (configuración conservadora para evitar bloqueos)
MAX_WORKERS = 5          # Reducido de 20 para evitar rate limiting
MAX_WORKERS_DETAILS = 3  # Aún más conservador para stats/lineups
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
REQUEST_DELAY = 0.5      # Delay entre requests en segundos

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.sofascore.com/',
}


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
            'phase1_completed_dates': [],
            'phase1_last_date': None,
            'phase2_completed_events': [],
            'total_events_scraped': 0,
            'total_odds_scraped': 0,
            'failed_dates': [],
            'started_at': datetime.now().isoformat()
        }

    def save(self):
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, default=str)

    def mark_date_completed(self, date: str, events_count: int, odds_count: int):
        if date not in self.data['phase1_completed_dates']:
            self.data['phase1_completed_dates'].append(date)
        self.data['phase1_last_date'] = date
        self.data['total_events_scraped'] += events_count
        self.data['total_odds_scraped'] += odds_count
        self.save()

    def mark_date_failed(self, date: str, error: str):
        self.data['failed_dates'].append({
            'date': date,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def is_date_completed(self, date: str) -> bool:
        return date in self.data['phase1_completed_dates']

    def get_stats(self) -> dict:
        return {
            'dates_completed': len(self.data['phase1_completed_dates']),
            'last_date': self.data['phase1_last_date'],
            'total_events': self.data['total_events_scraped'],
            'total_odds': self.data['total_odds_scraped'],
            'failed_dates': len(self.data['failed_dates'])
        }


# ============================================================================
# SOFASCORE CLIENT
# ============================================================================

class SofaScoreClient:
    """Cliente para la API de SofaScore con retry."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _make_request(self, url: str, retries: int = MAX_RETRIES) -> Optional[dict]:
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT, verify=False)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return None  # No data available
                elif response.status_code == 429:
                    logger.warning(f"Rate limited, waiting 60s...")
                    time.sleep(60)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout (attempt {attempt+1}/{retries}): {url}")
            except Exception as e:
                logger.error(f"Error (attempt {attempt+1}/{retries}): {e}")

            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def get_scheduled_events(self, date: str) -> Optional[dict]:
        """Obtiene todos los partidos de un día."""
        url = f"{BASE_URL}/sport/football/scheduled-events/{date}"
        return self._make_request(url)

    def get_odds_for_date(self, date: str) -> Optional[dict]:
        """Obtiene todas las cuotas de un día (bulk)."""
        url = f"{BASE_URL}/sport/football/odds/1/{date}"
        return self._make_request(url)

    def get_event_statistics(self, event_id: int) -> Optional[dict]:
        """Obtiene estadísticas de un partido."""
        url = f"{BASE_URL}/event/{event_id}/statistics"
        return self._make_request(url)

    def get_event_lineups(self, event_id: int) -> Optional[dict]:
        """Obtiene alineaciones de un partido."""
        url = f"{BASE_URL}/event/{event_id}/lineups"
        return self._make_request(url)


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Maneja la base de datos SQLite."""

    def __init__(self, db_file: str):
        self.db_file = db_file
        self.conn = None
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        # Tabla de eventos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
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
            )
        """)

        # Tabla de cuotas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                market_name TEXT,
                market_id INTEGER,
                choice_name TEXT,
                odds_initial REAL,
                odds_final REAL,
                winning INTEGER,
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
        """)

        # Tabla de estadísticas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                period TEXT,
                stat_name TEXT,
                stat_key TEXT,
                home_value REAL,
                away_value REAL,
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
        """)

        # Tabla de alineaciones
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lineups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                team_type TEXT,
                team_id INTEGER,
                player_id INTEGER,
                player_name TEXT,
                position TEXT,
                jersey_number INTEGER,
                is_starter INTEGER,
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
        """)

        # Índices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_league ON events(league_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_country ON events(country)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_odds_event ON odds(event_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_event ON statistics(event_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineups_event ON lineups(event_id)")

        self.conn.commit()

    def insert_event(self, event: dict) -> bool:
        """Inserta o actualiza un evento."""
        try:
            cursor = self.conn.cursor()

            # Extraer datos del evento
            tournament = event.get('tournament', {})
            category = tournament.get('category', {})
            home_team = event.get('homeTeam', {})
            away_team = event.get('awayTeam', {})
            home_score = event.get('homeScore', {})
            away_score = event.get('awayScore', {})
            status = event.get('status', {})
            season = event.get('season', {})
            round_info = event.get('roundInfo', {})

            timestamp = event.get('startTimestamp', 0)
            dt = datetime.fromtimestamp(timestamp) if timestamp else None

            cursor.execute("""
                INSERT OR REPLACE INTO events
                (event_id, date, time, timestamp, country, league_name, league_id,
                 season, round, home_team, home_team_id, away_team, away_team_id,
                 home_score, away_score, home_score_ht, away_score_ht,
                 status, winner_code, slug, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.get('id'),
                dt.strftime('%Y-%m-%d') if dt else None,
                dt.strftime('%H:%M') if dt else None,
                timestamp,
                category.get('name'),
                tournament.get('name'),
                tournament.get('uniqueTournament', {}).get('id') or tournament.get('id'),
                season.get('name'),
                round_info.get('round'),
                home_team.get('name'),
                home_team.get('id'),
                away_team.get('name'),
                away_team.get('id'),
                home_score.get('current'),
                away_score.get('current'),
                home_score.get('period1'),
                away_score.get('period1'),
                status.get('type'),
                event.get('winnerCode'),
                event.get('slug'),
                datetime.now().isoformat()
            ))

            return True
        except Exception as e:
            logger.error(f"Error inserting event {event.get('id')}: {e}")
            return False

    def insert_odds(self, event_id: int, odds_data: dict) -> int:
        """Inserta cuotas de un evento."""
        count = 0
        try:
            cursor = self.conn.cursor()

            # Eliminar cuotas anteriores
            cursor.execute("DELETE FROM odds WHERE event_id = ?", (event_id,))

            # La estructura puede ser directa o dentro de 'markets'
            markets = odds_data.get('markets', [odds_data]) if 'markets' in odds_data else [odds_data]

            for market in markets:
                market_name = market.get('marketName', 'Unknown')
                market_id = market.get('marketId')
                choices = market.get('choices', [])

                for choice in choices:
                    # Convertir cuotas fraccionarias a decimales
                    initial_frac = choice.get('initialFractionalValue', '')
                    final_frac = choice.get('fractionalValue', '')

                    initial_dec = self._frac_to_decimal(initial_frac)
                    final_dec = self._frac_to_decimal(final_frac)

                    cursor.execute("""
                        INSERT INTO odds (event_id, market_name, market_id, choice_name,
                                         odds_initial, odds_final, winning)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event_id,
                        market_name,
                        market_id,
                        choice.get('name'),
                        initial_dec,
                        final_dec,
                        1 if choice.get('winning') else 0
                    ))
                    count += 1

            return count
        except Exception as e:
            logger.error(f"Error inserting odds for event {event_id}: {e}")
            return 0

    def _frac_to_decimal(self, frac: str) -> Optional[float]:
        """Convierte cuota fraccionaria a decimal."""
        if not frac or '/' not in frac:
            return None
        try:
            num, den = frac.split('/')
            return round(1 + float(num) / float(den), 3)
        except:
            return None

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

    def get_stats(self) -> dict:
        """Obtiene estadísticas de la BD."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM events")
        total_events = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT event_id) FROM odds")
        events_with_odds = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT event_id) FROM statistics")
        events_with_stats = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT event_id) FROM lineups")
        events_with_lineups = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(date), MAX(date) FROM events")
        date_range = cursor.fetchone()

        cursor.execute("SELECT COUNT(DISTINCT country) FROM events")
        countries = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT league_name) FROM events")
        leagues = cursor.fetchone()[0]

        return {
            'total_events': total_events,
            'events_with_odds': events_with_odds,
            'events_with_stats': events_with_stats,
            'events_with_lineups': events_with_lineups,
            'date_min': date_range[0],
            'date_max': date_range[1],
            'countries': countries,
            'leagues': leagues
        }

    def close(self):
        if self.conn:
            self.conn.close()


# ============================================================================
# SOFASCORE SCRAPER
# ============================================================================

class SofaScoreScraper:
    """Orquestador principal del scraping."""

    def __init__(self):
        # Crear directorios
        os.makedirs(RAW_DIR, exist_ok=True)
        os.makedirs(DETAILS_DIR, exist_ok=True)

        self.client = SofaScoreClient()
        self.db = DatabaseManager(DB_FILE)
        self.checkpoint = CheckpointManager(CHECKPOINT_FILE)

    def fetch_date_data(self, date: str) -> Tuple[str, List[dict], dict, Optional[str]]:
        """
        Obtiene datos de un día (solo HTTP, sin BD).
        Returns: (date, events_list, odds_dict, error)
        """
        try:
            # Crear directorio para el año
            year = date[:4]
            year_dir = os.path.join(RAW_DIR, year)
            os.makedirs(year_dir, exist_ok=True)

            # Obtener eventos
            events_data = self.client.get_scheduled_events(date)
            if not events_data:
                return date, [], {}, None

            events = events_data.get('events', [])

            # Guardar JSON crudo
            events_file = os.path.join(year_dir, f"{date}_events.json")
            with open(events_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'date': date,
                    'scraped_at': datetime.now().isoformat(),
                    'events_count': len(events),
                    'events': events
                }, f, ensure_ascii=False)

            # Obtener cuotas
            odds_dict = {}
            odds_data = self.client.get_odds_for_date(date)
            if odds_data:
                # Guardar JSON crudo
                odds_file = os.path.join(year_dir, f"{date}_odds.json")
                with open(odds_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'date': date,
                        'scraped_at': datetime.now().isoformat(),
                        'odds': odds_data
                    }, f, ensure_ascii=False)
                odds_dict = odds_data.get('odds', odds_data)

            return date, events, odds_dict, None

        except Exception as e:
            return date, [], {}, str(e)

    def scrape_date(self, date: str) -> Tuple[int, int]:
        """
        Scrapea todos los eventos y cuotas de un día (secuencial).
        Returns: (events_count, odds_count)
        """
        date, events, odds_dict, error = self.fetch_date_data(date)
        if error:
            raise Exception(error)

        events_count = 0
        odds_count = 0

        # Insertar eventos en BD
        for event in events:
            if self.db.insert_event(event):
                events_count += 1

        # Insertar cuotas en BD
        for event_id_str, event_odds in odds_dict.items():
            try:
                event_id = int(event_id_str)
                count = self.db.insert_odds(event_id, event_odds)
                odds_count += count
            except (ValueError, TypeError):
                continue

        self.db.commit()
        return events_count, odds_count

    def scrape_year(self, year: int, parallel: bool = True):
        """Scrapea un año completo."""
        logger.info(f"=== SCRAPING AÑO {year} ===")

        # Generar lista de fechas
        start_date = datetime(year, 1, 1)
        end_date = min(datetime(year, 12, 31), datetime.now())

        dates = []
        current = start_date
        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            if not self.checkpoint.is_date_completed(date_str):
                dates.append(date_str)
            current += timedelta(days=1)

        if not dates:
            logger.info(f"Año {year} ya completado")
            return

        logger.info(f"Fechas a procesar: {len(dates)}")

        total_events = 0
        total_odds = 0

        if parallel:
            # Procesamiento paralelo: HTTP en threads, BD en thread principal
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(self.fetch_date_data, date): date for date in dates}

                for i, future in enumerate(as_completed(futures)):
                    date, events, odds_dict, error = future.result()

                    if error:
                        logger.error(f"Error {date}: {error}")
                        self.checkpoint.mark_date_failed(date, error)
                    else:
                        # Insertar en BD desde el thread principal
                        events_count = 0
                        odds_count = 0

                        for event in events:
                            if self.db.insert_event(event):
                                events_count += 1

                        for event_id_str, event_odds in odds_dict.items():
                            try:
                                event_id = int(event_id_str)
                                count = self.db.insert_odds(event_id, event_odds)
                                odds_count += count
                            except (ValueError, TypeError):
                                continue

                        self.db.commit()
                        self.checkpoint.mark_date_completed(date, events_count, odds_count)
                        total_events += events_count
                        total_odds += odds_count

                        if (i + 1) % 50 == 0:
                            logger.info(f"Progreso: {i+1}/{len(dates)} fechas, {total_events} eventos, {total_odds} cuotas")
        else:
            # Procesamiento secuencial
            for i, date in enumerate(dates):
                events, odds = self.scrape_date(date)
                self.checkpoint.mark_date_completed(date, events, odds)
                total_events += events
                total_odds += odds

                if (i + 1) % 10 == 0:
                    logger.info(f"Progreso: {i+1}/{len(dates)} fechas")

        logger.info(f"=== AÑO {year} COMPLETADO ===")
        logger.info(f"Total eventos: {total_events}")
        logger.info(f"Total cuotas: {total_odds}")

    def scrape_range(self, start_year: int, end_year: int):
        """Scrapea un rango de años (de más reciente a más antiguo)."""
        for year in range(end_year, start_year - 1, -1):
            self.scrape_year(year)

    def scrape_details_for_events(self, event_ids: List[int]):
        """Scrapea estadísticas y alineaciones para eventos específicos."""
        logger.info(f"Scrapeando detalles de {len(event_ids)} eventos...")
        logger.info(f"Configuracion: {MAX_WORKERS_DETAILS} workers, {REQUEST_DELAY}s delay")

        stats_count = 0
        lineups_count = 0
        errors_count = 0

        def fetch_details(event_id: int) -> Tuple[int, Optional[dict], Optional[dict], bool]:
            """Obtiene datos (solo HTTP, sin BD)."""
            stats = None
            lineups = None
            has_error = False

            # Delay entre requests para evitar rate limiting
            time.sleep(REQUEST_DELAY)

            # Estadísticas
            stats_data = self.client.get_event_statistics(event_id)
            if stats_data:
                # Guardar JSON
                stats_file = os.path.join(DETAILS_DIR, f"{event_id}_stats.json")
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats_data, f, ensure_ascii=False)
                stats = stats_data

            # Delay adicional entre stats y lineups
            time.sleep(REQUEST_DELAY)

            # Alineaciones
            lineups_data = self.client.get_event_lineups(event_id)
            if lineups_data:
                # Guardar JSON
                lineups_file = os.path.join(DETAILS_DIR, f"{event_id}_lineups.json")
                with open(lineups_file, 'w', encoding='utf-8') as f:
                    json.dump(lineups_data, f, ensure_ascii=False)
                lineups = lineups_data

            # Detectar si hubo error 403 (bloqueado)
            if stats is None and lineups is None:
                has_error = True

            return event_id, stats, lineups, has_error

        # Usar menos workers para detalles (más conservador)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DETAILS) as executor:
            futures = {executor.submit(fetch_details, eid): eid for eid in event_ids}

            for i, future in enumerate(as_completed(futures)):
                event_id, stats, lineups, has_error = future.result()

                if has_error:
                    errors_count += 1
                    # Si hay muchos errores seguidos, probablemente estamos bloqueados
                    if errors_count >= 10:
                        logger.warning(f"Muchos errores (403), posible bloqueo. Pausando 60s...")
                        time.sleep(60)
                        errors_count = 0
                else:
                    errors_count = 0  # Reset contador de errores

                # Insertar en BD desde el thread principal
                if stats:
                    stats_count += self.db.insert_statistics(event_id, stats)
                if lineups:
                    lineups_count += self.db.insert_lineups(event_id, lineups)

                if (i + 1) % 50 == 0:
                    logger.info(f"Progreso detalles: {i+1}/{len(event_ids)} ({stats_count} stats, {lineups_count} lineups)")
                    self.db.commit()

        self.db.commit()
        logger.info(f"Detalles completados: {stats_count} stats, {lineups_count} lineups")

    def print_stats(self):
        """Imprime estadísticas de la BD."""
        db_stats = self.db.get_stats()
        cp_stats = self.checkpoint.get_stats()

        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DE LA BASE DE DATOS")
        print("=" * 60)
        print(f"Total eventos:        {db_stats['total_events']:,}")
        print(f"Eventos con cuotas:   {db_stats['events_with_odds']:,}")
        print(f"Eventos con stats:    {db_stats['events_with_stats']:,}")
        print(f"Eventos con lineups:  {db_stats['events_with_lineups']:,}")
        print(f"Rango de fechas:      {db_stats['date_min']} -> {db_stats['date_max']}")
        print(f"Países:               {db_stats['countries']}")
        print(f"Ligas:                {db_stats['leagues']}")
        print("-" * 60)
        print("CHECKPOINT")
        print("-" * 60)
        print(f"Fechas completadas:   {cp_stats['dates_completed']}")
        print(f"Última fecha:         {cp_stats['last_date']}")
        print(f"Fechas fallidas:      {cp_stats['failed_dates']}")
        print("=" * 60)

    def export_to_csv(self, output_dir: str = None):
        """Exporta la BD a archivos CSV."""
        if output_dir is None:
            output_dir = os.path.join(OUTPUT_DIR, "csv_export")
        os.makedirs(output_dir, exist_ok=True)

        import pandas as pd

        # Exportar eventos
        events_df = pd.read_sql("SELECT * FROM events", self.db.conn)
        events_file = os.path.join(output_dir, "events.csv")
        events_df.to_csv(events_file, index=False)
        logger.info(f"Exportado: {events_file} ({len(events_df)} filas)")

        # Exportar cuotas
        odds_df = pd.read_sql("SELECT * FROM odds", self.db.conn)
        odds_file = os.path.join(output_dir, "odds.csv")
        odds_df.to_csv(odds_file, index=False)
        logger.info(f"Exportado: {odds_file} ({len(odds_df)} filas)")

        # Exportar estadísticas
        stats_df = pd.read_sql("SELECT * FROM statistics", self.db.conn)
        stats_file = os.path.join(output_dir, "statistics.csv")
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"Exportado: {stats_file} ({len(stats_df)} filas)")

        # Exportar alineaciones
        lineups_df = pd.read_sql("SELECT * FROM lineups", self.db.conn)
        lineups_file = os.path.join(output_dir, "lineups.csv")
        lineups_df.to_csv(lineups_file, index=False)
        logger.info(f"Exportado: {lineups_file} ({len(lineups_df)} filas)")

    def close(self):
        self.db.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SofaScore Full Database Scraper')
    parser.add_argument('--year', type=int, help='Scrapear año específico')
    parser.add_argument('--date', type=str, help='Scrapear día específico (YYYY-MM-DD)')
    parser.add_argument('--range', type=str, help='Rango de años (ej: 2019-2026)')
    parser.add_argument('--details', action='store_true', help='Scrapear stats/lineups')
    parser.add_argument('--stats', action='store_true', help='Mostrar estadísticas')
    parser.add_argument('--export', action='store_true', help='Exportar a CSV')
    parser.add_argument('--sequential', action='store_true', help='Modo secuencial (sin paralelismo)')

    args = parser.parse_args()

    scraper = SofaScoreScraper()

    try:
        if args.stats:
            scraper.print_stats()

        elif args.export:
            scraper.export_to_csv()

        elif args.date:
            logger.info(f"Scrapeando fecha: {args.date}")
            events, odds = scraper.scrape_date(args.date)
            logger.info(f"Completado: {events} eventos, {odds} cuotas")
            scraper.checkpoint.mark_date_completed(args.date, events, odds)

        elif args.year:
            scraper.scrape_year(args.year, parallel=not args.sequential)

        elif args.range:
            start, end = map(int, args.range.split('-'))
            scraper.scrape_range(start, end)

        elif args.details:
            # Obtener eventos sin detalles
            cursor = scraper.db.conn.cursor()
            cursor.execute("""
                SELECT e.event_id FROM events e
                LEFT JOIN statistics s ON e.event_id = s.event_id
                WHERE s.event_id IS NULL AND e.status = 'finished'
                LIMIT 10000
            """)
            event_ids = [row[0] for row in cursor.fetchall()]

            if event_ids:
                scraper.scrape_details_for_events(event_ids)
            else:
                logger.info("No hay eventos pendientes de detalles")

        else:
            # Por defecto: continuar desde 2026 hacia atrás
            logger.info("Iniciando scraping completo (2026 -> 2019)")
            scraper.scrape_range(2019, 2026)

    finally:
        scraper.close()


if __name__ == "__main__":
    main()
