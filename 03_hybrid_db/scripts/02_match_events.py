#!/usr/bin/env python3
"""
02_match_events.py
Hace matching entre fixtures de API-Football y eventos de SofaScore.

Algoritmo:
1. Para cada fixture de API-Football
2. Buscar en SofaScore por fecha (±1 día)
3. Filtrar por liga (fuzzy match)
4. Comparar equipos (fuzzy match con threshold 0.85)
5. Si match > threshold → vincular

Uso:
    python 02_match_events.py
    python 02_match_events.py --threshold 0.80  # Threshold más bajo
    python 02_match_events.py --limit 1000      # Solo primeros 1000
"""

import os
import sys
import sqlite3
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from collections import defaultdict

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
HYBRID_DB_PATH = os.path.join(DATA_DIR, 'hybrid.db')
SOFASCORE_DB_PATH = os.path.join(SCRIPT_DIR, '..', '..', '02_sofascore', 'data', 'sofascore.db')

# Configuración de matching
DEFAULT_THRESHOLD = 0.85
DATE_TOLERANCE_DAYS = 1


def normalize_team_name(name: str) -> str:
    """
    Normaliza el nombre de un equipo para comparación.

    Ejemplos:
        "Manchester United FC" -> "manchester united"
        "FC Barcelona" -> "barcelona"
        "Real Madrid CF" -> "real madrid"
    """
    if not name:
        return ""

    name = name.lower().strip()

    # Remover sufijos comunes
    suffixes = [' fc', ' cf', ' sc', ' ac', ' afc', ' sfc', ' bc', ' fk', ' sk',
                ' united', ' city', ' town', ' rovers', ' wanderers',
                ' athletic', ' albion', ' hotspur']

    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()

    # Remover prefijos comunes
    prefixes = ['fc ', 'cf ', 'sc ', 'ac ', 'afc ', 'real ', 'sporting ',
                'atletico ', 'atlético ']

    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):].strip()

    return name


def normalize_league_name(name: str) -> str:
    """Normaliza el nombre de una liga."""
    if not name:
        return ""

    name = name.lower().strip()

    # Mapeo de nombres comunes
    mappings = {
        'premier league': 'premier league',
        'la liga': 'la liga',
        'serie a': 'serie a',
        'bundesliga': 'bundesliga',
        'ligue 1': 'ligue 1',
        'primeira liga': 'primeira liga',
        'eredivisie': 'eredivisie',
        'champions league': 'champions league',
        'europa league': 'europa league',
        'conference league': 'conference league',
    }

    for key, value in mappings.items():
        if key in name:
            return value

    return name


def similarity_score(s1: str, s2: str) -> float:
    """Calcula la similitud entre dos strings (0-1)."""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1, s2).ratio()


def match_teams(home1: str, away1: str, home2: str, away2: str) -> Tuple[float, str]:
    """
    Compara dos pares de equipos y devuelve el score de matching.

    Returns:
        Tuple de (score, método) donde método es "exact" o "fuzzy"
    """
    # Normalizar nombres
    h1 = normalize_team_name(home1)
    a1 = normalize_team_name(away1)
    h2 = normalize_team_name(home2)
    a2 = normalize_team_name(away2)

    # Comparación exacta
    if h1 == h2 and a1 == a2:
        return 1.0, "exact"

    # Comparación fuzzy
    home_score = similarity_score(h1, h2)
    away_score = similarity_score(a1, a2)

    # Score combinado (promedio ponderado)
    combined_score = (home_score + away_score) / 2

    return combined_score, "fuzzy"


class EventMatcher:
    """Maneja el matching entre API-Football y SofaScore."""

    def __init__(self, hybrid_db_path: str, sofascore_db_path: str):
        self.hybrid_db_path = hybrid_db_path
        self.sofascore_db_path = sofascore_db_path
        self.hybrid_conn = None
        self.sofa_conn = None

    def connect(self):
        """Conecta a ambas bases de datos."""
        if not os.path.exists(self.hybrid_db_path):
            raise FileNotFoundError(f"No se encuentra {self.hybrid_db_path}. Ejecuta 01_export_api_football.py primero.")

        if not os.path.exists(self.sofascore_db_path):
            raise FileNotFoundError(f"No se encuentra {self.sofascore_db_path}. Copia la BD de SofaScore primero.")

        self.hybrid_conn = sqlite3.connect(self.hybrid_db_path)
        self.hybrid_conn.row_factory = sqlite3.Row

        self.sofa_conn = sqlite3.connect(self.sofascore_db_path)
        self.sofa_conn.row_factory = sqlite3.Row

        logger.info(f"Conectado a {self.hybrid_db_path}")
        logger.info(f"Conectado a {self.sofascore_db_path}")

    def close(self):
        """Cierra las conexiones."""
        if self.hybrid_conn:
            self.hybrid_conn.close()
        if self.sofa_conn:
            self.sofa_conn.close()

    def create_mapping_table(self):
        """Crea la tabla de mapping."""
        cursor = self.hybrid_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_mapping (
                fixture_id INTEGER PRIMARY KEY,
                event_id INTEGER,
                match_score REAL,
                match_method TEXT,
                home_team_api TEXT,
                away_team_api TEXT,
                home_team_sofa TEXT,
                away_team_sofa TEXT,
                date_api TEXT,
                date_sofa TEXT,
                league_api TEXT,
                league_sofa TEXT,
                created_at TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_mapping_event ON event_mapping(event_id)')
        self.hybrid_conn.commit()

    def get_api_fixtures(self, limit: int = None) -> List[Dict]:
        """Obtiene fixtures de API-Football."""
        cursor = self.hybrid_conn.cursor()

        sql = '''
            SELECT fixture_id, date, league_name, league_country,
                   home_team_name, away_team_name
            FROM api_fixtures
            WHERE goals_home IS NOT NULL
            ORDER BY date
        '''

        if limit:
            sql += f' LIMIT {limit}'

        cursor.execute(sql)
        return [dict(row) for row in cursor.fetchall()]

    def get_sofa_events_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Obtiene eventos de SofaScore en un rango de fechas."""
        cursor = self.sofa_conn.cursor()
        cursor.execute('''
            SELECT event_id, date, league_name, country,
                   home_team, away_team
            FROM events
            WHERE date BETWEEN ? AND ?
            AND status = 'finished'
        ''', (start_date, end_date))
        return [dict(row) for row in cursor.fetchall()]

    def build_sofa_index(self) -> Dict[str, List[Dict]]:
        """
        Construye un índice de eventos de SofaScore por fecha.
        Esto acelera mucho el matching.
        """
        logger.info("Construyendo índice de SofaScore por fecha...")
        cursor = self.sofa_conn.cursor()
        cursor.execute('''
            SELECT event_id, date, league_name, country,
                   home_team, away_team
            FROM events
            WHERE status = 'finished'
        ''')

        index = defaultdict(list)
        count = 0
        for row in cursor:
            event = dict(row)
            date = event['date']
            index[date].append(event)
            count += 1

        logger.info(f"Índice construido: {count:,} eventos en {len(index):,} fechas")
        return index

    def find_match(self, fixture: Dict, sofa_index: Dict[str, List[Dict]],
                   threshold: float) -> Optional[Tuple[int, float, str, Dict]]:
        """
        Busca el mejor match en SofaScore para un fixture de API-Football.

        Returns:
            Tuple de (event_id, score, método, evento_sofa) o None si no hay match
        """
        # Extraer fecha de API-Football (formato ISO)
        api_date_str = fixture['date'][:10]  # YYYY-MM-DD

        try:
            api_date = datetime.strptime(api_date_str, '%Y-%m-%d')
        except:
            return None

        # Buscar en rango de fechas
        best_match = None
        best_score = 0

        for delta in range(-DATE_TOLERANCE_DAYS, DATE_TOLERANCE_DAYS + 1):
            check_date = (api_date + timedelta(days=delta)).strftime('%Y-%m-%d')
            events = sofa_index.get(check_date, [])

            for event in events:
                # Comparar equipos
                score, method = match_teams(
                    fixture['home_team_name'], fixture['away_team_name'],
                    event['home_team'], event['away_team']
                )

                # Bonus por liga similar
                league_sim = similarity_score(
                    normalize_league_name(fixture['league_name']),
                    normalize_league_name(event['league_name'])
                )

                # Score final (80% equipos, 20% liga)
                final_score = score * 0.8 + league_sim * 0.2

                # Penalización si la fecha no es exacta
                if delta != 0:
                    final_score *= 0.95

                if final_score > best_score:
                    best_score = final_score
                    best_match = (event['event_id'], final_score, method, event)

        if best_match and best_match[1] >= threshold:
            return best_match

        return None

    def run_matching(self, threshold: float = DEFAULT_THRESHOLD, limit: int = None):
        """Ejecuta el proceso de matching completo."""

        # Crear tabla de mapping
        self.create_mapping_table()

        # Obtener fixtures de API-Football
        logger.info("Obteniendo fixtures de API-Football...")
        fixtures = self.get_api_fixtures(limit)
        logger.info(f"Total fixtures a procesar: {len(fixtures):,}")

        # Construir índice de SofaScore
        sofa_index = self.build_sofa_index()

        # Procesar cada fixture
        logger.info(f"\nIniciando matching (threshold={threshold})...")

        cursor = self.hybrid_conn.cursor()
        matched = 0
        not_matched = 0

        for i, fixture in enumerate(fixtures):
            # Buscar match
            result = self.find_match(fixture, sofa_index, threshold)

            if result:
                event_id, score, method, sofa_event = result

                cursor.execute('''
                    INSERT OR REPLACE INTO event_mapping
                    (fixture_id, event_id, match_score, match_method,
                     home_team_api, away_team_api, home_team_sofa, away_team_sofa,
                     date_api, date_sofa, league_api, league_sofa, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fixture['fixture_id'],
                    event_id,
                    score,
                    method,
                    fixture['home_team_name'],
                    fixture['away_team_name'],
                    sofa_event['home_team'],
                    sofa_event['away_team'],
                    fixture['date'][:10],
                    sofa_event['date'],
                    fixture['league_name'],
                    sofa_event['league_name'],
                    datetime.now().isoformat()
                ))
                matched += 1
            else:
                not_matched += 1

            # Progreso
            if (i + 1) % 5000 == 0:
                self.hybrid_conn.commit()
                pct = matched / (i + 1) * 100
                logger.info(f"Progreso: {i+1:,}/{len(fixtures):,} - Matched: {matched:,} ({pct:.1f}%)")

        self.hybrid_conn.commit()

        # Estadísticas finales
        logger.info("\n" + "=" * 60)
        logger.info("MATCHING COMPLETADO")
        logger.info("=" * 60)
        logger.info(f"Total fixtures: {len(fixtures):,}")
        logger.info(f"Matched: {matched:,} ({matched/len(fixtures)*100:.1f}%)")
        logger.info(f"Not matched: {not_matched:,} ({not_matched/len(fixtures)*100:.1f}%)")

        # Distribución por score
        cursor.execute('''
            SELECT
                CASE
                    WHEN match_score >= 0.95 THEN '0.95-1.00 (Excelente)'
                    WHEN match_score >= 0.90 THEN '0.90-0.95 (Muy bueno)'
                    WHEN match_score >= 0.85 THEN '0.85-0.90 (Bueno)'
                    ELSE '< 0.85 (Regular)'
                END as score_range,
                COUNT(*) as count
            FROM event_mapping
            GROUP BY score_range
            ORDER BY score_range DESC
        ''')

        logger.info("\nDistribución por score:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]:,}")

        # Distribución por método
        cursor.execute('''
            SELECT match_method, COUNT(*) as count
            FROM event_mapping
            GROUP BY match_method
        ''')

        logger.info("\nDistribución por método:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]:,}")


def main():
    parser = argparse.ArgumentParser(description='Matching API-Football ↔ SofaScore')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Threshold mínimo de matching (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--limit', type=int, default=None,
                        help='Límite de fixtures a procesar')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MATCHING API-FOOTBALL ↔ SOFASCORE")
    logger.info("=" * 60)

    matcher = EventMatcher(HYBRID_DB_PATH, SOFASCORE_DB_PATH)

    try:
        matcher.connect()
        matcher.run_matching(threshold=args.threshold, limit=args.limit)
    finally:
        matcher.close()


if __name__ == '__main__':
    main()
