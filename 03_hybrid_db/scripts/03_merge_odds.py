#!/usr/bin/env python3
"""
03_merge_odds.py
Une las cuotas de SofaScore a los fixtures de API-Football usando el mapping.

Cuotas a extraer:
- 1X2 (Full time): odds_home, odds_draw, odds_away
- Over/Under 2.5: odds_over25, odds_under25
- BTTS: odds_btts_yes, odds_btts_no

Casa de apuestas: Bet365 (configurable)

Uso:
    python 03_merge_odds.py
    python 03_merge_odds.py --bookmaker "1xBet"  # Otra casa
    python 03_merge_odds.py --analyze             # Solo analizar cobertura
"""

import os
import sys
import sqlite3
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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

# Casa de apuestas por defecto
DEFAULT_BOOKMAKER = "Bet365"


class OddsMerger:
    """Maneja el merge de cuotas de SofaScore."""

    def __init__(self, hybrid_db_path: str, sofascore_db_path: str):
        self.hybrid_db_path = hybrid_db_path
        self.sofascore_db_path = sofascore_db_path
        self.hybrid_conn = None
        self.sofa_conn = None

    def connect(self):
        """Conecta a ambas bases de datos."""
        if not os.path.exists(self.hybrid_db_path):
            raise FileNotFoundError(f"No se encuentra {self.hybrid_db_path}")

        if not os.path.exists(self.sofascore_db_path):
            raise FileNotFoundError(f"No se encuentra {self.sofascore_db_path}")

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

    def analyze_bookmakers(self) -> Dict[str, int]:
        """Analiza qué casas de apuestas tienen más cobertura."""
        logger.info("Analizando cobertura de casas de apuestas...")

        # Primero verificar la estructura de la tabla odds
        cursor = self.sofa_conn.cursor()
        cursor.execute("PRAGMA table_info(odds)")
        columns = [col[1] for col in cursor.fetchall()]
        logger.info(f"Columnas en tabla odds: {columns}")

        # La tabla odds de SofaScore tiene estructura diferente
        # market_name, choice_name, odds_initial, odds_final
        # No tiene bookmaker directamente

        # Verificar mercados disponibles
        cursor.execute('''
            SELECT market_name, COUNT(*) as count
            FROM odds
            GROUP BY market_name
            ORDER BY count DESC
            LIMIT 20
        ''')

        logger.info("\nMercados disponibles:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]:,}")

        # Verificar choices
        cursor.execute('''
            SELECT market_name, choice_name, COUNT(*) as count
            FROM odds
            WHERE market_name = 'Full time'
            GROUP BY market_name, choice_name
            ORDER BY count DESC
        ''')

        logger.info("\nChoices para Full time:")
        for row in cursor.fetchall():
            logger.info(f"  {row[1]}: {row[2]:,}")

        return {}

    def create_odds_table(self):
        """Crea la tabla de cuotas híbridas."""
        cursor = self.hybrid_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hybrid_odds (
                fixture_id INTEGER PRIMARY KEY,
                event_id INTEGER,
                -- 1X2 Full time
                odds_home_open REAL,
                odds_home_close REAL,
                odds_draw_open REAL,
                odds_draw_close REAL,
                odds_away_open REAL,
                odds_away_close REAL,
                -- Over/Under 2.5
                odds_over25_open REAL,
                odds_over25_close REAL,
                odds_under25_open REAL,
                odds_under25_close REAL,
                -- BTTS
                odds_btts_yes_open REAL,
                odds_btts_yes_close REAL,
                odds_btts_no_open REAL,
                odds_btts_no_close REAL,
                -- Metadata
                created_at TEXT,
                FOREIGN KEY (fixture_id) REFERENCES api_fixtures(fixture_id)
            )
        ''')
        self.hybrid_conn.commit()

    def get_mapped_events(self) -> List[Tuple[int, int]]:
        """Obtiene todos los pares (fixture_id, event_id) del mapping."""
        cursor = self.hybrid_conn.cursor()
        cursor.execute('SELECT fixture_id, event_id FROM event_mapping')
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def get_odds_for_event(self, event_id: int) -> Dict:
        """
        Obtiene todas las cuotas de un evento.

        Returns:
            Dict con las cuotas extraídas
        """
        cursor = self.sofa_conn.cursor()

        odds = {
            'odds_home_open': None, 'odds_home_close': None,
            'odds_draw_open': None, 'odds_draw_close': None,
            'odds_away_open': None, 'odds_away_close': None,
            'odds_over25_open': None, 'odds_over25_close': None,
            'odds_under25_open': None, 'odds_under25_close': None,
            'odds_btts_yes_open': None, 'odds_btts_yes_close': None,
            'odds_btts_no_open': None, 'odds_btts_no_close': None,
        }

        # Obtener cuotas 1X2 (Full time)
        cursor.execute('''
            SELECT choice_name, odds_initial, odds_final
            FROM odds
            WHERE event_id = ? AND market_name = 'Full time'
        ''', (event_id,))

        for row in cursor.fetchall():
            choice = row[0]
            if choice == '1':
                odds['odds_home_open'] = row[1]
                odds['odds_home_close'] = row[2]
            elif choice == 'X':
                odds['odds_draw_open'] = row[1]
                odds['odds_draw_close'] = row[2]
            elif choice == '2':
                odds['odds_away_open'] = row[1]
                odds['odds_away_close'] = row[2]

        # Obtener cuotas Over/Under 2.5
        cursor.execute('''
            SELECT choice_name, odds_initial, odds_final
            FROM odds
            WHERE event_id = ? AND market_name LIKE '%Over%2.5%'
        ''', (event_id,))

        for row in cursor.fetchall():
            choice = row[0].lower()
            if 'over' in choice:
                odds['odds_over25_open'] = row[1]
                odds['odds_over25_close'] = row[2]
            elif 'under' in choice:
                odds['odds_under25_open'] = row[1]
                odds['odds_under25_close'] = row[2]

        # Obtener cuotas BTTS
        cursor.execute('''
            SELECT choice_name, odds_initial, odds_final
            FROM odds
            WHERE event_id = ? AND market_name LIKE '%Both%'
        ''', (event_id,))

        for row in cursor.fetchall():
            choice = row[0].lower()
            if 'yes' in choice:
                odds['odds_btts_yes_open'] = row[1]
                odds['odds_btts_yes_close'] = row[2]
            elif 'no' in choice:
                odds['odds_btts_no_open'] = row[1]
                odds['odds_btts_no_close'] = row[2]

        return odds

    def run_merge(self):
        """Ejecuta el proceso de merge de cuotas."""

        # Crear tabla
        self.create_odds_table()

        # Obtener mappings
        logger.info("Obteniendo mappings...")
        mappings = self.get_mapped_events()
        logger.info(f"Total mappings: {len(mappings):,}")

        if not mappings:
            logger.warning("No hay mappings. Ejecuta 02_match_events.py primero.")
            return

        # Procesar cada mapping
        logger.info("Extrayendo cuotas de SofaScore...")

        cursor = self.hybrid_conn.cursor()
        with_odds = 0
        without_odds = 0

        for i, (fixture_id, event_id) in enumerate(mappings):
            odds = self.get_odds_for_event(event_id)

            # Verificar si tiene al menos cuotas 1X2
            has_odds = odds['odds_home_close'] is not None

            cursor.execute('''
                INSERT OR REPLACE INTO hybrid_odds
                (fixture_id, event_id,
                 odds_home_open, odds_home_close,
                 odds_draw_open, odds_draw_close,
                 odds_away_open, odds_away_close,
                 odds_over25_open, odds_over25_close,
                 odds_under25_open, odds_under25_close,
                 odds_btts_yes_open, odds_btts_yes_close,
                 odds_btts_no_open, odds_btts_no_close,
                 created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fixture_id, event_id,
                odds['odds_home_open'], odds['odds_home_close'],
                odds['odds_draw_open'], odds['odds_draw_close'],
                odds['odds_away_open'], odds['odds_away_close'],
                odds['odds_over25_open'], odds['odds_over25_close'],
                odds['odds_under25_open'], odds['odds_under25_close'],
                odds['odds_btts_yes_open'], odds['odds_btts_yes_close'],
                odds['odds_btts_no_open'], odds['odds_btts_no_close'],
                datetime.now().isoformat()
            ))

            if has_odds:
                with_odds += 1
            else:
                without_odds += 1

            # Progreso
            if (i + 1) % 10000 == 0:
                self.hybrid_conn.commit()
                logger.info(f"Progreso: {i+1:,}/{len(mappings):,}")

        self.hybrid_conn.commit()

        # Estadísticas finales
        logger.info("\n" + "=" * 60)
        logger.info("MERGE DE CUOTAS COMPLETADO")
        logger.info("=" * 60)
        logger.info(f"Total mappings procesados: {len(mappings):,}")
        logger.info(f"Con cuotas 1X2: {with_odds:,} ({with_odds/len(mappings)*100:.1f}%)")
        logger.info(f"Sin cuotas: {without_odds:,} ({without_odds/len(mappings)*100:.1f}%)")

        # Cobertura por mercado
        cursor.execute('''
            SELECT
                SUM(CASE WHEN odds_home_close IS NOT NULL THEN 1 ELSE 0 END) as has_1x2,
                SUM(CASE WHEN odds_over25_close IS NOT NULL THEN 1 ELSE 0 END) as has_over25,
                SUM(CASE WHEN odds_btts_yes_close IS NOT NULL THEN 1 ELSE 0 END) as has_btts,
                COUNT(*) as total
            FROM hybrid_odds
        ''')

        row = cursor.fetchone()
        if row:
            logger.info(f"\nCobertura por mercado:")
            logger.info(f"  1X2: {row[0]:,} ({row[0]/row[3]*100:.1f}%)")
            logger.info(f"  Over/Under 2.5: {row[1]:,} ({row[1]/row[3]*100:.1f}%)")
            logger.info(f"  BTTS: {row[2]:,} ({row[2]/row[3]*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Merge cuotas de SofaScore')
    parser.add_argument('--analyze', action='store_true',
                        help='Solo analizar cobertura de casas de apuestas')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MERGE DE CUOTAS SOFASCORE → HYBRID DB")
    logger.info("=" * 60)

    merger = OddsMerger(HYBRID_DB_PATH, SOFASCORE_DB_PATH)

    try:
        merger.connect()

        if args.analyze:
            merger.analyze_bookmakers()
        else:
            merger.run_merge()
    finally:
        merger.close()


if __name__ == '__main__':
    main()
