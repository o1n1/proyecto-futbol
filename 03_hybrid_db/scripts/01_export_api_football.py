#!/usr/bin/env python3
"""
01_export_api_football.py
Exporta datos de API-Football desde Supabase a SQLite local.

Tablas exportadas:
- fixtures (112K filas, 40 columnas)
- fixture_features (112K filas, 152 columnas)

Uso:
    python 01_export_api_football.py
    python 01_export_api_football.py --limit 1000  # Solo primeros 1000 para pruebas
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
import urllib3
urllib3.disable_warnings()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de Supabase
SUPABASE_URL = "https://ykqaplnfrhvjqkvejudg.supabase.co"
SUPABASE_KEY = os.environ.get('SUPABASE_KEY',
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlrcWFwbG5mcmh2anFrdmVqdWRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg2NjY4NjgsImV4cCI6MjA4NDI0Mjg2OH0.abeJY6QxUn4gT5GYJmoD2xJ7uPVNEwAVAxJ0wE5bMvM"
)

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
DB_PATH = os.path.join(DATA_DIR, 'hybrid.db')

# Configuración de paginación
PAGE_SIZE = 1000  # Límite de Supabase por request


class SupabaseClient:
    """Cliente para conectar con Supabase REST API."""

    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.session = requests.Session()
        self.session.headers.update({
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        })

    def fetch_table(self, table: str, columns: str = '*',
                    order: str = None, limit: int = None) -> List[Dict]:
        """
        Obtiene todos los registros de una tabla con paginación.

        Args:
            table: Nombre de la tabla
            columns: Columnas a seleccionar (default: todas)
            order: Columna para ordenar
            limit: Límite total de registros (None = todos)

        Returns:
            Lista de diccionarios con los registros
        """
        all_rows = []
        offset = 0

        while True:
            url = f"{self.url}/rest/v1/{table}"
            params = {
                'select': columns,
                'offset': offset,
                'limit': PAGE_SIZE
            }

            if order:
                params['order'] = order

            try:
                response = self.session.get(url, params=params, timeout=60, verify=False)
                response.raise_for_status()
                rows = response.json()

                if not rows:
                    break

                all_rows.extend(rows)
                logger.info(f"  Obtenidos {len(all_rows)} registros de {table}...")

                # Verificar límite
                if limit and len(all_rows) >= limit:
                    all_rows = all_rows[:limit]
                    break

                # Si obtuvimos menos de PAGE_SIZE, es la última página
                if len(rows) < PAGE_SIZE:
                    break

                offset += PAGE_SIZE

            except requests.exceptions.RequestException as e:
                logger.error(f"Error al obtener datos de {table}: {e}")
                raise

        return all_rows


class SQLiteDatabase:
    """Maneja la base de datos SQLite híbrida."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """Conecta a la base de datos."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        logger.info(f"Conectado a {self.db_path}")

    def close(self):
        """Cierra la conexión."""
        if self.conn:
            self.conn.close()
            logger.info("Conexión cerrada")

    def create_tables(self):
        """Crea las tablas necesarias."""

        # Tabla api_fixtures
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_fixtures (
                fixture_id INTEGER PRIMARY KEY,
                date TEXT,
                timestamp INTEGER,
                timezone TEXT,
                venue_id INTEGER,
                venue_name TEXT,
                venue_city TEXT,
                status_long TEXT,
                status_short TEXT,
                status_elapsed INTEGER,
                league_id INTEGER,
                league_name TEXT,
                league_country TEXT,
                league_season INTEGER,
                league_round TEXT,
                home_team_id INTEGER,
                home_team_name TEXT,
                away_team_id INTEGER,
                away_team_name TEXT,
                goals_home INTEGER,
                goals_away INTEGER,
                score_halftime_home INTEGER,
                score_halftime_away INTEGER,
                score_fulltime_home INTEGER,
                score_fulltime_away INTEGER,
                created_at TEXT,
                updated_at TEXT,
                stats_fetched INTEGER,
                odds_fetched INTEGER,
                home_shots_on_goal INTEGER,
                home_total_shots INTEGER,
                home_ball_possession REAL,
                away_shots_on_goal INTEGER,
                away_total_shots INTEGER,
                away_ball_possession REAL,
                odds_home REAL,
                odds_draw REAL,
                odds_away REAL,
                match_type TEXT
            )
        ''')

        # Índices para api_fixtures
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_fixtures_date ON api_fixtures(date)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_fixtures_league ON api_fixtures(league_name, league_country)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_fixtures_teams ON api_fixtures(home_team_name, away_team_name)')

        # Tabla api_features (se crea dinámicamente según las columnas)
        # La crearemos cuando insertemos los datos

        self.conn.commit()
        logger.info("Tablas creadas")

    def insert_fixtures(self, fixtures: List[Dict]):
        """Inserta fixtures en la tabla api_fixtures."""
        if not fixtures:
            return

        # Columnas a insertar (excluir json_data que es muy grande)
        columns = [
            'fixture_id', 'date', 'timestamp', 'timezone', 'venue_id', 'venue_name',
            'venue_city', 'status_long', 'status_short', 'status_elapsed',
            'league_id', 'league_name', 'league_country', 'league_season', 'league_round',
            'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
            'goals_home', 'goals_away', 'score_halftime_home', 'score_halftime_away',
            'score_fulltime_home', 'score_fulltime_away', 'created_at', 'updated_at',
            'stats_fetched', 'odds_fetched', 'home_shots_on_goal', 'home_total_shots',
            'home_ball_possession', 'away_shots_on_goal', 'away_total_shots',
            'away_ball_possession', 'odds_home', 'odds_draw', 'odds_away', 'match_type'
        ]

        placeholders = ','.join(['?' for _ in columns])
        sql = f"INSERT OR REPLACE INTO api_fixtures ({','.join(columns)}) VALUES ({placeholders})"

        rows = []
        for f in fixtures:
            row = tuple(f.get(col) for col in columns)
            rows.append(row)

        self.cursor.executemany(sql, rows)
        self.conn.commit()
        logger.info(f"Insertados {len(rows)} fixtures")

    def insert_features(self, features: List[Dict]):
        """Inserta features en la tabla api_features."""
        if not features:
            return

        # Obtener todas las columnas del primer registro
        columns = list(features[0].keys())

        # Crear tabla dinámicamente
        col_defs = []
        for col in columns:
            if col == 'fixture_id':
                col_defs.append('fixture_id INTEGER PRIMARY KEY')
            elif col in ['is_home_new_team', 'is_away_new_team', 'has_h2h_history']:
                col_defs.append(f'{col} INTEGER')
            elif col in ['result']:
                col_defs.append(f'{col} TEXT')
            elif col in ['calculated_at']:
                col_defs.append(f'{col} TEXT')
            elif col in ['features_version', 'home_goals', 'away_goals', 'total_goals',
                        'home_form_matches_available', 'away_form_matches_available',
                        'day_of_week', 'month']:
                col_defs.append(f'{col} INTEGER')
            else:
                col_defs.append(f'{col} REAL')

        create_sql = f"CREATE TABLE IF NOT EXISTS api_features ({', '.join(col_defs)})"
        self.cursor.execute(create_sql)

        # Insertar datos
        placeholders = ','.join(['?' for _ in columns])
        sql = f"INSERT OR REPLACE INTO api_features ({','.join(columns)}) VALUES ({placeholders})"

        rows = []
        for f in features:
            row = tuple(f.get(col) for col in columns)
            rows.append(row)

        # Insertar en batches
        batch_size = 1000
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            self.cursor.executemany(sql, batch)
            self.conn.commit()
            logger.info(f"  Insertadas {min(i+batch_size, len(rows))}/{len(rows)} features...")

        # Crear índice
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_features_fixture ON api_features(fixture_id)')
        self.conn.commit()

        logger.info(f"Insertados {len(rows)} registros de features")

    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la base de datos."""
        stats = {}

        self.cursor.execute('SELECT COUNT(*) FROM api_fixtures')
        stats['fixtures'] = self.cursor.fetchone()[0]

        self.cursor.execute('SELECT COUNT(*) FROM api_fixtures WHERE goals_home IS NOT NULL')
        stats['fixtures_finished'] = self.cursor.fetchone()[0]

        try:
            self.cursor.execute('SELECT COUNT(*) FROM api_features')
            stats['features'] = self.cursor.fetchone()[0]
        except:
            stats['features'] = 0

        return stats


def main():
    parser = argparse.ArgumentParser(description='Exportar API-Football de Supabase a SQLite')
    parser.add_argument('--limit', type=int, default=None,
                        help='Límite de registros (para pruebas)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EXPORTACIÓN API-FOOTBALL → SQLITE")
    logger.info("=" * 60)

    # Conectar a Supabase
    supabase = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)

    # Conectar a SQLite
    db = SQLiteDatabase(DB_PATH)
    db.connect()
    db.create_tables()

    try:
        # 1. Exportar fixtures
        logger.info("\n[1/2] Exportando fixtures...")
        fixtures = supabase.fetch_table(
            'fixtures',
            order='fixture_id',
            limit=args.limit
        )
        logger.info(f"Obtenidos {len(fixtures)} fixtures de Supabase")

        db.insert_fixtures(fixtures)

        # 2. Exportar features
        logger.info("\n[2/2] Exportando fixture_features...")
        features = supabase.fetch_table(
            'fixture_features',
            order='fixture_id',
            limit=args.limit
        )
        logger.info(f"Obtenidos {len(features)} registros de features de Supabase")

        db.insert_features(features)

        # Estadísticas finales
        logger.info("\n" + "=" * 60)
        logger.info("EXPORTACIÓN COMPLETADA")
        logger.info("=" * 60)

        stats = db.get_stats()
        logger.info(f"Total fixtures: {stats['fixtures']:,}")
        logger.info(f"Fixtures terminados: {stats['fixtures_finished']:,}")
        logger.info(f"Total features: {stats['features']:,}")
        logger.info(f"Base de datos: {DB_PATH}")

        # Tamaño del archivo
        db_size = os.path.getsize(DB_PATH) / (1024 * 1024)
        logger.info(f"Tamaño: {db_size:.1f} MB")

    finally:
        db.close()


if __name__ == '__main__':
    main()
