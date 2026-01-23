#!/usr/bin/env python3
"""
04_prepare_training_data.py
Prepara el dataset final para entrenamiento combinando features + cuotas.

Proceso:
1. JOIN api_features + hybrid_odds + event_mapping
2. Filtrar partidos con features completas, cuotas y resultado
3. Agregar features derivadas de las cuotas
4. Crear tabla 'training_data' lista para ML

Uso:
    python 04_prepare_training_data.py
    python 04_prepare_training_data.py --stats  # Solo mostrar estadísticas
"""

import os
import sys
import sqlite3
import logging
import argparse
from datetime import datetime
from typing import Dict, List

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


class TrainingDataPreparer:
    """Prepara los datos de entrenamiento."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Conecta a la base de datos."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Conectado a {self.db_path}")

    def close(self):
        """Cierra la conexión."""
        if self.conn:
            self.conn.close()

    def get_stats(self) -> Dict:
        """Obtiene estadísticas de las tablas."""
        cursor = self.conn.cursor()
        stats = {}

        # Fixtures
        cursor.execute('SELECT COUNT(*) FROM api_fixtures')
        stats['fixtures_total'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM api_fixtures WHERE goals_home IS NOT NULL')
        stats['fixtures_finished'] = cursor.fetchone()[0]

        # Features
        cursor.execute('SELECT COUNT(*) FROM api_features')
        stats['features_total'] = cursor.fetchone()[0]

        # Mapping
        try:
            cursor.execute('SELECT COUNT(*) FROM event_mapping')
            stats['mappings'] = cursor.fetchone()[0]
        except:
            stats['mappings'] = 0

        # Odds
        try:
            cursor.execute('SELECT COUNT(*) FROM hybrid_odds WHERE odds_home_close IS NOT NULL')
            stats['with_odds'] = cursor.fetchone()[0]
        except:
            stats['with_odds'] = 0

        # Training data (si existe)
        try:
            cursor.execute('SELECT COUNT(*) FROM training_data')
            stats['training_data'] = cursor.fetchone()[0]
        except:
            stats['training_data'] = 0

        return stats

    def show_stats(self):
        """Muestra estadísticas actuales."""
        stats = self.get_stats()

        logger.info("\n" + "=" * 60)
        logger.info("ESTADÍSTICAS DE LA BASE DE DATOS HÍBRIDA")
        logger.info("=" * 60)
        logger.info(f"Fixtures totales: {stats['fixtures_total']:,}")
        logger.info(f"Fixtures terminados: {stats['fixtures_finished']:,}")
        logger.info(f"Features calculadas: {stats['features_total']:,}")
        logger.info(f"Mappings API↔SofaScore: {stats['mappings']:,}")
        logger.info(f"Con cuotas de SofaScore: {stats['with_odds']:,}")
        logger.info(f"Training data: {stats['training_data']:,}")

        if stats['fixtures_finished'] > 0:
            logger.info(f"\nCobertura:")
            logger.info(f"  Mapping: {stats['mappings']/stats['fixtures_finished']*100:.1f}%")
            if stats['mappings'] > 0:
                logger.info(f"  Con cuotas: {stats['with_odds']/stats['mappings']*100:.1f}%")

    def prepare_training_data(self):
        """Prepara la tabla de training_data."""
        cursor = self.conn.cursor()

        # Verificar que existen las tablas necesarias
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        required = ['api_fixtures', 'api_features', 'event_mapping', 'hybrid_odds']
        missing = [t for t in required if t not in tables]

        if missing:
            logger.error(f"Faltan tablas: {missing}")
            logger.error("Ejecuta los scripts anteriores primero.")
            return

        logger.info("Creando tabla training_data...")

        # Eliminar tabla si existe
        cursor.execute('DROP TABLE IF EXISTS training_data')

        # Obtener columnas de api_features (excluyendo fixture_id que ya está en fixtures)
        cursor.execute('PRAGMA table_info(api_features)')
        feature_cols = [col[1] for col in cursor.fetchall() if col[1] != 'fixture_id']

        # Crear la tabla con todos los datos combinados
        # Usamos una vista materializada (tabla creada desde SELECT)

        # Primero crear la query
        feature_select = ', '.join([f'feat.{col}' for col in feature_cols])

        sql = f'''
            CREATE TABLE training_data AS
            SELECT
                -- Datos básicos del fixture
                f.fixture_id,
                f.date,
                f.league_name,
                f.league_country,
                f.home_team_name,
                f.away_team_name,
                f.goals_home,
                f.goals_away,

                -- Features calculadas
                {feature_select},

                -- Cuotas de SofaScore
                o.odds_home_open,
                o.odds_home_close,
                o.odds_draw_open,
                o.odds_draw_close,
                o.odds_away_open,
                o.odds_away_close,
                o.odds_over25_open,
                o.odds_over25_close,
                o.odds_under25_open,
                o.odds_under25_close,
                o.odds_btts_yes_open,
                o.odds_btts_yes_close,
                o.odds_btts_no_open,
                o.odds_btts_no_close,

                -- Features derivadas de cuotas
                CASE
                    WHEN o.odds_home_close IS NOT NULL THEN 1.0 / o.odds_home_close
                    ELSE NULL
                END as implied_prob_home,
                CASE
                    WHEN o.odds_draw_close IS NOT NULL THEN 1.0 / o.odds_draw_close
                    ELSE NULL
                END as implied_prob_draw,
                CASE
                    WHEN o.odds_away_close IS NOT NULL THEN 1.0 / o.odds_away_close
                    ELSE NULL
                END as implied_prob_away,

                -- Movimiento de cuotas (apertura vs cierre)
                CASE
                    WHEN o.odds_home_open IS NOT NULL AND o.odds_home_close IS NOT NULL
                    THEN o.odds_home_close - o.odds_home_open
                    ELSE NULL
                END as odds_movement_home,
                CASE
                    WHEN o.odds_draw_open IS NOT NULL AND o.odds_draw_close IS NOT NULL
                    THEN o.odds_draw_close - o.odds_draw_open
                    ELSE NULL
                END as odds_movement_draw,
                CASE
                    WHEN o.odds_away_open IS NOT NULL AND o.odds_away_close IS NOT NULL
                    THEN o.odds_away_close - o.odds_away_open
                    ELSE NULL
                END as odds_movement_away,

                -- Metadata de matching
                m.match_score as mapping_score,
                m.match_method as mapping_method

            FROM api_fixtures f
            INNER JOIN api_features feat ON f.fixture_id = feat.fixture_id
            INNER JOIN event_mapping m ON f.fixture_id = m.fixture_id
            INNER JOIN hybrid_odds o ON f.fixture_id = o.fixture_id
            WHERE f.goals_home IS NOT NULL
            AND o.odds_home_close IS NOT NULL
        '''

        cursor.execute(sql)

        # Crear índices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_fixture ON training_data(fixture_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_date ON training_data(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_league ON training_data(league_name)')

        self.conn.commit()

        # Estadísticas
        cursor.execute('SELECT COUNT(*) FROM training_data')
        count = cursor.fetchone()[0]

        cursor.execute('SELECT MIN(date), MAX(date) FROM training_data')
        date_range = cursor.fetchone()

        cursor.execute('SELECT COUNT(DISTINCT league_name) FROM training_data')
        leagues = cursor.fetchone()[0]

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING DATA CREADA")
        logger.info("=" * 60)
        logger.info(f"Total registros: {count:,}")
        logger.info(f"Período: {date_range[0]} → {date_range[1]}")
        logger.info(f"Ligas únicas: {leagues}")

        # Distribución de resultados
        cursor.execute('''
            SELECT result, COUNT(*) as count
            FROM training_data
            GROUP BY result
            ORDER BY count DESC
        ''')

        logger.info("\nDistribución de resultados:")
        for row in cursor.fetchall():
            pct = row[1] / count * 100
            logger.info(f"  {row[0]}: {row[1]:,} ({pct:.1f}%)")

        # Verificar features de cuotas
        cursor.execute('''
            SELECT
                AVG(implied_prob_home) as avg_prob_h,
                AVG(implied_prob_draw) as avg_prob_d,
                AVG(implied_prob_away) as avg_prob_a,
                AVG(odds_movement_home) as avg_mov_h
            FROM training_data
        ''')

        row = cursor.fetchone()
        logger.info(f"\nProbabilidades implícitas promedio:")
        logger.info(f"  Home: {row[0]:.3f}")
        logger.info(f"  Draw: {row[1]:.3f}")
        logger.info(f"  Away: {row[2]:.3f}")
        logger.info(f"  Suma: {row[0]+row[1]+row[2]:.3f} (margen de la casa)")
        logger.info(f"\nMovimiento promedio cuota Home: {row[3]:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Preparar datos de entrenamiento')
    parser.add_argument('--stats', action='store_true',
                        help='Solo mostrar estadísticas')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PREPARACIÓN DE DATOS DE ENTRENAMIENTO")
    logger.info("=" * 60)

    preparer = TrainingDataPreparer(HYBRID_DB_PATH)

    try:
        preparer.connect()

        if args.stats:
            preparer.show_stats()
        else:
            preparer.prepare_training_data()
            preparer.show_stats()
    finally:
        preparer.close()


if __name__ == '__main__':
    main()
