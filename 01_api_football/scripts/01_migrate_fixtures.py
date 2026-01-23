"""
Script de migración: Excel a Supabase
Tabla: fixtures (partidos de fútbol)
Registros: ~108,446
Usa requests directamente con la API REST de Supabase
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import json
import time
import logging
import os
import urllib3

# Deshabilitar warnings de SSL para entornos corporativos con proxy
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURACIÓN
# ============================================

SUPABASE_URL = "https://ykqaplnfrhvjqkvejudg.supabase.co"
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlrcWFwbG5mcmh2anFrdmVqdWRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg2NjY4NjgsImV4cCI6MjA4NDI0Mjg2OH0.abeJY6QxUn4gT5GYJmoD2xJ7uPVNEwAVAxJ0wE5bMvM')

EXCEL_PATH = r"c:\Users\peralhe001\Documents\proyecto\full_database_export.xlsx"
TABLE_NAME = "fixtures"
BATCH_SIZE = 500  # Registros por lote

# Headers para la API REST
HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json',
    'Prefer': 'resolution=merge-duplicates'  # Para upsert
}

# ============================================
# FUNCIONES DE UTILIDAD
# ============================================

def clean_value(value, target_type='string'):
    """
    Limpia valores para compatibilidad con PostgreSQL.
    """
    if pd.isna(value) or value is pd.NaT:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    if target_type == 'int':
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    elif target_type == 'float':
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    elif target_type == 'datetime':
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        elif isinstance(value, datetime):
            return value.isoformat()
        return str(value) if value else None
    elif target_type == 'json':
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {"raw": value}
        return value
    else:
        return str(value) if value is not None else None


def transform_row(row: pd.Series) -> dict:
    """
    Transforma una fila del DataFrame al formato esperado por Supabase.
    """
    return {
        'fixture_id': clean_value(row['fixture_id'], 'int'),
        'date': clean_value(row['date'], 'datetime'),
        'timestamp': clean_value(row['timestamp'], 'int'),
        'timezone': clean_value(row['timezone']),
        'venue_id': clean_value(row['venue_id'], 'int'),
        'venue_name': clean_value(row['venue_name']),
        'venue_city': clean_value(row['venue_city']),
        'status_long': clean_value(row['status_long']),
        'status_short': clean_value(row['status_short']),
        'status_elapsed': clean_value(row['status_elapsed'], 'int'),
        'league_id': clean_value(row['league_id'], 'int'),
        'league_name': clean_value(row['league_name']),
        'league_country': clean_value(row['league_country']),
        'league_season': clean_value(row['league_season'], 'int'),
        'league_round': clean_value(row['league_round']),
        'home_team_id': clean_value(row['home_team_id'], 'int'),
        'home_team_name': clean_value(row['home_team_name']),
        'away_team_id': clean_value(row['away_team_id'], 'int'),
        'away_team_name': clean_value(row['away_team_name']),
        'goals_home': clean_value(row['goals_home'], 'int'),
        'goals_away': clean_value(row['goals_away'], 'int'),
        'score_halftime_home': clean_value(row['score_halftime_home'], 'int'),
        'score_halftime_away': clean_value(row['score_halftime_away'], 'int'),
        'score_fulltime_home': clean_value(row['score_fulltime_home'], 'int'),
        'score_fulltime_away': clean_value(row['score_fulltime_away'], 'int'),
        'json_data': clean_value(row['json_data'], 'json'),
        'created_at': clean_value(row['created_at'], 'datetime'),
        'updated_at': clean_value(row['updated_at'], 'datetime'),
        'stats_fetched': clean_value(row['stats_fetched'], 'int'),
        'odds_fetched': clean_value(row['odds_fetched'], 'int'),
        'home_shots_on_goal': clean_value(row['home_shots_on_goal'], 'int'),
        'home_total_shots': clean_value(row['home_total_shots'], 'int'),
        'home_ball_possession': clean_value(row['home_ball_possession'], 'float'),
        'away_shots_on_goal': clean_value(row['away_shots_on_goal'], 'int'),
        'away_total_shots': clean_value(row['away_total_shots'], 'int'),
        'away_ball_possession': clean_value(row['away_ball_possession'], 'float'),
        'odds_home': clean_value(row['odds_home'], 'float'),
        'odds_draw': clean_value(row['odds_draw'], 'float'),
        'odds_away': clean_value(row['odds_away'], 'float'),
        'match_type': clean_value(row['match_type']) or 'Otro',
    }


def upsert_batch(records: list, batch_num: int) -> tuple:
    """
    Realiza upsert de un lote de registros usando la API REST.
    """
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"

    try:
        response = requests.post(
            url,
            headers=HEADERS,
            json=records,
            timeout=60,
            verify=False
        )

        if response.status_code in [200, 201]:
            return len(records), 0
        else:
            logger.error(f"Error en batch {batch_num}: {response.status_code} - {response.text[:200]}")
            # Intentar uno por uno
            exitosos = 0
            fallidos = 0
            for record in records:
                try:
                    r = requests.post(url, headers=HEADERS, json=[record], timeout=30, verify=False)
                    if r.status_code in [200, 201]:
                        exitosos += 1
                    else:
                        fallidos += 1
                        logger.error(f"Error fixture_id {record.get('fixture_id')}: {r.text[:100]}")
                except Exception as e:
                    fallidos += 1
            return exitosos, fallidos

    except Exception as e:
        logger.error(f"Error en batch {batch_num}: {str(e)}")
        return 0, len(records)


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def migrate_fixtures():
    """
    Ejecuta la migración completa.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO MIGRACIÓN DE FIXTURES A SUPABASE")
    logger.info("=" * 60)

    # 1. Verificar conexión
    logger.info("Verificando conexión a Supabase...")
    test_url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}?select=fixture_id&limit=1"
    test_response = requests.get(test_url, headers=HEADERS, timeout=10, verify=False)
    if test_response.status_code != 200:
        raise Exception(f"No se pudo conectar a Supabase: {test_response.text}")
    logger.info("Conexión verificada")

    # 2. Leer archivo Excel
    logger.info(f"Leyendo archivo Excel: {EXCEL_PATH}")
    start_read = time.time()
    df = pd.read_excel(EXCEL_PATH)
    read_time = time.time() - start_read
    logger.info(f"Archivo leído en {read_time:.2f}s")
    logger.info(f"Total de registros: {len(df):,}")

    # 3. Transformar datos
    logger.info("Transformando datos...")
    start_transform = time.time()
    records = [transform_row(row) for _, row in df.iterrows()]
    transform_time = time.time() - start_transform
    logger.info(f"Transformación completada en {transform_time:.2f}s")

    # 4. Ejecutar upsert por lotes
    logger.info(f"Iniciando upsert en lotes de {BATCH_SIZE}...")
    total_records = len(records)
    total_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
    total_exitosos = 0
    total_fallidos = 0

    start_upsert = time.time()

    for i in range(0, total_records, BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        batch = records[i:i + BATCH_SIZE]

        exitosos, fallidos = upsert_batch(batch, batch_num)
        total_exitosos += exitosos
        total_fallidos += fallidos

        progress = (batch_num / total_batches) * 100
        logger.info(
            f"Batch {batch_num}/{total_batches} ({progress:.1f}%) - "
            f"Exitosos: {exitosos}, Fallidos: {fallidos}"
        )

        # Pausa para no sobrecargar
        if batch_num % 5 == 0:
            time.sleep(0.2)

    upsert_time = time.time() - start_upsert

    # 5. Resumen
    logger.info("=" * 60)
    logger.info("MIGRACIÓN COMPLETADA")
    logger.info("=" * 60)
    logger.info(f"Total registros procesados: {total_records:,}")
    logger.info(f"Registros exitosos: {total_exitosos:,}")
    logger.info(f"Registros fallidos: {total_fallidos:,}")
    logger.info(f"Tiempo total: {read_time + transform_time + upsert_time:.2f}s")

    return total_exitosos, total_fallidos


def verify_migration():
    """
    Verifica la migración.
    """
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}?select=fixture_id"
    headers = {**HEADERS, 'Prefer': 'count=exact'}

    response = requests.head(url, headers=headers, timeout=10, verify=False)
    count = response.headers.get('content-range', '').split('/')[-1]
    logger.info(f"Registros en Supabase: {count}")

    # Muestra
    sample_url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}?select=fixture_id,home_team_name,away_team_name&limit=5"
    sample = requests.get(sample_url, headers=HEADERS, timeout=10, verify=False).json()
    logger.info("Muestra de registros:")
    for row in sample:
        logger.info(f"  - {row['fixture_id']}: {row['home_team_name']} vs {row['away_team_name']}")

    return count


if __name__ == "__main__":
    try:
        exitosos, fallidos = migrate_fixtures()
        print("\n")
        verify_migration()
    except Exception as e:
        logger.error(f"Error fatal: {str(e)}")
        raise
