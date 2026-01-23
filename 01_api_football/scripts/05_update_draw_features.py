"""
Script optimizado para actualizar las nuevas features de empates en fixture_features.
Usa batch updates para mayor velocidad.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import time
import logging
import os
import urllib3
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = "https://ykqaplnfrhvjqkvejudg.supabase.co"
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlrcWFwbG5mcmh2anFrdmVqdWRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg2NjY4NjgsImV4cCI6MjA4NDI0Mjg2OH0.abeJY6QxUn4gT5GYJmoD2xJ7uPVNEwAVAxJ0wE5bMvM')

HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json',
    'Prefer': 'resolution=merge-duplicates'
}

BATCH_SIZE = 500  # Aumentado para batch upserts


def fetch_all_fixtures() -> pd.DataFrame:
    """Obtiene todos los fixtures de Supabase."""
    logger.info("Obteniendo todos los fixtures...")

    url = f"{SUPABASE_URL}/rest/v1/fixtures"
    params = {
        'select': 'fixture_id,date,home_team_id,away_team_id,goals_home,goals_away,match_type',
        'order': 'date.asc,fixture_id.asc'
    }

    all_data = []
    offset = 0
    page_size = 1000

    while True:
        params['offset'] = offset
        params['limit'] = page_size
        response = requests.get(url, headers=HEADERS, params=params, verify=False, timeout=60)

        if response.status_code != 200:
            raise Exception(f"Error: {response.text}")

        data = response.json()
        if not data:
            break

        all_data.extend(data)
        offset += page_size

        if len(data) < page_size:
            break

    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Obtenidos {len(df):,} fixtures")
    return df


def fetch_fixture_ids_without_draw_features() -> list:
    """Obtiene IDs de fixtures sin features de empates."""
    logger.info("Obteniendo IDs sin features de empates...")

    url = f"{SUPABASE_URL}/rest/v1/fixture_features"
    all_ids = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            'select': 'fixture_id',
            'home_draw_rate_last10': 'is.null',
            'order': 'fixture_id.asc',
            'offset': offset,
            'limit': page_size
        }
        response = requests.get(url, headers=HEADERS, params=params, verify=False, timeout=60)

        if response.status_code != 200:
            break

        data = response.json()
        if not data:
            break

        all_ids.extend([r['fixture_id'] for r in data])
        offset += page_size

        if len(data) < page_size:
            break

    logger.info(f"Fixtures sin features de empates: {len(all_ids):,}")
    return all_ids


def get_team_matches_before_date(df: pd.DataFrame, team_id: int, date) -> pd.DataFrame:
    """Obtiene partidos de un equipo antes de una fecha."""
    mask = (df['match_type'] == 'Terminado') & (df['date'] < date)
    mask &= (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
    return df[mask].sort_values('date', ascending=False)


def calculate_draw_features(df: pd.DataFrame, team_id: int, date, n_matches: int = 10) -> dict:
    """Calcula features específicas para detectar empates."""
    matches = get_team_matches_before_date(df, team_id, date).head(n_matches)

    if len(matches) == 0:
        return {
            'draw_rate': None,
            'result_volatility': None,
            'balance_ratio': None,
            'low_scoring_rate': None,
            'defense_strength': None
        }

    draws = 0
    points_list = []
    goals_for_total = 0
    goals_against_total = 0
    clean_sheets = 0
    low_scoring_matches = 0

    for _, match in matches.iterrows():
        is_home = match['home_team_id'] == team_id

        if is_home:
            gf = match['goals_home'] or 0
            ga = match['goals_away'] or 0
        else:
            gf = match['goals_away'] or 0
            ga = match['goals_home'] or 0

        goals_for_total += gf
        goals_against_total += ga

        if gf > ga:
            points_list.append(3)
        elif gf == ga:
            points_list.append(1)
            draws += 1
        else:
            points_list.append(0)

        if ga == 0:
            clean_sheets += 1
        if gf + ga <= 2:
            low_scoring_matches += 1

    n = len(matches)

    return {
        'draw_rate': round(draws / n * 100, 2),
        'result_volatility': round(float(np.std(points_list)), 2) if len(points_list) > 1 else None,
        'balance_ratio': round(goals_for_total / (goals_against_total + 1), 2),
        'low_scoring_rate': round(low_scoring_matches / n * 100, 2),
        'defense_strength': round(clean_sheets / n * 100, 2)
    }


def calculate_momentum_balance(df: pd.DataFrame, home_id: int, away_id: int, date) -> float:
    """Calcula el balance de momentum entre equipos."""
    def get_ppg(team_id):
        matches = get_team_matches_before_date(df, team_id, date).head(5)
        if len(matches) == 0:
            return None

        points = 0
        for _, m in matches.iterrows():
            is_home = m['home_team_id'] == team_id
            gf = (m['goals_home'] if is_home else m['goals_away']) or 0
            ga = (m['goals_away'] if is_home else m['goals_home']) or 0
            points += 3 if gf > ga else (1 if gf == ga else 0)
        return points / len(matches)

    home_ppg = get_ppg(home_id)
    away_ppg = get_ppg(away_id)

    if home_ppg is not None and away_ppg is not None:
        ppg_diff = abs(home_ppg - away_ppg)
        return round(1 - (ppg_diff / 3.0), 2)
    return None


def batch_upsert(updates: list) -> tuple:
    """Hace upsert de un batch de registros."""
    if not updates:
        return 0, 0

    url = f"{SUPABASE_URL}/rest/v1/fixture_features"

    try:
        response = requests.post(url, headers=HEADERS, json=updates, verify=False, timeout=120)

        if response.status_code in [200, 201]:
            return len(updates), 0
        else:
            logger.error(f"Error batch: {response.status_code} - {response.text[:200]}")
            return 0, len(updates)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 0, len(updates)


def main():
    logger.info("=" * 60)
    logger.info("ACTUALIZANDO FEATURES DE EMPATES (OPTIMIZADO)")
    logger.info("=" * 60)

    start_time = time.time()

    # Cargar datos
    df = fetch_all_fixtures()
    fixture_ids = fetch_fixture_ids_without_draw_features()

    if not fixture_ids:
        logger.info("No hay fixtures pendientes de actualizar")
        return

    logger.info(f"\nProcesando {len(fixture_ids):,} registros pendientes...")

    updates = []
    total_ok = 0
    total_fail = 0

    for idx, fixture_id in enumerate(fixture_ids):
        fixture = df[df['fixture_id'] == fixture_id]

        if len(fixture) == 0:
            continue

        fixture = fixture.iloc[0]
        date = fixture['date']
        home_id = fixture['home_team_id']
        away_id = fixture['away_team_id']

        # Calcular nuevas features
        home_draw = calculate_draw_features(df, home_id, date, 10)
        away_draw = calculate_draw_features(df, away_id, date, 10)
        momentum = calculate_momentum_balance(df, home_id, away_id, date)

        update = {
            'fixture_id': fixture_id,
            'home_draw_rate_last10': home_draw['draw_rate'],
            'home_result_volatility': home_draw['result_volatility'],
            'home_balance_ratio': home_draw['balance_ratio'],
            'home_low_scoring_rate': home_draw['low_scoring_rate'],
            'home_defense_strength': home_draw['defense_strength'],
            'away_draw_rate_last10': away_draw['draw_rate'],
            'away_result_volatility': away_draw['result_volatility'],
            'away_balance_ratio': away_draw['balance_ratio'],
            'away_low_scoring_rate': away_draw['low_scoring_rate'],
            'away_defense_strength': away_draw['defense_strength'],
            'momentum_balance': momentum
        }

        updates.append(update)

        if len(updates) >= BATCH_SIZE:
            ok, fail = batch_upsert(updates)
            total_ok += ok
            total_fail += fail
            updates = []

            progress = (idx + 1) / len(fixture_ids) * 100
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(fixture_ids) - idx - 1) / rate / 60 if rate > 0 else 0
            logger.info(f"Progreso: {progress:.1f}% ({idx + 1:,}/{len(fixture_ids):,}) - OK: {total_ok:,} - ETA: {remaining:.1f} min")

    # Procesar remanente
    if updates:
        ok, fail = batch_upsert(updates)
        total_ok += ok
        total_fail += fail

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"COMPLETADO: {total_ok:,} OK, {total_fail:,} fallidos")
    logger.info(f"Tiempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
