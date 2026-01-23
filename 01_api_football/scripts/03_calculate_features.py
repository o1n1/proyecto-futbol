"""
Script para calcular features derivadas de partidos de fútbol.
Calcula ~90 features para cada partido basándose ÚNICAMENTE en datos históricos.

Principios:
1. Features para TODOS los partidos (terminados Y futuros)
2. Repetibilidad matemática: Mismo input → Mismo output
3. Cero data leakage: Solo datos con fecha < fecha_partido
4. Optimización de recursos: Minimizar llamadas API y procesamiento

Categorías de Features:
- Forma reciente (últimos 3, 5, 10 partidos)
- Forma específica local/visitante
- Estadísticas de temporada
- Head-to-Head (H2H)
- Estadísticas de liga
- Rachas actuales
- Goles por mitad
- Días de descanso
- Features combinadas (ataque vs defensa, momentum)
- Contextuales (día semana, mes, inicio/fin temporada)
- Indicadores (is_new_team, has_h2h_history, matches_available)
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import json
import time
import logging
import os
import urllib3
from typing import Dict, List, Optional, Tuple

# Deshabilitar warnings de SSL
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

HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json',
}

BATCH_SIZE = 500
MAX_FIXTURES_PER_RUN = 5000  # Límite por ejecución para GitHub Actions


# ============================================
# FUNCIONES DE DATOS
# ============================================

def fetch_all_fixtures() -> pd.DataFrame:
    """Obtiene todos los fixtures de Supabase."""
    logger.info("Obteniendo todos los fixtures de Supabase...")

    url = f"{SUPABASE_URL}/rest/v1/fixtures"
    params = {
        'select': 'fixture_id,date,home_team_id,home_team_name,away_team_id,away_team_name,'
                  'goals_home,goals_away,score_halftime_home,score_halftime_away,'
                  'league_id,league_name,league_season,match_type',
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
            raise Exception(f"Error al obtener fixtures: {response.text}")

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


def fetch_all_ids_paginated(url_base: str) -> set:
    """Obtiene todos los IDs con paginación."""
    all_ids = set()
    offset = 0
    page_size = 1000

    while True:
        url = f"{url_base}&offset={offset}&limit={page_size}"
        response = requests.get(url, headers=HEADERS, verify=False, timeout=60)

        if response.status_code != 200:
            break

        data = response.json()
        if not data:
            break

        all_ids.update(r['fixture_id'] for r in data)
        offset += page_size

        if len(data) < page_size:
            break

    return all_ids


def fetch_fixtures_without_features() -> List[int]:
    """Obtiene IDs de fixtures que no tienen features calculadas."""
    logger.info("Buscando fixtures sin features...")

    # Obtener todos los fixture_ids (paginado)
    url_fixtures = f"{SUPABASE_URL}/rest/v1/fixtures?select=fixture_id&order=date.asc"
    all_fixture_ids = fetch_all_ids_paginated(url_fixtures)
    logger.info(f"Total fixtures en BD: {len(all_fixture_ids):,}")

    # Obtener fixture_ids con features (paginado)
    url_features = f"{SUPABASE_URL}/rest/v1/fixture_features?select=fixture_id&order=fixture_id.asc"
    with_features = fetch_all_ids_paginated(url_features)
    logger.info(f"Fixtures con features: {len(with_features):,}")

    # Diferencia
    without_features = list(all_fixture_ids - with_features)
    logger.info(f"Fixtures sin features: {len(without_features):,}")

    return without_features


def fetch_fixtures_pending_targets() -> List[Dict]:
    """Obtiene fixtures terminados que no tienen targets calculados."""
    logger.info("Buscando fixtures con targets pendientes...")

    url = f"{SUPABASE_URL}/rest/v1/fixture_features?select=fixture_id&result=is.null"
    response = requests.get(url, headers=HEADERS, verify=False, timeout=60)

    if response.status_code != 200:
        return []

    pending_ids = [r['fixture_id'] for r in response.json()]

    if not pending_ids:
        return []

    # Obtener datos de fixtures terminados
    ids_str = ','.join(str(i) for i in pending_ids[:1000])
    url = f"{SUPABASE_URL}/rest/v1/fixtures?fixture_id=in.({ids_str})&match_type=eq.Terminado&select=fixture_id,goals_home,goals_away"
    response = requests.get(url, headers=HEADERS, verify=False, timeout=60)

    if response.status_code != 200:
        return []

    logger.info(f"Fixtures con targets pendientes: {len(response.json()):,}")
    return response.json()


# ============================================
# FUNCIONES DE CÁLCULO DE FEATURES
# ============================================

def get_team_matches_before_date(df: pd.DataFrame, team_id: int, date: pd.Timestamp,
                                  as_home: bool = None) -> pd.DataFrame:
    """
    Obtiene partidos de un equipo antes de una fecha.

    Args:
        df: DataFrame con todos los fixtures
        team_id: ID del equipo
        date: Fecha límite (exclusiva)
        as_home: None=todos, True=solo local, False=solo visitante
    """
    # Solo partidos terminados antes de la fecha
    mask = (df['match_type'] == 'Terminado') & (df['date'] < date)

    if as_home is None:
        mask &= (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
    elif as_home:
        mask &= df['home_team_id'] == team_id
    else:
        mask &= df['away_team_id'] == team_id

    return df[mask].sort_values('date', ascending=False)


def calculate_form_features(df: pd.DataFrame, team_id: int, date: pd.Timestamp,
                            n_matches: int, as_home: bool = None) -> Dict:
    """
    Calcula features de forma para un equipo.

    Returns dict con keys como:
    - points, wins, draws, losses
    - goals_scored, goals_conceded, goal_diff
    - clean_sheets, failed_to_score, btts, over25
    """
    matches = get_team_matches_before_date(df, team_id, date, as_home).head(n_matches)

    if len(matches) == 0:
        return {
            'points': None, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_scored': None, 'goals_conceded': None, 'goal_diff': None,
            'clean_sheets': 0, 'failed_to_score': 0, 'btts': 0, 'over25': 0
        }

    wins = 0
    draws = 0
    losses = 0
    goals_scored = 0
    goals_conceded = 0
    clean_sheets = 0
    failed_to_score = 0
    btts = 0
    over25 = 0

    for _, match in matches.iterrows():
        is_home = match['home_team_id'] == team_id

        if is_home:
            gf = match['goals_home'] or 0
            ga = match['goals_away'] or 0
        else:
            gf = match['goals_away'] or 0
            ga = match['goals_home'] or 0

        goals_scored += gf
        goals_conceded += ga

        # Resultado
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

        # Estadísticas adicionales
        if ga == 0:
            clean_sheets += 1
        if gf == 0:
            failed_to_score += 1
        if gf > 0 and ga > 0:
            btts += 1
        if gf + ga > 2.5:
            over25 += 1

    n = len(matches)
    points = (wins * 3 + draws) / n if n > 0 else None

    return {
        'points': round(points, 2) if points else None,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'goals_scored': round(goals_scored / n, 2) if n > 0 else None,
        'goals_conceded': round(goals_conceded / n, 2) if n > 0 else None,
        'goal_diff': round((goals_scored - goals_conceded) / n, 2) if n > 0 else None,
        'clean_sheets': clean_sheets,
        'failed_to_score': failed_to_score,
        'btts': btts,
        'over25': over25
    }


def calculate_streak_features(df: pd.DataFrame, team_id: int, date: pd.Timestamp) -> Dict:
    """Calcula rachas actuales del equipo."""
    matches = get_team_matches_before_date(df, team_id, date).head(20)

    if len(matches) == 0:
        return {
            'wins': 0, 'draws': 0, 'losses': 0,
            'unbeaten': 0, 'without_win': 0,
            'scoring': 0, 'conceding': 0, 'clean_sheets': 0
        }

    streaks = {
        'wins': 0, 'draws': 0, 'losses': 0,
        'unbeaten': 0, 'without_win': 0,
        'scoring': 0, 'conceding': 0, 'clean_sheets': 0
    }

    # Calcular cada racha
    for streak_type in ['wins', 'draws', 'losses', 'unbeaten', 'without_win',
                        'scoring', 'conceding', 'clean_sheets']:
        count = 0
        for _, match in matches.iterrows():
            is_home = match['home_team_id'] == team_id
            gf = (match['goals_home'] if is_home else match['goals_away']) or 0
            ga = (match['goals_away'] if is_home else match['goals_home']) or 0

            result = 'W' if gf > ga else ('D' if gf == ga else 'L')

            if streak_type == 'wins' and result == 'W':
                count += 1
            elif streak_type == 'draws' and result == 'D':
                count += 1
            elif streak_type == 'losses' and result == 'L':
                count += 1
            elif streak_type == 'unbeaten' and result != 'L':
                count += 1
            elif streak_type == 'without_win' and result != 'W':
                count += 1
            elif streak_type == 'scoring' and gf > 0:
                count += 1
            elif streak_type == 'conceding' and ga > 0:
                count += 1
            elif streak_type == 'clean_sheets' and ga == 0:
                count += 1
            else:
                break

        streaks[streak_type] = count

    return streaks


def calculate_season_features(df: pd.DataFrame, team_id: int, date: pd.Timestamp,
                               league_id: int, season: int) -> Dict:
    """Calcula estadísticas de temporada del equipo."""
    mask = (
        (df['match_type'] == 'Terminado') &
        (df['date'] < date) &
        (df['league_id'] == league_id) &
        (df['league_season'] == season) &
        ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id))
    )

    matches = df[mask]

    if len(matches) == 0:
        return {
            'points_per_game': None,
            'goals_scored_avg': None,
            'goals_conceded_avg': None,
            'home_ppg': None,
            'away_ppg': None,
            'matches_played': 0,
            'position_estimate': None
        }

    total_points = 0
    total_gf = 0
    total_ga = 0
    home_points = 0
    home_matches = 0
    away_points = 0
    away_matches = 0

    for _, match in matches.iterrows():
        is_home = match['home_team_id'] == team_id
        gf = (match['goals_home'] if is_home else match['goals_away']) or 0
        ga = (match['goals_away'] if is_home else match['goals_home']) or 0

        total_gf += gf
        total_ga += ga

        if gf > ga:
            pts = 3
        elif gf == ga:
            pts = 1
        else:
            pts = 0

        total_points += pts

        if is_home:
            home_points += pts
            home_matches += 1
        else:
            away_points += pts
            away_matches += 1

    n = len(matches)

    # Estimar posición basada en PPG relativo
    all_teams_in_league = df[
        (df['match_type'] == 'Terminado') &
        (df['date'] < date) &
        (df['league_id'] == league_id) &
        (df['league_season'] == season)
    ]

    team_ppgs = {}
    for tid in set(all_teams_in_league['home_team_id'].tolist() +
                   all_teams_in_league['away_team_id'].tolist()):
        team_matches = all_teams_in_league[
            (all_teams_in_league['home_team_id'] == tid) |
            (all_teams_in_league['away_team_id'] == tid)
        ]
        if len(team_matches) > 0:
            pts = 0
            for _, m in team_matches.iterrows():
                is_h = m['home_team_id'] == tid
                gf = (m['goals_home'] if is_h else m['goals_away']) or 0
                ga = (m['goals_away'] if is_h else m['goals_home']) or 0
                pts += 3 if gf > ga else (1 if gf == ga else 0)
            team_ppgs[tid] = pts / len(team_matches)

    if team_id in team_ppgs:
        position = sum(1 for ppg in team_ppgs.values() if ppg > team_ppgs[team_id]) + 1
    else:
        position = None

    return {
        'points_per_game': round(total_points / n, 2),
        'goals_scored_avg': round(total_gf / n, 2),
        'goals_conceded_avg': round(total_ga / n, 2),
        'home_ppg': round(home_points / home_matches, 2) if home_matches > 0 else None,
        'away_ppg': round(away_points / away_matches, 2) if away_matches > 0 else None,
        'matches_played': n,
        'position_estimate': position
    }


def calculate_h2h_features(df: pd.DataFrame, home_team_id: int, away_team_id: int,
                            date: pd.Timestamp) -> Dict:
    """Calcula features de enfrentamientos directos."""
    mask = (
        (df['match_type'] == 'Terminado') &
        (df['date'] < date) &
        (
            ((df['home_team_id'] == home_team_id) & (df['away_team_id'] == away_team_id)) |
            ((df['home_team_id'] == away_team_id) & (df['away_team_id'] == home_team_id))
        )
    )

    matches = df[mask]

    if len(matches) == 0:
        return {
            'matches_total': 0,
            'home_wins': 0, 'away_wins': 0, 'draws': 0,
            'home_goals_avg': None, 'away_goals_avg': None,
            'total_goals_avg': None,
            'btts_pct': None, 'over25_pct': None
        }

    home_wins = 0
    away_wins = 0
    draws = 0
    home_goals = 0
    away_goals = 0
    btts = 0
    over25 = 0

    for _, match in matches.iterrows():
        # En este H2H, home_team_id es el "home" del partido actual
        if match['home_team_id'] == home_team_id:
            hg = match['goals_home'] or 0
            ag = match['goals_away'] or 0
        else:
            hg = match['goals_away'] or 0
            ag = match['goals_home'] or 0

        home_goals += hg
        away_goals += ag

        if hg > ag:
            home_wins += 1
        elif hg < ag:
            away_wins += 1
        else:
            draws += 1

        if hg > 0 and ag > 0:
            btts += 1
        if hg + ag > 2.5:
            over25 += 1

    n = len(matches)

    return {
        'matches_total': n,
        'home_wins': home_wins,
        'away_wins': away_wins,
        'draws': draws,
        'home_goals_avg': round(home_goals / n, 2),
        'away_goals_avg': round(away_goals / n, 2),
        'total_goals_avg': round((home_goals + away_goals) / n, 2),
        'btts_pct': round(btts / n * 100, 2),
        'over25_pct': round(over25 / n * 100, 2)
    }


def calculate_league_features(df: pd.DataFrame, league_id: int, season: int,
                               date: pd.Timestamp) -> Dict:
    """Calcula estadísticas generales de la liga."""
    mask = (
        (df['match_type'] == 'Terminado') &
        (df['date'] < date) &
        (df['league_id'] == league_id) &
        (df['league_season'] == season)
    )

    matches = df[mask]

    if len(matches) == 0:
        return {
            'avg_goals': None,
            'home_win_pct': None,
            'draw_pct': None,
            'btts_pct': None,
            'over25_pct': None
        }

    total_goals = 0
    home_wins = 0
    draws = 0
    btts = 0
    over25 = 0

    for _, match in matches.iterrows():
        hg = match['goals_home'] or 0
        ag = match['goals_away'] or 0

        total_goals += hg + ag

        if hg > ag:
            home_wins += 1
        elif hg == ag:
            draws += 1

        if hg > 0 and ag > 0:
            btts += 1
        if hg + ag > 2.5:
            over25 += 1

    n = len(matches)

    return {
        'avg_goals': round(total_goals / n, 2),
        'home_win_pct': round(home_wins / n * 100, 2),
        'draw_pct': round(draws / n * 100, 2),
        'btts_pct': round(btts / n * 100, 2),
        'over25_pct': round(over25 / n * 100, 2)
    }


def calculate_halftime_features(df: pd.DataFrame, team_id: int, date: pd.Timestamp) -> Dict:
    """Calcula features de goles por mitad."""
    matches = get_team_matches_before_date(df, team_id, date).head(10)

    # Filtrar partidos con datos de medio tiempo
    matches = matches[matches['score_halftime_home'].notna()]

    if len(matches) == 0:
        return {
            'avg_goals_first_half': None,
            'avg_goals_second_half': None,
            'avg_conceded_first_half': None,
            'avg_conceded_second_half': None,
            'pct_goals_first_half': None
        }

    first_half_scored = 0
    second_half_scored = 0
    first_half_conceded = 0
    second_half_conceded = 0

    for _, match in matches.iterrows():
        is_home = match['home_team_id'] == team_id

        if is_home:
            ht_gf = match['score_halftime_home'] or 0
            ft_gf = match['goals_home'] or 0
            ht_ga = match['score_halftime_away'] or 0
            ft_ga = match['goals_away'] or 0
        else:
            ht_gf = match['score_halftime_away'] or 0
            ft_gf = match['goals_away'] or 0
            ht_ga = match['score_halftime_home'] or 0
            ft_ga = match['goals_home'] or 0

        first_half_scored += ht_gf
        second_half_scored += (ft_gf - ht_gf)
        first_half_conceded += ht_ga
        second_half_conceded += (ft_ga - ht_ga)

    n = len(matches)
    total_scored = first_half_scored + second_half_scored

    return {
        'avg_goals_first_half': round(first_half_scored / n, 2),
        'avg_goals_second_half': round(second_half_scored / n, 2),
        'avg_conceded_first_half': round(first_half_conceded / n, 2),
        'avg_conceded_second_half': round(second_half_conceded / n, 2),
        'pct_goals_first_half': round(first_half_scored / total_scored * 100, 2) if total_scored > 0 else 50.0
    }


def calculate_days_rest(df: pd.DataFrame, team_id: int, date: pd.Timestamp) -> int:
    """Calcula días desde el último partido."""
    matches = get_team_matches_before_date(df, team_id, date).head(1)

    if len(matches) == 0:
        return None

    last_match_date = matches.iloc[0]['date']
    return (date - last_match_date).days


def calculate_draw_features(df: pd.DataFrame, team_id: int, date: pd.Timestamp,
                            n_matches: int = 10) -> Dict:
    """
    Calcula features específicas para detectar empates.

    Estas features capturan patrones que tienden a resultar en empates:
    - Tasa histórica de empates
    - Consistencia de resultados (baja varianza → más empates)
    - Balance ofensivo-defensivo
    - Tendencia a partidos cerrados (low scoring)
    """
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
    low_scoring_matches = 0  # Partidos con <=2 goles totales

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

        # Resultado
        if gf > ga:
            points_list.append(3)
        elif gf == ga:
            points_list.append(1)
            draws += 1
        else:
            points_list.append(0)

        # Clean sheet
        if ga == 0:
            clean_sheets += 1

        # Partido cerrado (low scoring)
        if gf + ga <= 2:
            low_scoring_matches += 1

    n = len(matches)

    # 1. Tasa de empates histórica
    draw_rate = round(draws / n * 100, 2)

    # 2. Volatilidad de resultados (desviación estándar de puntos)
    # Baja volatilidad = más empates esperados
    result_volatility = round(np.std(points_list), 2) if len(points_list) > 1 else None

    # 3. Balance ratio: goles a favor / (goles en contra + 1)
    # Ratio cercano a 1 = equipos equilibrados = más empates
    balance_ratio = round(goals_for_total / (goals_against_total + 1), 2)

    # 4. Tasa de partidos cerrados (low scoring)
    low_scoring_rate = round(low_scoring_matches / n * 100, 2)

    # 5. Fortaleza defensiva (clean sheets rate)
    defense_strength = round(clean_sheets / n * 100, 2)

    return {
        'draw_rate': draw_rate,
        'result_volatility': result_volatility,
        'balance_ratio': balance_ratio,
        'low_scoring_rate': low_scoring_rate,
        'defense_strength': defense_strength
    }


def calculate_all_features(df: pd.DataFrame, fixture: pd.Series) -> Dict:
    """Calcula todas las features para un fixture."""
    fixture_id = fixture['fixture_id']
    date = fixture['date']
    home_id = fixture['home_team_id']
    away_id = fixture['away_team_id']
    league_id = fixture['league_id']
    season = fixture['league_season']

    features = {
        'fixture_id': fixture_id,
        'calculated_at': datetime.now(timezone.utc).isoformat(),
        'features_version': 1
    }

    # ========== HOME TEAM FEATURES ==========

    # Forma general (3, 5, 10 partidos)
    for n in [3, 5, 10]:
        form = calculate_form_features(df, home_id, date, n)
        for key, value in form.items():
            features[f'home_form_{key}_last{n}'] = value

    # Forma como local específica (últimos 5 en casa)
    home_form = calculate_form_features(df, home_id, date, 5, as_home=True)
    features['home_home_form_points_last5'] = home_form['points']
    features['home_home_form_goals_scored_last5'] = home_form['goals_scored']
    features['home_home_form_goals_conceded_last5'] = home_form['goals_conceded']

    # Estadísticas de temporada
    season_stats = calculate_season_features(df, home_id, date, league_id, season)
    features['home_season_points_per_game'] = season_stats['points_per_game']
    features['home_season_goals_scored_avg'] = season_stats['goals_scored_avg']
    features['home_season_goals_conceded_avg'] = season_stats['goals_conceded_avg']
    features['home_season_home_ppg'] = season_stats['home_ppg']
    features['home_season_matches_played'] = season_stats['matches_played']
    features['home_season_position_estimate'] = season_stats['position_estimate']

    # Rachas
    streaks = calculate_streak_features(df, home_id, date)
    for key, value in streaks.items():
        features[f'home_streak_{key}'] = value

    # Goles por mitad
    halftime = calculate_halftime_features(df, home_id, date)
    features['home_avg_goals_first_half'] = halftime['avg_goals_first_half']
    features['home_avg_goals_second_half'] = halftime['avg_goals_second_half']
    features['home_avg_conceded_first_half'] = halftime['avg_conceded_first_half']
    features['home_avg_conceded_second_half'] = halftime['avg_conceded_second_half']
    features['home_pct_goals_first_half'] = halftime['pct_goals_first_half']

    # ========== AWAY TEAM FEATURES ==========

    # Forma general (3, 5, 10 partidos)
    for n in [3, 5, 10]:
        form = calculate_form_features(df, away_id, date, n)
        for key, value in form.items():
            features[f'away_form_{key}_last{n}'] = value

    # Forma como visitante específica (últimos 5 fuera)
    away_form = calculate_form_features(df, away_id, date, 5, as_home=False)
    features['away_away_form_points_last5'] = away_form['points']
    features['away_away_form_goals_scored_last5'] = away_form['goals_scored']
    features['away_away_form_goals_conceded_last5'] = away_form['goals_conceded']

    # Estadísticas de temporada
    season_stats = calculate_season_features(df, away_id, date, league_id, season)
    features['away_season_points_per_game'] = season_stats['points_per_game']
    features['away_season_goals_scored_avg'] = season_stats['goals_scored_avg']
    features['away_season_goals_conceded_avg'] = season_stats['goals_conceded_avg']
    features['away_season_away_ppg'] = season_stats['away_ppg']
    features['away_season_matches_played'] = season_stats['matches_played']
    features['away_season_position_estimate'] = season_stats['position_estimate']

    # Rachas
    streaks = calculate_streak_features(df, away_id, date)
    for key, value in streaks.items():
        features[f'away_streak_{key}'] = value

    # Goles por mitad
    halftime = calculate_halftime_features(df, away_id, date)
    features['away_avg_goals_first_half'] = halftime['avg_goals_first_half']
    features['away_avg_goals_second_half'] = halftime['avg_goals_second_half']
    features['away_avg_conceded_first_half'] = halftime['avg_conceded_first_half']
    features['away_avg_conceded_second_half'] = halftime['avg_conceded_second_half']
    features['away_pct_goals_first_half'] = halftime['pct_goals_first_half']

    # ========== HEAD TO HEAD ==========
    h2h = calculate_h2h_features(df, home_id, away_id, date)
    features['h2h_matches_total'] = h2h['matches_total']
    features['h2h_home_wins'] = h2h['home_wins']
    features['h2h_away_wins'] = h2h['away_wins']
    features['h2h_draws'] = h2h['draws']
    features['h2h_home_goals_avg'] = h2h['home_goals_avg']
    features['h2h_away_goals_avg'] = h2h['away_goals_avg']
    features['h2h_total_goals_avg'] = h2h['total_goals_avg']
    features['h2h_btts_pct'] = h2h['btts_pct']
    features['h2h_over25_pct'] = h2h['over25_pct']

    # ========== LEAGUE STATS ==========
    league = calculate_league_features(df, league_id, season, date)
    features['league_avg_goals'] = league['avg_goals']
    features['league_home_win_pct'] = league['home_win_pct']
    features['league_draw_pct'] = league['draw_pct']
    features['league_btts_pct'] = league['btts_pct']
    features['league_over25_pct'] = league['over25_pct']

    # ========== DAYS REST ==========
    features['home_days_rest'] = calculate_days_rest(df, home_id, date)
    features['away_days_rest'] = calculate_days_rest(df, away_id, date)

    if features['home_days_rest'] and features['away_days_rest']:
        features['rest_advantage'] = features['home_days_rest'] - features['away_days_rest']
    else:
        features['rest_advantage'] = None

    # ========== COMBINED FEATURES ==========
    # Ataque vs Defensa
    if features['home_season_goals_scored_avg'] and features['away_season_goals_conceded_avg']:
        features['attack_vs_defense_home'] = round(
            features['home_season_goals_scored_avg'] - features['away_season_goals_conceded_avg'], 2
        )
    else:
        features['attack_vs_defense_home'] = None

    if features['away_season_goals_scored_avg'] and features['home_season_goals_conceded_avg']:
        features['attack_vs_defense_away'] = round(
            features['away_season_goals_scored_avg'] - features['home_season_goals_conceded_avg'], 2
        )
    else:
        features['attack_vs_defense_away'] = None

    # Momentum
    if features['home_form_points_last5'] and features['away_form_points_last5']:
        features['form_momentum_diff'] = round(
            features['home_form_points_last5'] - features['away_form_points_last5'], 2
        )
    else:
        features['form_momentum_diff'] = None

    # Expected goals diff
    if features['home_season_goals_scored_avg'] and features['away_season_goals_scored_avg']:
        features['expected_goals_diff'] = round(
            features['home_season_goals_scored_avg'] - features['away_season_goals_scored_avg'], 2
        )
    else:
        features['expected_goals_diff'] = None

    # ========== CONTEXTUAL ==========
    features['is_derby'] = False  # Simplificado por ahora
    features['day_of_week'] = date.dayofweek
    features['month'] = date.month

    # Inicio/fin de temporada (aproximado)
    features['is_season_start'] = date.month in [7, 8]  # Julio-Agosto
    features['is_season_end'] = date.month in [5, 6]  # Mayo-Junio

    # ========== DRAW-SPECIFIC FEATURES (v2.2) ==========
    # Features diseñadas para detectar empates

    # Home team draw features
    home_draw = calculate_draw_features(df, home_id, date, 10)
    features['home_draw_rate_last10'] = home_draw['draw_rate']
    features['home_result_volatility'] = home_draw['result_volatility']
    features['home_balance_ratio'] = home_draw['balance_ratio']
    features['home_low_scoring_rate'] = home_draw['low_scoring_rate']
    features['home_defense_strength'] = home_draw['defense_strength']

    # Away team draw features
    away_draw = calculate_draw_features(df, away_id, date, 10)
    features['away_draw_rate_last10'] = away_draw['draw_rate']
    features['away_result_volatility'] = away_draw['result_volatility']
    features['away_balance_ratio'] = away_draw['balance_ratio']
    features['away_low_scoring_rate'] = away_draw['low_scoring_rate']
    features['away_defense_strength'] = away_draw['defense_strength']

    # Combined draw indicator: momentum balance
    # Cuando ambos equipos tienen PPG similar, más probable empate
    if features['home_form_points_last5'] and features['away_form_points_last5']:
        ppg_diff = abs(features['home_form_points_last5'] - features['away_form_points_last5'])
        # 1 = equipos muy igualados, 0 = muy diferentes
        features['momentum_balance'] = round(1 - (ppg_diff / 3.0), 2)
    else:
        features['momentum_balance'] = None

    # ========== INDICATOR COLUMNS (para ML) ==========
    # Indica si el equipo no tiene historial suficiente
    features['is_home_new_team'] = features['home_form_points_last3'] is None
    features['is_away_new_team'] = features['away_form_points_last3'] is None

    # Indica si hay historial de enfrentamientos directos
    features['has_h2h_history'] = features['h2h_matches_total'] > 0

    # Número de partidos disponibles para calcular forma (0, 3, 5 o 10)
    home_matches = get_team_matches_before_date(df, home_id, date)
    away_matches = get_team_matches_before_date(df, away_id, date)
    features['home_form_matches_available'] = min(len(home_matches), 10)
    features['away_form_matches_available'] = min(len(away_matches), 10)

    # ========== TARGETS (solo si partido terminado) ==========
    if fixture['match_type'] == 'Terminado':
        goals_home = fixture['goals_home']
        goals_away = fixture['goals_away']

        if goals_home is not None and goals_away is not None:
            features['home_goals'] = int(goals_home)
            features['away_goals'] = int(goals_away)
            features['total_goals'] = int(goals_home + goals_away)

            if goals_home > goals_away:
                features['result'] = 'H'
            elif goals_home < goals_away:
                features['result'] = 'A'
            else:
                features['result'] = 'D'

            features['btts'] = goals_home > 0 and goals_away > 0
            features['over15'] = (goals_home + goals_away) > 1.5
            features['over25'] = (goals_home + goals_away) > 2.5
            features['over35'] = (goals_home + goals_away) > 3.5

    return features


# ============================================
# FUNCIONES DE PERSISTENCIA
# ============================================

def insert_features_batch(features_list: List[Dict]) -> Tuple[int, int]:
    """Inserta un lote de features en Supabase."""
    url = f"{SUPABASE_URL}/rest/v1/fixture_features"
    headers = {**HEADERS, 'Prefer': 'resolution=merge-duplicates'}

    try:
        response = requests.post(
            url, headers=headers, json=features_list,
            verify=False, timeout=60
        )

        if response.status_code in [200, 201]:
            return len(features_list), 0
        else:
            logger.error(f"Error en batch insert: {response.status_code} - {response.text[:200]}")
            # Intentar uno por uno
            ok = 0
            fail = 0
            for f in features_list:
                try:
                    r = requests.post(url, headers=headers, json=[f], verify=False, timeout=30)
                    if r.status_code in [200, 201]:
                        ok += 1
                    else:
                        fail += 1
                        logger.error(f"Error fixture {f.get('fixture_id')}: {r.text[:100]}")
                except:
                    fail += 1
            return ok, fail

    except Exception as e:
        logger.error(f"Error en batch: {str(e)}")
        return 0, len(features_list)


def update_targets(fixtures: List[Dict]) -> int:
    """Actualiza targets de fixtures terminados."""
    url = f"{SUPABASE_URL}/rest/v1/fixture_features"
    updated = 0

    for fixture in fixtures:
        goals_home = fixture.get('goals_home')
        goals_away = fixture.get('goals_away')

        if goals_home is None or goals_away is None:
            continue

        targets = {
            'home_goals': int(goals_home),
            'away_goals': int(goals_away),
            'total_goals': int(goals_home + goals_away),
            'result': 'H' if goals_home > goals_away else ('A' if goals_home < goals_away else 'D'),
            'btts': goals_home > 0 and goals_away > 0,
            'over15': (goals_home + goals_away) > 1.5,
            'over25': (goals_home + goals_away) > 2.5,
            'over35': (goals_home + goals_away) > 3.5
        }

        update_url = f"{url}?fixture_id=eq.{fixture['fixture_id']}"
        response = requests.patch(
            update_url, headers=HEADERS, json=targets,
            verify=False, timeout=30
        )

        if response.status_code in [200, 204]:
            updated += 1
        else:
            logger.error(f"Error actualizando targets {fixture['fixture_id']}: {response.text[:100]}")

    return updated


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    """Ejecuta el pipeline de cálculo de features."""
    logger.info("=" * 60)
    logger.info("INICIO DEL CÁLCULO DE FEATURES")
    logger.info("=" * 60)

    start_time = time.time()

    # 1. Cargar todos los fixtures
    df = fetch_all_fixtures()

    # 2. FASE 1: Calcular features para fixtures sin features
    logger.info("\n--- FASE 1: Calcular nuevas features ---")

    without_features = fetch_fixtures_without_features()

    if without_features:
        # Limitar por ejecución
        to_process = without_features[:MAX_FIXTURES_PER_RUN]
        logger.info(f"Procesando {len(to_process):,} fixtures (de {len(without_features):,} pendientes)")

        # Obtener datos completos de fixtures a procesar
        fixtures_to_calc = df[df['fixture_id'].isin(to_process)].sort_values('date')

        features_list = []
        total_ok = 0
        total_fail = 0

        for idx, (_, fixture) in enumerate(fixtures_to_calc.iterrows()):
            try:
                features = calculate_all_features(df, fixture)
                features_list.append(features)

                # Batch insert cada BATCH_SIZE
                if len(features_list) >= BATCH_SIZE:
                    ok, fail = insert_features_batch(features_list)
                    total_ok += ok
                    total_fail += fail
                    features_list = []

                    progress = (idx + 1) / len(fixtures_to_calc) * 100
                    logger.info(f"Progreso: {progress:.1f}% - OK: {total_ok:,}, Fail: {total_fail:,}")

            except Exception as e:
                logger.error(f"Error calculando fixture {fixture['fixture_id']}: {str(e)}")
                total_fail += 1

        # Insertar remanente
        if features_list:
            ok, fail = insert_features_batch(features_list)
            total_ok += ok
            total_fail += fail

        logger.info(f"FASE 1 completada: {total_ok:,} OK, {total_fail:,} fallidos")
    else:
        logger.info("No hay fixtures nuevos sin features")

    # 3. FASE 2: Actualizar targets de partidos terminados
    logger.info("\n--- FASE 2: Actualizar targets pendientes ---")

    pending_targets = fetch_fixtures_pending_targets()

    if pending_targets:
        updated = update_targets(pending_targets)
        logger.info(f"FASE 2 completada: {updated:,} targets actualizados")
    else:
        logger.info("No hay targets pendientes de actualizar")

    # Resumen
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 60)
    logger.info(f"Tiempo total: {elapsed:.2f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
