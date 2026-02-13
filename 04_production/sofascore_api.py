"""
Funciones para consultar la API de SofaScore.
"""

import time
import requests
import urllib3
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Importar config relativo al script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import SOFASCORE_BASE_URL, SOFASCORE_HEADERS, COPAS_EUROPEAS, COPA_ALIASES


def frac_to_decimal(frac_str):
    """Convierte cuota fraccionaria a decimal. '7/10' -> 1.70"""
    if not frac_str or '/' not in str(frac_str):
        return None
    try:
        num, den = str(frac_str).split('/')
        return round(1 + float(num) / float(den), 3)
    except (ValueError, ZeroDivisionError):
        return None


def _request(endpoint, retries=3):
    """Hace un GET a la API de SofaScore con reintentos."""
    url = f"{SOFASCORE_BASE_URL}{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=SOFASCORE_HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            print(f"[API] {url} -> status {resp.status_code}")
            if resp.status_code == 429:
                time.sleep(60)
                continue
            if resp.status_code == 404:
                return None
            # Log response body for debugging
            print(f"[API] Response: {resp.text[:500]}")
        except requests.RequestException as e:
            print(f"[API] {url} -> exception: {e}")
        time.sleep(2 ** attempt)
    return None


def is_european_cup(tournament_name):
    """Verifica si un torneo es una de las 18 copas europeas."""
    if not tournament_name:
        return False
    name_lower = tournament_name.lower()

    # Verificar contra lista principal
    for copa in COPAS_EUROPEAS:
        if copa.lower() in name_lower or name_lower in copa.lower():
            return True

    # Verificar aliases
    for alias in COPA_ALIASES:
        if alias.lower() in name_lower or name_lower in alias.lower():
            return True

    return False


def get_cup_name(tournament_name):
    """Devuelve el nombre normalizado de la copa, o None si no es copa."""
    if not tournament_name:
        return None
    name_lower = tournament_name.lower()

    for copa in COPAS_EUROPEAS:
        if copa.lower() in name_lower or name_lower in copa.lower():
            return copa

    for alias, copa in COPA_ALIASES.items():
        if alias.lower() in name_lower or name_lower in alias.lower():
            return copa

    return None


def get_events_by_date(date_str):
    """
    Obtiene todos los eventos de futbol para una fecha.
    date_str: 'YYYY-MM-DD'
    Retorna lista de eventos filtrados a copas europeas.
    """
    data = _request(f"/sport/football/scheduled-events/{date_str}")
    if not data or 'events' not in data:
        return []

    cup_events = []
    for event in data['events']:
        tournament = event.get('tournament', {})
        unique_tournament = tournament.get('uniqueTournament', {})
        tournament_name = unique_tournament.get('name', '') or tournament.get('name', '')

        cup_name = get_cup_name(tournament_name)
        if not cup_name:
            continue

        home = event.get('homeTeam', {})
        away = event.get('awayTeam', {})
        status = event.get('status', {})

        cup_events.append({
            'event_id': event.get('id'),
            'cup_name': cup_name,
            'cup_name_sofascore': tournament_name,
            'home_team': home.get('name', 'Unknown'),
            'away_team': away.get('name', 'Unknown'),
            'start_timestamp': event.get('startTimestamp'),
            'status_code': status.get('code'),
            'status_type': status.get('type'),
            'home_score': event.get('homeScore', {}).get('current'),
            'away_score': event.get('awayScore', {}).get('current'),
        })

    return cup_events


def get_over25_odds(event_id):
    """
    Obtiene cuotas Over/Under 2.5 de un evento.
    Retorna dict con open, current, change o None.
    """
    data = _request(f"/event/{event_id}/odds/1/all")
    if not data or 'markets' not in data:
        return None

    for market in data['markets']:
        if market.get('marketId') == 9 and str(market.get('choiceGroup', '')) == '2.5':
            result = {}
            for choice in market.get('choices', []):
                name = choice.get('name', '').lower()
                if name == 'over':
                    result['over_open'] = frac_to_decimal(choice.get('initialFractionalValue'))
                    result['over_current'] = frac_to_decimal(choice.get('fractionalValue'))
                    result['over_change'] = choice.get('change', 0)
                    result['over_winning'] = choice.get('winning')
                elif name == 'under':
                    result['under_open'] = frac_to_decimal(choice.get('initialFractionalValue'))
                    result['under_current'] = frac_to_decimal(choice.get('fractionalValue'))
                    result['under_winning'] = choice.get('winning')
            if result:
                return result

    return None


def get_event_result(event_id):
    """
    Obtiene el resultado de un evento.
    Retorna dict con score, total_goals, status, o None.
    """
    data = _request(f"/event/{event_id}")
    if not data or 'event' not in data:
        return None

    event = data['event']
    status = event.get('status', {})
    home_score = event.get('homeScore', {})
    away_score = event.get('awayScore', {})

    status_type = status.get('type', '')

    # Para partidos terminados, usar el score normal
    # Para partidos con extra time o penales, usar score de tiempo regular si existe
    h = home_score.get('current')
    a = away_score.get('current')

    # Si hay penales, el score 'current' incluye penales
    # Usar 'normaltime' si existe (excluye penales/extra time para algunos mercados)
    if home_score.get('normaltime') is not None:
        h = home_score.get('normaltime')
        a = away_score.get('normaltime')

    total = None
    if h is not None and a is not None:
        total = h + a

    return {
        'status_type': status_type,
        'status_code': status.get('code'),
        'home_score': h,
        'away_score': a,
        'total_goals': total,
        'is_finished': status_type == 'finished',
    }
