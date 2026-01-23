"""
Script de Sincronización Diaria: API-Football → Supabase
Tabla: fixtures (partidos de fútbol)

Modos de operación:
1. BACKFILL: Llenar gap histórico (oct 2025 → ene 2026)
2. DAILY: Sincronización diaria de nuevos partidos y resultados

Optimizado para Plan Free: 100 requests/día
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
import time
import logging
import os
import urllib3
from typing import List, Dict, Optional, Tuple, Any

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

FOOTBALL_API_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_KEY = os.environ.get('FOOTBALL_API_KEY', 'e09583304de2b04f4a046c31bdff0a75')

# Límites
DAILY_REQUEST_LIMIT = int(os.environ.get('API_DAILY_LIMIT', '100'))
REQUESTS_PER_MINUTE = 30
DAYS_AHEAD = 5  # Días futuros a buscar
DAYS_BACK_ODDS = 7  # Días hacia atrás para odds (límite de la API)
BATCH_SIZE = 500  # Para inserts en Supabase

# Modo de prueba
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() == 'true'

# Modo de obtención de odds en batch
FETCH_ODDS_BATCH = os.environ.get('FETCH_ODDS_BATCH', 'false').lower() == 'true'

# Ligas prioritarias (de estrategias validadas)
LIGAS_ESTRATEGIA = [
    102,   # Emperor Cup (Japan)
    344,   # Primera División (Bolivia)
    32,    # World Cup - Qualification Europe
    169,   # Super League (China)
    253,   # Major League Soccer (USA)
    71,    # Serie A (Brazil)
    241,   # Copa Colombia
    2,     # UEFA Champions League
    103,   # Eliteserien (Norway)
    848,   # UEFA Europa Conference League
    292,   # K League 1 (South-Korea)
    283,   # Liga I (Romania)
    265,   # Primera División (Chile)
    262,   # Liga MX (Mexico)
    357,   # Premier Division (Ireland)
    242,   # Liga Pro (Ecuador)
    239,   # Primera A (Colombia)
]

LIGAS_TOP = [
    39,    # Premier League (England)
    140,   # La Liga (Spain)
    135,   # Serie A (Italy)
    78,    # Bundesliga (Germany)
    61,    # Ligue 1 (France)
    94,    # Primeira Liga (Portugal)
    88,    # Eredivisie (Netherlands)
]

ALL_PRIORITY_LEAGUES = LIGAS_ESTRATEGIA + LIGAS_TOP

# ============================================
# RATE LIMITER
# ============================================

class RateLimiter:
    """Controla el rate limiting de la API."""

    def __init__(self, daily_limit: int = 100, per_minute: int = 30):
        self.daily_limit = daily_limit
        self.per_minute = per_minute
        self.requests_today = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()
        self.remaining_daily = daily_limit

    def can_make_request(self) -> bool:
        """Verifica si se puede hacer una request."""
        return self.remaining_daily > 0

    def wait_if_needed(self):
        """Espera si se excedió el límite por minuto."""
        now = time.time()
        if now - self.minute_start >= 60:
            # Nuevo minuto
            self.minute_start = now
            self.requests_this_minute = 0

        # Verificar límite por minuto desde headers de API (si está disponible)
        if hasattr(self, 'minute_remaining') and self.minute_remaining <= 0:
            logger.info("Límite por minuto (API) alcanzado. Esperando 60s...")
            time.sleep(60)
            self.minute_remaining = self.minute_limit if hasattr(self, 'minute_limit') else 10
            return

        # Verificar límite por minuto local
        if self.requests_this_minute >= self.per_minute:
            # Esperar hasta el siguiente minuto
            wait_time = 60 - (now - self.minute_start)
            if wait_time > 0:
                logger.info(f"Rate limit por minuto alcanzado. Esperando {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.minute_start = time.time()
                self.requests_this_minute = 0

    def record_request(self):
        """Registra una request realizada."""
        self.requests_today += 1
        self.requests_this_minute += 1
        self.remaining_daily -= 1

    def update_from_headers(self, headers: Dict):
        """Actualiza límites desde los headers de respuesta de la API.

        Headers de API-Sports:
        - x-ratelimit-requests-limit: Límite DIARIO (100 para free plan)
        - x-ratelimit-requests-remaining: Restantes DIARIOS
        - X-RateLimit-Limit: Límite por MINUTO (10 para free plan)
        - X-RateLimit-Remaining: Restantes por MINUTO

        IMPORTANTE: Debemos leer los headers DIARIOS (con "requests" en el nombre),
        no los de minuto.
        """
        # Headers diarios (con "requests" en el nombre) - ESTOS SON LOS CORRECTOS
        if 'x-ratelimit-requests-remaining' in headers:
            self.remaining_daily = int(headers['x-ratelimit-requests-remaining'])
        if 'x-ratelimit-requests-limit' in headers:
            self.daily_limit = int(headers['x-ratelimit-requests-limit'])

        # Headers por minuto (para logging y espera automática)
        if 'x-ratelimit-remaining' in headers:
            self.minute_remaining = int(headers['x-ratelimit-remaining'])
            if self.minute_remaining <= 2:
                logger.warning(f"Rate limit por minuto bajo: {self.minute_remaining} restantes")
        if 'x-ratelimit-limit' in headers:
            self.minute_limit = int(headers['x-ratelimit-limit'])

    def get_status(self) -> Dict:
        """Retorna el estado actual del rate limiter."""
        return {
            'requests_today': self.requests_today,
            'remaining_daily': self.remaining_daily,
            'daily_limit': self.daily_limit
        }


# ============================================
# FOOTBALL API CLIENT
# ============================================

class FootballAPIClient:
    """Cliente para la API de football."""

    def __init__(self, api_key: str, rate_limiter: RateLimiter):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.base_url = FOOTBALL_API_URL
        self.headers = {
            'x-apisports-key': api_key
        }

    def initialize_rate_limiter(self) -> bool:
        """Inicializa el rate limiter con el estado real de la API.

        Usa el endpoint /timezone que es gratuito para obtener los headers
        con el estado actual de requests disponibles.

        Returns:
            True si se pudo inicializar, False si hubo error
        """
        logger.info("Verificando estado de requests en la API...")

        try:
            response = requests.get(
                f"{self.base_url}/timezone",
                headers=self.headers,
                timeout=30,
                verify=False
            )

            if response.status_code == 200:
                # Actualizar rate limiter desde headers
                self.rate_limiter.update_from_headers(response.headers)

                # Verificar si hay error en el body (límite alcanzado)
                data = response.json()
                if data.get('errors') and 'requests' in data.get('errors', {}):
                    logger.warning(f"API bloqueada: {data['errors']['requests']}")
                    self.rate_limiter.remaining_daily = 0
                    return False

                logger.info(f"  Requests disponibles: {self.rate_limiter.remaining_daily}/{self.rate_limiter.daily_limit}")
                return True
            else:
                logger.error(f"Error verificando estado API: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error conectando a la API: {str(e)}")
            return False

    def _make_request(self, endpoint: str, params: Dict = None, retries: int = 3) -> Optional[Dict]:
        """Realiza una request a la API con reintentos."""
        if not self.rate_limiter.can_make_request():
            logger.warning("Límite diario de requests alcanzado")
            return None

        self.rate_limiter.wait_if_needed()

        url = f"{self.base_url}/{endpoint}"

        for attempt in range(retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30,
                    verify=False
                )

                self.rate_limiter.record_request()
                self.rate_limiter.update_from_headers(response.headers)

                if response.status_code == 429:
                    # Rate limited
                    logger.warning("Rate limit 429. Esperando 60s...")
                    time.sleep(60)
                    continue

                if response.status_code != 200:
                    logger.error(f"Error API {response.status_code}: {response.text[:200]}")
                    if attempt < retries - 1:
                        time.sleep(5 * (attempt + 1))  # Backoff exponencial
                        continue
                    return None

                data = response.json()

                if data.get('errors'):
                    logger.error(f"API Error: {data['errors']}")
                    return None

                return data

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout en request (intento {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
            except Exception as e:
                logger.error(f"Error en request: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))

        return None

    def get_fixtures_by_league_season(self, league_id: int, season: int) -> Optional[List[Dict]]:
        """Obtiene TODOS los fixtures de una liga/temporada."""
        logger.info(f"Obteniendo fixtures para liga {league_id}, temporada {season}")

        data = self._make_request('fixtures', {
            'league': league_id,
            'season': season
        })

        if data and 'response' in data:
            logger.info(f"  -> {len(data['response'])} fixtures obtenidos")
            return data['response']
        return None

    def get_fixtures_by_date(self, date: str) -> Optional[List[Dict]]:
        """Obtiene fixtures por fecha (YYYY-MM-DD)."""
        logger.info(f"Obteniendo fixtures para fecha {date}")

        data = self._make_request('fixtures', {
            'date': date
        })

        if data and 'response' in data:
            logger.info(f"  -> {len(data['response'])} fixtures obtenidos")
            return data['response']
        return None

    def get_fixtures_by_ids(self, fixture_ids: List[int]) -> Optional[List[Dict]]:
        """Obtiene fixtures por IDs (hasta 20, incluye stats)."""
        if len(fixture_ids) > 20:
            logger.warning("Máximo 20 IDs por request. Truncando lista.")
            fixture_ids = fixture_ids[:20]

        ids_str = '-'.join(str(id) for id in fixture_ids)
        logger.info(f"Obteniendo fixtures por IDs: {len(fixture_ids)} fixtures")

        data = self._make_request('fixtures', {
            'ids': ids_str
        })

        if data and 'response' in data:
            return data['response']
        return None

    def get_odds(self, fixture_id: int) -> Optional[Dict]:
        """Obtiene cuotas para un fixture."""
        logger.debug(f"Obteniendo odds para fixture {fixture_id}")

        data = self._make_request('odds', {
            'fixture': fixture_id
        })

        if data and 'response' in data and len(data['response']) > 0:
            return data['response'][0]
        return None

    def get_odds_by_league(self, league_id: int, season: int) -> Optional[List[Dict]]:
        """Obtiene TODAS las cuotas de una liga/temporada en una sola request."""
        logger.info(f"Obteniendo odds para liga {league_id}, temporada {season}")

        data = self._make_request('odds', {
            'league': league_id,
            'season': season
        })

        if data and 'response' in data:
            logger.info(f"  -> {len(data['response'])} fixtures con odds obtenidos")
            return data['response']
        return None

    def get_odds_by_date(self, date: str) -> Optional[List[Dict]]:
        """Obtiene TODAS las cuotas de una fecha en una sola request."""
        logger.info(f"Obteniendo odds para fecha {date}")

        data = self._make_request('odds', {
            'date': date
        })

        if data and 'response' in data:
            logger.info(f"  -> {len(data['response'])} fixtures con odds obtenidos")
            return data['response']
        return None


# ============================================
# SUPABASE CLIENT
# ============================================

class SupabaseClient:
    """Cliente para Supabase."""

    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.headers = {
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'Prefer': 'resolution=merge-duplicates'
        }

    def get_last_fixture_date(self) -> Optional[str]:
        """Obtiene la fecha del último fixture terminado."""
        url = f"{self.url}/rest/v1/fixtures"
        params = {
            'select': 'date',
            'status_short': 'eq.FT',
            'order': 'date.desc',
            'limit': 1
        }

        response = requests.get(url, headers=self.headers, params=params, verify=False)

        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]['date'][:10]  # Solo la fecha
        return None

    def get_pending_fixtures(self, before_date: str = None) -> List[Dict]:
        """Obtiene fixtures con status NS que deberían estar terminados."""
        url = f"{self.url}/rest/v1/fixtures"

        if before_date is None:
            before_date = datetime.now().strftime('%Y-%m-%d')

        params = {
            'select': 'fixture_id,date,home_team_name,away_team_name,league_id',
            'status_short': 'eq.NS',
            'date': f'lt.{before_date}',
            'order': 'date.asc',
            'limit': 1000
        }

        response = requests.get(url, headers=self.headers, params=params, verify=False)

        if response.status_code == 200:
            return response.json()
        return []

    def get_fixtures_needing_odds(self, days_back: int = 7) -> List[Dict]:
        """Obtiene fixtures futuros sin cuotas."""
        url = f"{self.url}/rest/v1/fixtures"

        today = datetime.now()
        min_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
        max_date = (today + timedelta(days=DAYS_AHEAD)).strftime('%Y-%m-%d')

        params = {
            'select': 'fixture_id,date,league_id',
            'odds_fetched': 'is.null',
            'date': f'gte.{min_date}',
            'order': 'date.asc',
            'limit': 100
        }

        response = requests.get(url, headers=self.headers, params=params, verify=False)

        if response.status_code == 200:
            fixtures = response.json()
            # Filtrar solo los de ligas prioritarias
            return [f for f in fixtures if f.get('league_id') in ALL_PRIORITY_LEAGUES]
        return []

    def get_existing_fixture_ids(self, league_id: int = None, min_date: str = None) -> set:
        """Obtiene IDs de fixtures existentes."""
        url = f"{self.url}/rest/v1/fixtures"
        params = {
            'select': 'fixture_id'
        }

        if league_id:
            params['league_id'] = f'eq.{league_id}'
        if min_date:
            params['date'] = f'gte.{min_date}'

        all_ids = set()
        offset = 0
        limit = 1000

        while True:
            params['offset'] = offset
            params['limit'] = limit

            response = requests.get(url, headers=self.headers, params=params, verify=False)

            if response.status_code != 200:
                break

            data = response.json()
            if not data:
                break

            all_ids.update(d['fixture_id'] for d in data)

            if len(data) < limit:
                break

            offset += limit

        return all_ids

    def upsert_fixtures(self, records: List[Dict]) -> Tuple[int, int]:
        """Inserta o actualiza fixtures en batch."""
        if not records:
            return 0, 0

        if DRY_RUN:
            logger.info(f"[DRY RUN] Se insertarían {len(records)} fixtures")
            return len(records), 0

        url = f"{self.url}/rest/v1/fixtures"

        # Intentar batch
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=records,
                timeout=60,
                verify=False
            )

            if response.status_code in [200, 201]:
                return len(records), 0
            else:
                logger.warning(f"Error en batch insert: {response.status_code}")
                # Fallback: uno por uno
                exitosos = 0
                fallidos = 0
                for record in records:
                    try:
                        r = requests.post(url, headers=self.headers, json=[record], timeout=30, verify=False)
                        if r.status_code in [200, 201]:
                            exitosos += 1
                        else:
                            fallidos += 1
                            logger.debug(f"Error fixture {record.get('fixture_id')}: {r.text[:100]}")
                    except Exception as e:
                        fallidos += 1
                        logger.debug(f"Error fixture {record.get('fixture_id')}: {str(e)}")
                return exitosos, fallidos

        except Exception as e:
            logger.error(f"Error en upsert: {str(e)}")
            return 0, len(records)

    def update_fixture(self, fixture_id: int, data: Dict) -> bool:
        """Actualiza un fixture específico."""
        if DRY_RUN:
            logger.info(f"[DRY RUN] Se actualizaría fixture {fixture_id}")
            return True

        url = f"{self.url}/rest/v1/fixtures"
        params = {'fixture_id': f'eq.{fixture_id}'}

        headers = {**self.headers, 'Prefer': 'return=minimal'}

        response = requests.patch(
            url,
            headers=headers,
            params=params,
            json=data,
            timeout=30,
            verify=False
        )

        return response.status_code in [200, 204]

    def update_odds_batch(self, odds_updates: List[Dict]) -> Tuple[int, int]:
        """Actualiza odds para múltiples fixtures.

        Args:
            odds_updates: Lista de dicts con {fixture_id, odds_home, odds_draw, odds_away}

        Returns:
            Tupla (exitosos, fallidos)
        """
        if not odds_updates:
            return 0, 0

        if DRY_RUN:
            logger.info(f"[DRY RUN] Se actualizarían {len(odds_updates)} fixtures con odds")
            return len(odds_updates), 0

        exitosos = 0
        fallidos = 0

        for update in odds_updates:
            fixture_id = update.pop('fixture_id')
            if self.update_fixture(fixture_id, update):
                exitosos += 1
            else:
                fallidos += 1

        return exitosos, fallidos

    def get_fixtures_without_odds(self, league_ids: List[int] = None, min_date: str = None) -> List[Dict]:
        """Obtiene fixtures sin cuotas."""
        url = f"{self.url}/rest/v1/fixtures"

        params = {
            'select': 'fixture_id,date,league_id,league_name,home_team_name,away_team_name,status_short',
            'odds_home': 'is.null',
            'order': 'date.desc',
            'limit': 1000
        }

        if min_date:
            params['date'] = f'gte.{min_date}'

        response = requests.get(url, headers=self.headers, params=params, verify=False)

        if response.status_code == 200:
            fixtures = response.json()
            # Filtrar por ligas si se especifica
            if league_ids:
                fixtures = [f for f in fixtures if f.get('league_id') in league_ids]
            return fixtures
        return []

    def get_fixtures_needing_odds_in_range(self, days_back: int = 7, days_forward: int = 14,
                                           league_ids: List[int] = None) -> Dict[str, List[Dict]]:
        """Obtiene fixtures sin odds dentro del rango donde la API tiene datos disponibles.

        Args:
            days_back: Días hacia el pasado (límite API = 7)
            days_forward: Días hacia el futuro (límite API = ~14)
            league_ids: Lista de IDs de ligas prioritarias

        Returns:
            Dict con 'finished' (terminados sin odds) y 'upcoming' (futuros sin odds)
        """
        url = f"{self.url}/rest/v1/fixtures"

        today = datetime.now()
        min_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
        max_date = (today + timedelta(days=days_forward)).strftime('%Y-%m-%d')

        # Obtener todos los fixtures sin odds en el rango
        params = {
            'select': 'fixture_id,date,league_id,league_name,home_team_name,away_team_name,status_short',
            'odds_home': 'is.null',
            'date': f'gte.{min_date}',
            'order': 'date.asc',
            'limit': 1000
        }

        response = requests.get(url, headers=self.headers, params=params, verify=False)

        result = {'finished': [], 'upcoming': []}

        if response.status_code != 200:
            return result

        fixtures = response.json()

        # Filtrar por ligas prioritarias si se especifica
        if league_ids:
            fixtures = [f for f in fixtures if f.get('league_id') in league_ids]

        # Filtrar por fecha máxima (futuros solo hasta days_forward)
        today_str = today.strftime('%Y-%m-%d')

        for f in fixtures:
            fixture_date = f.get('date', '')[:10]

            # Ignorar fixtures más allá del rango futuro
            if fixture_date > max_date:
                continue

            # Clasificar: terminados vs futuros
            status = f.get('status_short', '')
            if status in ['FT', 'AET', 'PEN']:
                result['finished'].append(f)
            elif status == 'NS' and fixture_date >= today_str:
                result['upcoming'].append(f)
            elif fixture_date < today_str:
                # Partido pasado pero no marcado como terminado (podría necesitar actualización)
                result['finished'].append(f)

        return result


# ============================================
# TRANSFORMERS
# ============================================

def transform_fixture_from_api(api_fixture: Dict) -> Dict:
    """Transforma un fixture de la API al formato de la BD."""
    fixture = api_fixture.get('fixture', {})
    league = api_fixture.get('league', {})
    teams = api_fixture.get('teams', {})
    goals = api_fixture.get('goals', {})
    score = api_fixture.get('score', {})
    venue = fixture.get('venue', {})
    status = fixture.get('status', {})

    # Determinar match_type
    status_short = status.get('short', '')
    if status_short in ['FT', 'AET', 'PEN']:
        match_type = 'Terminado'
    elif status_short == 'NS':
        match_type = 'Proximo'
    else:
        match_type = 'Otro'

    record = {
        'fixture_id': fixture.get('id'),
        'date': fixture.get('date'),
        'timestamp': fixture.get('timestamp'),
        'timezone': fixture.get('timezone'),
        'venue_id': venue.get('id'),
        'venue_name': venue.get('name'),
        'venue_city': venue.get('city'),
        'status_long': status.get('long'),
        'status_short': status_short,
        'status_elapsed': status.get('elapsed'),
        'league_id': league.get('id'),
        'league_name': league.get('name'),
        'league_country': league.get('country'),
        'league_season': league.get('season'),
        'league_round': league.get('round'),
        'home_team_id': teams.get('home', {}).get('id'),
        'home_team_name': teams.get('home', {}).get('name'),
        'away_team_id': teams.get('away', {}).get('id'),
        'away_team_name': teams.get('away', {}).get('name'),
        'goals_home': goals.get('home'),
        'goals_away': goals.get('away'),
        'score_halftime_home': score.get('halftime', {}).get('home'),
        'score_halftime_away': score.get('halftime', {}).get('away'),
        'score_fulltime_home': score.get('fulltime', {}).get('home'),
        'score_fulltime_away': score.get('fulltime', {}).get('away'),
        'match_type': match_type,
        'updated_at': datetime.now().isoformat()
    }

    # Agregar estadísticas si vienen
    statistics = api_fixture.get('statistics', [])
    if statistics and len(statistics) >= 2:
        record.update(extract_statistics(statistics, teams))

    return record


def extract_statistics(statistics: List[Dict], teams: Dict) -> Dict:
    """Extrae estadísticas del fixture."""
    home_team_id = teams.get('home', {}).get('id')

    stats = {
        'stats_fetched': 1
    }

    for team_stats in statistics:
        team_id = team_stats.get('team', {}).get('id')
        prefix = 'home_' if team_id == home_team_id else 'away_'

        for stat in team_stats.get('statistics', []):
            stat_type = stat.get('type', '')
            value = stat.get('value')

            if stat_type == 'Shots on Goal':
                stats[f'{prefix}shots_on_goal'] = value
            elif stat_type == 'Total Shots':
                stats[f'{prefix}total_shots'] = value
            elif stat_type == 'Ball Possession':
                if value and isinstance(value, str):
                    value = value.replace('%', '')
                    try:
                        stats[f'{prefix}ball_possession'] = float(value)
                    except ValueError:
                        pass

    return stats


def transform_odds_from_api(odds_data: Dict) -> Dict:
    """Transforma odds de la API al formato de la BD."""
    result = {'odds_fetched': 1}

    bookmakers = odds_data.get('bookmakers', [])
    if not bookmakers:
        return result

    # Buscar bookmaker preferido o usar el primero
    bookmaker = bookmakers[0]

    for bet in bookmaker.get('bets', []):
        if bet.get('name') == 'Match Winner':
            for value in bet.get('values', []):
                odd_value = value.get('odd')
                if odd_value:
                    try:
                        odd_float = float(odd_value)
                        if value.get('value') == 'Home':
                            result['odds_home'] = odd_float
                        elif value.get('value') == 'Draw':
                            result['odds_draw'] = odd_float
                        elif value.get('value') == 'Away':
                            result['odds_away'] = odd_float
                    except ValueError:
                        pass
            break

    return result


# ============================================
# SYNC ORCHESTRATOR
# ============================================

class SyncOrchestrator:
    """Orquesta la sincronización."""

    def __init__(self, api_client: FootballAPIClient, db_client: SupabaseClient, rate_limiter: RateLimiter):
        self.api = api_client
        self.db = db_client
        self.rate_limiter = rate_limiter
        self.stats = {
            'fixtures_inserted': 0,
            'fixtures_updated': 0,
            'fixtures_failed': 0,
            'odds_fetched': 0,
            'requests_used': 0
        }

    def detect_sync_mode(self) -> str:
        """Detecta si estamos en modo backfill o daily."""
        last_date = self.db.get_last_fixture_date()

        if last_date:
            last_dt = datetime.strptime(last_date, '%Y-%m-%d')
            today = datetime.now()
            gap_days = (today - last_dt).days

            logger.info(f"Ultimo fixture terminado: {last_date} ({gap_days} dias atras)")

            if gap_days > 14:
                return 'backfill'
        else:
            return 'backfill'

        return 'daily'

    def backfill_leagues(self, leagues: List[int], season: int = 2024) -> Dict:
        """Modo backfill: obtiene todos los fixtures de las ligas prioritarias."""
        logger.info(f"=== MODO BACKFILL: {len(leagues)} ligas, temporada {season} ===")

        total_inserted = 0
        total_updated = 0
        total_failed = 0

        for league_id in leagues:
            if not self.rate_limiter.can_make_request():
                logger.warning("Limite diario alcanzado. Deteniendo backfill.")
                break

            # Obtener IDs existentes para esta liga
            existing_ids = self.db.get_existing_fixture_ids(league_id=league_id)
            logger.info(f"Liga {league_id}: {len(existing_ids)} fixtures existentes")

            # Obtener fixtures de la API
            fixtures = self.api.get_fixtures_by_league_season(league_id, season)

            if not fixtures:
                continue

            # Filtrar solo fixtures nuevos o actualizados
            records_to_upsert = []
            for api_fixture in fixtures:
                fixture_id = api_fixture.get('fixture', {}).get('id')
                record = transform_fixture_from_api(api_fixture)

                # Solo insertar si es nuevo o si tiene resultados actualizados
                if fixture_id not in existing_ids or record.get('status_short') != 'NS':
                    records_to_upsert.append(record)

            if records_to_upsert:
                logger.info(f"  -> Insertando/actualizando {len(records_to_upsert)} fixtures")

                # Insertar en batches
                for i in range(0, len(records_to_upsert), BATCH_SIZE):
                    batch = records_to_upsert[i:i + BATCH_SIZE]
                    ok, failed = self.db.upsert_fixtures(batch)
                    total_inserted += ok
                    total_failed += failed

            # Pausa entre ligas
            time.sleep(1)

        self.stats['fixtures_inserted'] += total_inserted
        self.stats['fixtures_failed'] += total_failed

        return {
            'inserted': total_inserted,
            'failed': total_failed,
            'leagues_processed': len(leagues)
        }

    def sync_upcoming_fixtures(self, days_ahead: int = DAYS_AHEAD) -> Dict:
        """Obtiene fixtures de los próximos días."""
        logger.info(f"=== SYNC UPCOMING: proximos {days_ahead} dias ===")

        total_inserted = 0
        total_failed = 0

        today = datetime.now()

        for i in range(days_ahead):
            if not self.rate_limiter.can_make_request():
                break

            date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
            fixtures = self.api.get_fixtures_by_date(date)

            if not fixtures:
                continue

            # Filtrar solo ligas prioritarias
            records = []
            for api_fixture in fixtures:
                league_id = api_fixture.get('league', {}).get('id')
                if league_id in ALL_PRIORITY_LEAGUES:
                    records.append(transform_fixture_from_api(api_fixture))

            if records:
                logger.info(f"  {date}: {len(records)} fixtures de ligas prioritarias")
                ok, failed = self.db.upsert_fixtures(records)
                total_inserted += ok
                total_failed += failed

            time.sleep(0.5)

        self.stats['fixtures_inserted'] += total_inserted
        self.stats['fixtures_failed'] += total_failed

        return {
            'inserted': total_inserted,
            'failed': total_failed
        }

    def update_pending_results(self) -> Dict:
        """Actualiza fixtures NS que deberían estar terminados."""
        logger.info("=== UPDATE PENDING RESULTS ===")

        pending = self.db.get_pending_fixtures()

        if not pending:
            logger.info("No hay fixtures pendientes de actualizar")
            return {'updated': 0}

        # Filtrar solo ligas prioritarias
        pending = [f for f in pending if f.get('league_id') in ALL_PRIORITY_LEAGUES]

        logger.info(f"Fixtures pendientes en ligas prioritarias: {len(pending)}")

        total_updated = 0
        total_failed = 0

        # Procesar en grupos de 20 (límite de la API para ?ids=)
        fixture_ids = [f['fixture_id'] for f in pending]

        for i in range(0, len(fixture_ids), 20):
            if not self.rate_limiter.can_make_request():
                break

            batch_ids = fixture_ids[i:i + 20]
            fixtures = self.api.get_fixtures_by_ids(batch_ids)

            if not fixtures:
                continue

            records = [transform_fixture_from_api(f) for f in fixtures]
            ok, failed = self.db.upsert_fixtures(records)
            total_updated += ok
            total_failed += failed

            time.sleep(0.5)

        self.stats['fixtures_updated'] += total_updated
        self.stats['fixtures_failed'] += total_failed

        return {
            'updated': total_updated,
            'failed': total_failed
        }

    def fetch_odds_for_upcoming(self, max_requests: int = 20) -> Dict:
        """Obtiene odds para fixtures próximos."""
        logger.info("=== FETCH ODDS ===")

        fixtures = self.db.get_fixtures_needing_odds()

        if not fixtures:
            logger.info("No hay fixtures que necesiten odds")
            return {'fetched': 0}

        logger.info(f"Fixtures sin odds: {len(fixtures)}")

        total_fetched = 0
        requests_made = 0

        for fixture in fixtures:
            if requests_made >= max_requests:
                break
            if not self.rate_limiter.can_make_request():
                break

            fixture_id = fixture['fixture_id']
            odds_data = self.api.get_odds(fixture_id)
            requests_made += 1

            if odds_data:
                odds_record = transform_odds_from_api(odds_data)
                if self.db.update_fixture(fixture_id, odds_record):
                    total_fetched += 1
            else:
                # Marcar como intentado para no reintentar
                self.db.update_fixture(fixture_id, {'odds_fetched': 0})

            time.sleep(0.3)

        self.stats['odds_fetched'] += total_fetched

        return {
            'fetched': total_fetched,
            'requests': requests_made
        }

    def fetch_odds_by_league_batch(self, leagues: List[int], season: int = 2024) -> Dict:
        """Obtiene odds en batch por liga (1 request = todas las odds de una liga/temporada)."""
        logger.info(f"=== FETCH ODDS BY LEAGUE (BATCH): {len(leagues)} ligas, temporada {season} ===")

        total_fixtures_updated = 0
        total_odds_found = 0
        leagues_processed = 0

        for league_id in leagues:
            if not self.rate_limiter.can_make_request():
                logger.warning("Limite diario alcanzado. Deteniendo fetch de odds.")
                break

            odds_list = self.api.get_odds_by_league(league_id, season)
            leagues_processed += 1

            if not odds_list:
                continue

            total_odds_found += len(odds_list)

            # Transformar y preparar updates
            odds_updates = []
            for odds_item in odds_list:
                fixture_data = odds_item.get('fixture', {})
                fixture_id = fixture_data.get('id')

                if not fixture_id:
                    continue

                odds_record = transform_odds_from_api(odds_item)
                if odds_record.get('odds_home'):  # Solo si tiene odds válidas
                    odds_record['fixture_id'] = fixture_id
                    odds_updates.append(odds_record)

            if odds_updates:
                ok, failed = self.db.update_odds_batch(odds_updates)
                total_fixtures_updated += ok
                logger.info(f"  Liga {league_id}: {ok} fixtures actualizados con odds")

            time.sleep(0.5)

        self.stats['odds_fetched'] += total_fixtures_updated

        return {
            'leagues_processed': leagues_processed,
            'odds_found': total_odds_found,
            'fixtures_updated': total_fixtures_updated
        }

    def fetch_odds_automatic(self) -> Dict:
        """Obtiene odds automáticamente para partidos en el rango válido de la API.

        Estrategia:
        1. Partidos FUTUROS (próximos 14 días): Obtener odds y guardarlas
           - Cuando el partido termine, ya tendrá las odds guardadas
        2. Partidos TERMINADOS (últimos 7 días) sin odds: Obtener odds (fase de transición)
           - Eventualmente ya no será necesario porque todos tendrán odds desde futuros

        Usa el endpoint por FECHA que es más eficiente (1 request = todas las odds del día)
        """
        logger.info("=== FETCH ODDS AUTOMATIC ===")

        # Obtener fixtures que necesitan odds
        fixtures_data = self.db.get_fixtures_needing_odds_in_range(
            days_back=DAYS_BACK_ODDS,
            days_forward=DAYS_AHEAD,
            league_ids=ALL_PRIORITY_LEAGUES
        )

        finished_count = len(fixtures_data['finished'])
        upcoming_count = len(fixtures_data['upcoming'])

        logger.info(f"  Partidos terminados sin odds (ultimos {DAYS_BACK_ODDS} dias): {finished_count}")
        logger.info(f"  Partidos futuros sin odds (proximos {DAYS_AHEAD} dias): {upcoming_count}")

        if finished_count == 0 and upcoming_count == 0:
            logger.info("  No hay fixtures que necesiten odds")
            return {'finished_updated': 0, 'upcoming_updated': 0, 'dates_processed': 0}

        # Obtener las fechas únicas que necesitan odds
        dates_needed = set()
        for f in fixtures_data['finished'] + fixtures_data['upcoming']:
            date = f.get('date', '')[:10]
            if date:
                dates_needed.add(date)

        dates_needed = sorted(dates_needed)
        logger.info(f"  Fechas a consultar: {len(dates_needed)}")

        # Crear set de fixture_ids que necesitan odds para filtrar después
        fixture_ids_needed = set()
        for f in fixtures_data['finished'] + fixtures_data['upcoming']:
            fixture_ids_needed.add(f['fixture_id'])

        total_finished_updated = 0
        total_upcoming_updated = 0
        dates_processed = 0

        for date in dates_needed:
            if not self.rate_limiter.can_make_request():
                logger.warning("Limite diario alcanzado. Deteniendo fetch de odds.")
                break

            odds_list = self.api.get_odds_by_date(date)
            dates_processed += 1

            if not odds_list:
                continue

            # Transformar y preparar updates (solo para fixtures que necesitamos)
            odds_updates = []
            for odds_item in odds_list:
                fixture_data = odds_item.get('fixture', {})
                fixture_id = fixture_data.get('id')

                # Solo procesar si está en nuestra lista de necesitados
                if not fixture_id or fixture_id not in fixture_ids_needed:
                    continue

                odds_record = transform_odds_from_api(odds_item)
                if odds_record.get('odds_home'):  # Solo si tiene odds válidas
                    odds_record['fixture_id'] = fixture_id
                    odds_updates.append(odds_record)

            if odds_updates:
                ok, failed = self.db.update_odds_batch(odds_updates)

                # Contar terminados vs futuros
                for update in odds_updates:
                    fid = update.get('fixture_id')
                    # Buscar en qué categoría está
                    is_finished = any(f['fixture_id'] == fid for f in fixtures_data['finished'])
                    if is_finished:
                        total_finished_updated += 1
                    else:
                        total_upcoming_updated += 1

                logger.info(f"  Fecha {date}: {ok} fixtures actualizados con odds")

            time.sleep(0.5)

        self.stats['odds_fetched'] += total_finished_updated + total_upcoming_updated

        return {
            'finished_updated': total_finished_updated,
            'upcoming_updated': total_upcoming_updated,
            'dates_processed': dates_processed,
            'dates_total': len(dates_needed)
        }

    def fetch_odds_by_date_batch(self, days_back: int = 7, days_forward: int = 7) -> Dict:
        """Obtiene odds en batch por fecha (1 request = todas las odds de un día)."""
        logger.info(f"=== FETCH ODDS BY DATE (BATCH): -{days_back} dias a +{days_forward} dias ===")

        today = datetime.now()
        total_fixtures_updated = 0
        total_odds_found = 0
        dates_processed = 0

        # Generar lista de fechas
        dates = []
        for i in range(-days_back, days_forward + 1):
            date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
            dates.append(date)

        for date in dates:
            if not self.rate_limiter.can_make_request():
                logger.warning("Limite diario alcanzado. Deteniendo fetch de odds.")
                break

            odds_list = self.api.get_odds_by_date(date)
            dates_processed += 1

            if not odds_list:
                continue

            total_odds_found += len(odds_list)

            # Transformar y preparar updates
            odds_updates = []
            for odds_item in odds_list:
                fixture_data = odds_item.get('fixture', {})
                fixture_id = fixture_data.get('id')

                if not fixture_id:
                    continue

                odds_record = transform_odds_from_api(odds_item)
                if odds_record.get('odds_home'):  # Solo si tiene odds válidas
                    odds_record['fixture_id'] = fixture_id
                    odds_updates.append(odds_record)

            if odds_updates:
                ok, failed = self.db.update_odds_batch(odds_updates)
                total_fixtures_updated += ok
                logger.info(f"  Fecha {date}: {ok} fixtures actualizados con odds")

            time.sleep(0.5)

        self.stats['odds_fetched'] += total_fixtures_updated

        return {
            'dates_processed': dates_processed,
            'odds_found': total_odds_found,
            'fixtures_updated': total_fixtures_updated
        }

    def run(self, fetch_odds_batch: bool = False) -> Dict:
        """Ejecuta la sincronización completa.

        Args:
            fetch_odds_batch: Si es True, obtiene odds en batch para todas las ligas
        """
        logger.info("=" * 60)
        logger.info("INICIANDO SINCRONIZACION")
        logger.info("=" * 60)

        if DRY_RUN:
            logger.info("*** MODO DRY RUN - No se escribira en la BD ***")

        # Verificar API key
        if not self.api.api_key:
            logger.error("FOOTBALL_API_KEY no configurada")
            return {'error': 'No API key'}

        # IMPORTANTE: Inicializar rate limiter con estado real de la API
        if not self.api.initialize_rate_limiter():
            logger.error("No se pudo verificar estado de la API. Posiblemente límite alcanzado.")
            return {
                'error': 'API limit reached or connection error',
                'rate_limiter': self.rate_limiter.get_status()
            }

        # Verificar si hay requests disponibles
        if not self.rate_limiter.can_make_request():
            logger.warning("No hay requests disponibles. Límite diario alcanzado.")
            return {
                'error': 'Daily limit reached',
                'rate_limiter': self.rate_limiter.get_status()
            }

        # Detectar modo
        mode = self.detect_sync_mode()
        logger.info(f"Modo detectado: {mode}")

        results = {'mode': mode}

        if mode == 'backfill':
            # Fase 1: Backfill de ligas prioritarias
            logger.info("\n--- FASE 1: Backfill ligas estrategia (temporada 2024) ---")
            results['backfill_2024'] = self.backfill_leagues(ALL_PRIORITY_LEAGUES, season=2024)

            # Si sobran requests, también temporada 2025
            if self.rate_limiter.can_make_request():
                logger.info("\n--- FASE 1b: Backfill ligas estrategia (temporada 2025) ---")
                results['backfill_2025'] = self.backfill_leagues(ALL_PRIORITY_LEAGUES, season=2025)

        else:
            # Modo daily

            # Fase 1: Actualizar pending
            logger.info("\n--- FASE 1: Actualizar pendientes ---")
            results['pending'] = self.update_pending_results()

            # Fase 2: Upcoming fixtures
            if self.rate_limiter.can_make_request():
                logger.info("\n--- FASE 2: Proximos partidos ---")
                results['upcoming'] = self.sync_upcoming_fixtures()

            # Fase 3: Odds automático (terminados últimos 7 días + futuros próximos 14 días)
            if self.rate_limiter.can_make_request():
                logger.info("\n--- FASE 3: Odds automatico ---")
                results['odds_automatic'] = self.fetch_odds_automatic()

        # Fase adicional: Odds en batch (si se solicita)
        if fetch_odds_batch and self.rate_limiter.can_make_request():
            logger.info("\n--- FASE ODDS BATCH ---")

            # Obtener odds por liga para temporadas 2024 y 2025
            if self.rate_limiter.can_make_request():
                logger.info("\n--- Odds por liga (temporada 2024) ---")
                results['odds_batch_2024'] = self.fetch_odds_by_league_batch(ALL_PRIORITY_LEAGUES, season=2024)

            if self.rate_limiter.can_make_request():
                logger.info("\n--- Odds por liga (temporada 2025) ---")
                results['odds_batch_2025'] = self.fetch_odds_by_league_batch(ALL_PRIORITY_LEAGUES, season=2025)

        # Resumen
        self.stats['requests_used'] = self.rate_limiter.requests_today
        results['stats'] = self.stats
        results['rate_limiter'] = self.rate_limiter.get_status()

        return results


# ============================================
# MAIN
# ============================================

def main():
    """Función principal."""
    logger.info("=" * 60)
    logger.info("SYNC DAILY - Football API to Supabase")
    logger.info("=" * 60)

    if FETCH_ODDS_BATCH:
        logger.info("*** MODO FETCH_ODDS_BATCH ACTIVADO ***")

    # Inicializar componentes
    rate_limiter = RateLimiter(daily_limit=DAILY_REQUEST_LIMIT)
    api_client = FootballAPIClient(FOOTBALL_API_KEY, rate_limiter)
    db_client = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
    orchestrator = SyncOrchestrator(api_client, db_client, rate_limiter)

    # Ejecutar sincronización
    start_time = time.time()
    results = orchestrator.run(fetch_odds_batch=FETCH_ODDS_BATCH)
    elapsed = time.time() - start_time

    # Mostrar resumen
    logger.info("\n" + "=" * 60)
    logger.info("SINCRONIZACION COMPLETADA")
    logger.info("=" * 60)
    logger.info(f"Modo: {results.get('mode', 'unknown')}")
    logger.info(f"Tiempo: {elapsed:.2f}s")

    stats = results.get('stats', {})
    logger.info(f"Fixtures insertados: {stats.get('fixtures_inserted', 0)}")
    logger.info(f"Fixtures actualizados: {stats.get('fixtures_updated', 0)}")
    logger.info(f"Fixtures fallidos: {stats.get('fixtures_failed', 0)}")
    logger.info(f"Odds obtenidas: {stats.get('odds_fetched', 0)}")

    rate_status = results.get('rate_limiter', {})
    logger.info(f"Requests usadas: {rate_status.get('requests_today', 0)}/{rate_status.get('daily_limit', 100)}")

    # Guardar resultados en JSON
    results_file = 'sync_results.json'
    with open(results_file, 'w') as f:
        # Convertir a serializable
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Resultados guardados en {results_file}")

    return results


if __name__ == "__main__":
    try:
        results = main()

        # Exit code basado en resultados
        stats = results.get('stats', {})
        if stats.get('fixtures_failed', 0) > stats.get('fixtures_inserted', 0):
            exit(1)  # Más fallidos que exitosos
        exit(0)

    except Exception as e:
        logger.error(f"Error fatal: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
