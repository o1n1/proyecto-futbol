"""
Importador de JSONs de baseball a mlb.db

Importa los JSONs raw scrapeados por GitHub Actions a una base de datos SQLite.
Adaptado de 04_sofascore_complete/scripts/02_import_to_db.py para baseball.

Diferencias con fútbol:
    - Sin tabla de incidents (SofaScore no tiene incidents para baseball)
    - Scores por inning en vez de halftime
    - Metadata del evento viene directamente del JSON (tournament, home, away)
    - Solo 1 mercado de cuotas (moneyline)
    - 77 stat keys de jugadores (batting/pitching/fielding)

Uso:
    python 02_import_to_db.py --source raw              # Importar JSONs
    python 02_import_to_db.py --source raw --year 2024  # Año específico
    python 02_import_to_db.py --stats                   # Ver estadísticas
    python 02_import_to_db.py --source raw --force      # Re-importar
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# RUTAS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
MODULE_DIR = SCRIPT_DIR.parent
DATA_DIR = MODULE_DIR / "data"
DB_FILE = DATA_DIR / "mlb.db"
RAW_DIR = DATA_DIR / "raw"

# ============================================================================
# CONVERSIÓN DE CUOTAS
# ============================================================================

def frac_to_decimal(frac: str) -> Optional[float]:
    """Convierte cuota fraccionaria a decimal. '7/10' -> 1.70"""
    if not frac or '/' not in str(frac):
        return None
    try:
        num, den = str(frac).split('/')
        return round(1 + float(num) / float(den), 4)
    except (ValueError, ZeroDivisionError):
        return None

# ============================================================================
# ESQUEMA DE BASE DE DATOS (6 tablas - sin incidents)
# ============================================================================

SCHEMA_SQL = """
-- Tabla principal: metadata del evento + raw JSON backup
CREATE TABLE IF NOT EXISTS events (
    event_id        INTEGER PRIMARY KEY,
    date            TEXT NOT NULL,
    time            TEXT,
    timestamp       INTEGER,
    sport           TEXT DEFAULT 'baseball',
    country         TEXT,
    league_name     TEXT,
    league_id       INTEGER,
    season_name     TEXT,
    home_team       TEXT,
    home_team_id    INTEGER,
    away_team       TEXT,
    away_team_id    INTEGER,
    -- Scores
    home_score      INTEGER,
    away_score      INTEGER,
    -- Scores por inning (JSON array)
    home_innings    TEXT,
    away_innings    TEXT,
    num_innings     INTEGER,
    status          TEXT,
    winner_code     INTEGER,
    slug            TEXT,
    venue_name      TEXT,
    venue_city      TEXT,
    attendance      INTEGER,
    -- Scraping metadata
    scraped_at      TEXT,
    has_odds        INTEGER DEFAULT 0,
    has_statistics  INTEGER DEFAULT 0,
    has_lineups     INTEGER DEFAULT 0,
    -- RAW JSON BACKUP
    raw_odds        TEXT,
    raw_statistics  TEXT,
    raw_lineups     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);
CREATE INDEX IF NOT EXISTS idx_events_league ON events(league_name);
CREATE INDEX IF NOT EXISTS idx_events_country ON events(country);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_teams ON events(home_team_id, away_team_id);

-- Cuotas: una fila por choice (para baseball: solo moneyline, 2 choices)
CREATE TABLE IF NOT EXISTS odds (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id                INTEGER NOT NULL,
    market_id               INTEGER NOT NULL,
    market_name             TEXT NOT NULL,
    choice_group            TEXT,
    market_source_id        INTEGER,
    structure_type          INTEGER,
    is_live                 INTEGER,
    suspended               INTEGER,
    market_db_id            INTEGER,
    market_fid              INTEGER,
    market_group            TEXT,
    market_period           TEXT,
    choice_name             TEXT NOT NULL,
    odds_open               REAL,
    odds_close              REAL,
    odds_open_fractional    TEXT,
    odds_close_fractional   TEXT,
    winning                 INTEGER,
    change                  INTEGER,
    source_id               INTEGER,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
CREATE INDEX IF NOT EXISTS idx_odds_event ON odds(event_id);
CREATE INDEX IF NOT EXISTS idx_odds_market ON odds(market_id, choice_group);

-- Estadísticas de equipo: una fila por período + stat
CREATE TABLE IF NOT EXISTS statistics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        INTEGER NOT NULL,
    period          TEXT NOT NULL,
    group_name      TEXT NOT NULL,
    stat_name       TEXT NOT NULL,
    stat_key        TEXT NOT NULL,
    home_value      REAL,
    away_value      REAL,
    home_total      REAL,
    away_total      REAL,
    home_display    TEXT,
    away_display    TEXT,
    compare_code    INTEGER,
    statistics_type TEXT,
    value_type      TEXT,
    render_type     INTEGER,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
CREATE INDEX IF NOT EXISTS idx_stats_event ON statistics(event_id);
CREATE INDEX IF NOT EXISTS idx_stats_event_period ON statistics(event_id, period);
CREATE INDEX IF NOT EXISTS idx_stats_key ON statistics(stat_key);

-- Alineaciones: una fila por jugador por partido
CREATE TABLE IF NOT EXISTS lineups (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id            INTEGER NOT NULL,
    side                TEXT NOT NULL,
    formation           TEXT,
    confirmed           INTEGER,
    player_id           INTEGER NOT NULL,
    player_name         TEXT,
    player_first_name   TEXT,
    player_last_name    TEXT,
    player_short_name   TEXT,
    player_slug         TEXT,
    player_position     TEXT,
    player_height       INTEGER,
    player_country_name TEXT,
    player_country_alpha2 TEXT,
    player_dob_ts       INTEGER,
    player_gender       TEXT,
    player_sofascore_id INTEGER,
    market_value        INTEGER,
    market_value_currency TEXT,
    user_count          INTEGER,
    shirt_number        INTEGER,
    jersey_number       TEXT,
    position_in_match   TEXT,
    is_substitute       INTEGER NOT NULL,
    is_captain          INTEGER DEFAULT 0,
    team_id             INTEGER,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
CREATE INDEX IF NOT EXISTS idx_lineups_event ON lineups(event_id);
CREATE INDEX IF NOT EXISTS idx_lineups_player ON lineups(player_id);
CREATE INDEX IF NOT EXISTS idx_lineups_event_side ON lineups(event_id, side);

-- Stats individuales por jugador: EAV (77 keys batting/pitching/fielding)
CREATE TABLE IF NOT EXISTS player_match_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        INTEGER NOT NULL,
    player_id       INTEGER NOT NULL,
    stat_key        TEXT NOT NULL,
    stat_value      REAL,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
CREATE INDEX IF NOT EXISTS idx_pms_event_player ON player_match_stats(event_id, player_id);
CREATE INDEX IF NOT EXISTS idx_pms_stat ON player_match_stats(stat_key);

-- Jugadores lesionados/suspendidos
CREATE TABLE IF NOT EXISTS missing_players (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id            INTEGER NOT NULL,
    side                TEXT NOT NULL,
    player_id           INTEGER,
    player_name         TEXT,
    player_short_name   TEXT,
    player_position     TEXT,
    player_height       INTEGER,
    player_country      TEXT,
    player_dob_ts       INTEGER,
    type                TEXT,
    reason              INTEGER,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
CREATE INDEX IF NOT EXISTS idx_missing_event ON missing_players(event_id);
"""


def create_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    logger.info("Esquema creado/verificado")


# ============================================================================
# FUNCIONES DE PARSEO (reutilizadas de fútbol, sin incidents)
# ============================================================================

def parse_odds(event_id: int, odds_data: dict) -> List[tuple]:
    rows = []
    if not odds_data or 'markets' not in odds_data:
        return rows

    for market in odds_data['markets']:
        market_id = market.get('marketId')
        market_name = market.get('marketName', '')
        choice_group = market.get('choiceGroup')
        market_source_id = market.get('sourceId')
        structure_type = market.get('structureType')
        is_live = 1 if market.get('isLive') else 0
        suspended = 1 if market.get('suspended') else 0
        market_db_id = market.get('id')
        market_fid = market.get('fid')
        market_group = market.get('marketGroup')
        market_period = market.get('marketPeriod')

        for choice in market.get('choices', []):
            choice_name = choice.get('name', '')
            odds_open_frac = choice.get('initialFractionalValue')
            odds_close_frac = choice.get('fractionalValue')
            odds_open = frac_to_decimal(odds_open_frac)
            odds_close = frac_to_decimal(odds_close_frac)
            winning_raw = choice.get('winning')
            winning = 1 if winning_raw is True else (0 if winning_raw is False else None)
            change = choice.get('change')
            source_id = choice.get('sourceId')

            rows.append((
                event_id, market_id, market_name, choice_group,
                market_source_id, structure_type, is_live, suspended,
                market_db_id, market_fid, market_group, market_period,
                choice_name, odds_open, odds_close,
                odds_open_frac, odds_close_frac,
                winning, change, source_id
            ))

    return rows


def parse_statistics(event_id: int, stats_data: dict) -> List[tuple]:
    rows = []
    if not stats_data or 'statistics' not in stats_data:
        return rows

    for period_data in stats_data['statistics']:
        period = period_data.get('period', 'ALL')
        for group in period_data.get('groups', []):
            group_name = group.get('groupName', '')
            for item in group.get('statisticsItems', []):
                stat_name = item.get('name', '')
                stat_key = item.get('key', '')
                home_value = item.get('homeValue')
                away_value = item.get('awayValue')
                home_total = item.get('homeTotal')
                away_total = item.get('awayTotal')
                home_display = item.get('home')
                away_display = item.get('away')
                compare_code = item.get('compareCode')
                statistics_type = item.get('statisticsType')
                value_type = item.get('valueType')
                render_type = item.get('renderType')

                rows.append((
                    event_id, period, group_name, stat_name, stat_key,
                    home_value, away_value, home_total, away_total,
                    home_display, away_display,
                    compare_code, statistics_type, value_type, render_type
                ))

    return rows


def parse_lineups(event_id: int, lineups_data: dict) -> Tuple[List[tuple], List[tuple], List[tuple]]:
    lineup_rows = []
    pstat_rows = []
    missing_rows = []

    if not lineups_data:
        return lineup_rows, pstat_rows, missing_rows

    confirmed = 1 if lineups_data.get('confirmed') else 0

    for side in ['home', 'away']:
        side_data = lineups_data.get(side)
        if not side_data or not isinstance(side_data, dict):
            continue

        formation = side_data.get('formation')

        for player_entry in side_data.get('players', []):
            player = player_entry.get('player', {})
            player_id = player.get('id', 0)
            country = player.get('country', {})
            market_raw = player.get('proposedMarketValueRaw', {})

            lineup_rows.append((
                event_id, side, formation, confirmed,
                player_id,
                player.get('name'),
                player.get('firstName'),
                player.get('lastName'),
                player.get('shortName'),
                player.get('slug'),
                player.get('position'),
                player.get('height'),
                country.get('name') if isinstance(country, dict) else None,
                country.get('alpha2') if isinstance(country, dict) else None,
                player.get('dateOfBirthTimestamp'),
                player.get('gender'),
                player.get('sofascoreId'),
                market_raw.get('value') if isinstance(market_raw, dict) else None,
                market_raw.get('currency') if isinstance(market_raw, dict) else (
                    player.get('marketValueCurrency')),
                player.get('userCount'),
                player_entry.get('shirtNumber'),
                player_entry.get('jerseyNumber'),
                player_entry.get('position'),
                1 if player_entry.get('substitute') else 0,
                1 if player_entry.get('captain') else 0,
                player_entry.get('teamId'),
            ))

            # Player match statistics (EAV) - 77 keys batting/pitching/fielding
            stats = player_entry.get('statistics', {})
            if stats and isinstance(stats, dict):
                for key, value in stats.items():
                    if key == 'statisticsType':
                        continue
                    if key == 'ratingVersions':
                        if isinstance(value, dict):
                            for rk, rv in value.items():
                                if isinstance(rv, (int, float)):
                                    pstat_rows.append((event_id, player_id, f'rating_{rk}', float(rv)))
                        continue
                    if isinstance(value, (int, float)):
                        pstat_rows.append((event_id, player_id, key, float(value)))

        # Missing players
        for mp in side_data.get('missingPlayers', []):
            mp_player = mp.get('player', {})
            mp_country = mp_player.get('country', {})
            missing_rows.append((
                event_id, side,
                mp_player.get('id'),
                mp_player.get('name'),
                mp_player.get('shortName'),
                mp_player.get('position'),
                mp_player.get('height'),
                mp_country.get('name') if isinstance(mp_country, dict) else None,
                mp_player.get('dateOfBirthTimestamp'),
                mp.get('type'),
                mp.get('reason'),
            ))

    return lineup_rows, pstat_rows, missing_rows


# ============================================================================
# PARSEO DE SCORES POR INNING
# ============================================================================

def parse_inning_scores(score_data: dict) -> Tuple[Optional[str], Optional[int]]:
    """Extrae scores por inning del objeto homeScore/awayScore.

    Returns: (innings_json, num_innings)
    innings_json es un JSON array: [0, 1, 0, 3, ...]
    """
    if not score_data or not isinstance(score_data, dict):
        return None, None

    innings = []
    i = 1
    while True:
        key = f'inning{i}' if i > 0 else 'period1'
        alt_key = f'period{i}'
        val = score_data.get(key)
        if val is None:
            val = score_data.get(alt_key)
        if val is None:
            break
        innings.append(val)
        i += 1

    if not innings:
        return None, None

    return json.dumps(innings), len(innings)


# ============================================================================
# IMPORTACIÓN PRINCIPAL
# ============================================================================

def import_json_file(filepath: Path, conn: sqlite3.Connection,
                     force: bool = False) -> dict:
    result = {'odds': 0, 'stats': 0, 'lineups': 0, 'player_stats': 0,
              'missing': 0, 'skipped': False, 'error': None}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        result['error'] = str(e)
        return result

    event_id = data.get('event_id')
    if not event_id:
        result['error'] = 'No event_id in JSON'
        return result

    # Check si ya existe
    if not force:
        existing = conn.execute(
            "SELECT event_id FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if existing:
            result['skipped'] = True
            return result

    # Extraer metadata del JSON (viene directamente del scraper MLB)
    tournament = data.get('tournament', '')
    home = data.get('home', '')
    away = data.get('away', '')
    sport = data.get('sport', 'baseball')

    # Extraer fecha del filename: YYYY-MM-DD_eventid.json
    date_str = None
    fname = filepath.stem
    parts = fname.split('_')
    if len(parts) >= 2 and len(parts[0]) == 10:
        date_str = parts[0]

    if not date_str:
        date_str = data.get('date', '')

    if not date_str:
        result['error'] = f'No date found for event {event_id}'
        return result

    # Extraer scores por inning del evento detail si disponible
    # Los JSONs del scraper guardan los scores en el raw
    event_detail = data.get('event_detail', {})
    home_score_obj = event_detail.get('homeScore', {}) if event_detail else {}
    away_score_obj = event_detail.get('awayScore', {}) if event_detail else {}

    home_innings, num_innings_h = parse_inning_scores(home_score_obj)
    away_innings, num_innings_a = parse_inning_scores(away_score_obj)
    num_innings = max(num_innings_h or 0, num_innings_a or 0) or None

    home_score = home_score_obj.get('current') if home_score_obj else None
    away_score = away_score_obj.get('current') if away_score_obj else None
    winner_code = event_detail.get('winnerCode') if event_detail else None
    slug = event_detail.get('slug') if event_detail else None
    timestamp = event_detail.get('startTimestamp') if event_detail else None

    venue = event_detail.get('venue', {}) if event_detail else {}
    venue_name = venue.get('stadium', {}).get('name') if isinstance(venue, dict) else None
    venue_city = venue.get('city', {}).get('name') if isinstance(venue, dict) else None
    attendance = event_detail.get('attendance') if event_detail else None

    season = event_detail.get('season', {}) if event_detail else {}
    season_name = season.get('name') if isinstance(season, dict) else None

    # Raw JSONs
    raw_odds = json.dumps(data.get('odds')) if data.get('odds') else None
    raw_statistics = json.dumps(data.get('statistics')) if data.get('statistics') else None
    raw_lineups = json.dumps(data.get('lineups')) if data.get('lineups') else None

    has_odds = 1 if (data.get('odds') and data['odds'].get('markets')) else 0
    has_statistics = 1 if (data.get('statistics') and data['statistics'].get('statistics')) else 0
    has_lineups = 1 if (data.get('lineups') and
                        (data['lineups'].get('home') or data['lineups'].get('away'))) else 0

    # Si force, borrar datos previos
    if force:
        for table in ['odds', 'statistics', 'lineups', 'player_match_stats', 'missing_players']:
            conn.execute(f"DELETE FROM {table} WHERE event_id = ?", (event_id,))
        conn.execute("DELETE FROM events WHERE event_id = ?", (event_id,))

    # Insertar evento
    conn.execute("""
        INSERT OR REPLACE INTO events (
            event_id, date, time, timestamp, sport, country, league_name, league_id,
            season_name, home_team, home_team_id, away_team, away_team_id,
            home_score, away_score, home_innings, away_innings, num_innings,
            status, winner_code, slug, venue_name, venue_city, attendance,
            scraped_at, has_odds, has_statistics, has_lineups,
            raw_odds, raw_statistics, raw_lineups
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event_id, date_str, None, timestamp, sport,
        'USA', tournament, None, season_name,
        home, None, away, None,
        home_score, away_score, home_innings, away_innings, num_innings,
        'finished', winner_code, slug, venue_name, venue_city, attendance,
        data.get('scraped_at'),
        has_odds, has_statistics, has_lineups,
        raw_odds, raw_statistics, raw_lineups
    ))

    # Parsear e insertar odds
    if has_odds:
        odds_rows = parse_odds(event_id, data['odds'])
        if odds_rows:
            conn.executemany("""
                INSERT INTO odds (
                    event_id, market_id, market_name, choice_group,
                    market_source_id, structure_type, is_live, suspended,
                    market_db_id, market_fid, market_group, market_period,
                    choice_name, odds_open, odds_close,
                    odds_open_fractional, odds_close_fractional,
                    winning, change, source_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, odds_rows)
            result['odds'] = len(odds_rows)

    # Parsear e insertar statistics
    if has_statistics:
        stats_rows = parse_statistics(event_id, data['statistics'])
        if stats_rows:
            conn.executemany("""
                INSERT INTO statistics (
                    event_id, period, group_name, stat_name, stat_key,
                    home_value, away_value, home_total, away_total,
                    home_display, away_display,
                    compare_code, statistics_type, value_type, render_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, stats_rows)
            result['stats'] = len(stats_rows)

    # Parsear e insertar lineups + player stats + missing
    if has_lineups:
        lineup_rows, pstat_rows, missing_rows = parse_lineups(event_id, data['lineups'])

        if lineup_rows:
            conn.executemany("""
                INSERT INTO lineups (
                    event_id, side, formation, confirmed,
                    player_id, player_name, player_first_name, player_last_name,
                    player_short_name, player_slug, player_position,
                    player_height, player_country_name, player_country_alpha2,
                    player_dob_ts, player_gender, player_sofascore_id,
                    market_value, market_value_currency, user_count,
                    shirt_number, jersey_number, position_in_match,
                    is_substitute, is_captain, team_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, lineup_rows)
            result['lineups'] = len(lineup_rows)

        if pstat_rows:
            conn.executemany("""
                INSERT INTO player_match_stats (event_id, player_id, stat_key, stat_value)
                VALUES (?, ?, ?, ?)
            """, pstat_rows)
            result['player_stats'] = len(pstat_rows)

        if missing_rows:
            conn.executemany("""
                INSERT INTO missing_players (
                    event_id, side, player_id, player_name, player_short_name,
                    player_position, player_height, player_country,
                    player_dob_ts, type, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, missing_rows)
            result['missing'] = len(missing_rows)

    return result


def get_json_files(source_dir: Path, year: Optional[int] = None) -> List[Path]:
    files = []
    if not source_dir.exists():
        logger.warning(f"Directorio no existe: {source_dir}")
        return files

    if year:
        year_dir = source_dir / str(year)
        if year_dir.exists():
            files = sorted(year_dir.glob("*.json"))
    else:
        for year_dir in sorted(source_dir.iterdir()):
            if year_dir.is_dir():
                files.extend(sorted(year_dir.glob("*.json")))

    return files


def import_from_source(source_dir: Path, conn: sqlite3.Connection,
                       year: Optional[int] = None, force: bool = False):
    files = get_json_files(source_dir, year)
    total = len(files)
    logger.info(f"Encontrados {total:,} archivos JSON en {source_dir}")

    if total == 0:
        return

    imported = 0
    skipped = 0
    errors = 0
    totals = {'odds': 0, 'stats': 0, 'lineups': 0, 'player_stats': 0, 'missing': 0}

    start_time = time.time()
    batch_size = 500

    for i, filepath in enumerate(files):
        result = import_json_file(filepath, conn, force=force)

        if result['error']:
            errors += 1
            if errors <= 10:
                logger.warning(f"Error en {filepath.name}: {result['error']}")
        elif result['skipped']:
            skipped += 1
        else:
            imported += 1
            for key in totals:
                totals[key] += result.get(key, 0)

        if (i + 1) % batch_size == 0:
            conn.commit()
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"  [{i+1:,}/{total:,}] {imported:,} importados, {skipped:,} skip, "
                f"{errors:,} err | {rate:.1f} archivos/seg | ETA: {remaining/60:.1f} min"
            )

    conn.commit()
    elapsed = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info(f"IMPORTACIÓN COMPLETADA en {elapsed:.1f}s")
    logger.info(f"  Archivos procesados: {total:,}")
    logger.info(f"  Importados: {imported:,}")
    logger.info(f"  Omitidos (ya existían): {skipped:,}")
    logger.info(f"  Errores: {errors:,}")
    logger.info(f"  Filas insertadas:")
    logger.info(f"    Odds: {totals['odds']:,}")
    logger.info(f"    Statistics: {totals['stats']:,}")
    logger.info(f"    Lineups: {totals['lineups']:,}")
    logger.info(f"    Player stats: {totals['player_stats']:,}")
    logger.info(f"    Missing players: {totals['missing']:,}")


def show_stats(conn: sqlite3.Connection):
    tables = ['events', 'odds', 'statistics', 'lineups',
              'player_match_stats', 'missing_players']

    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DE mlb.db")
    print("=" * 60)

    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:.<30} {count:>12,} filas")
        except Exception:
            print(f"  {table:.<30} (tabla no existe)")

    try:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(has_odds) as with_odds,
                SUM(has_statistics) as with_stats,
                SUM(has_lineups) as with_lineups,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM events
        """).fetchone()
        if row and row[0] > 0:
            print(f"\n  Eventos con odds: {row[1]:,}")
            print(f"  Eventos con statistics: {row[2]:,}")
            print(f"  Eventos con lineups: {row[3]:,}")
            print(f"  Período: {row[4]} → {row[5]}")
    except Exception:
        pass

    # Torneos
    try:
        tournaments = conn.execute("""
            SELECT league_name, COUNT(*) as cnt
            FROM events
            GROUP BY league_name
            ORDER BY cnt DESC
            LIMIT 15
        """).fetchall()
        if tournaments:
            print(f"\n  Torneos:")
            for t in tournaments:
                print(f"    {t[0]}: {t[1]:,} eventos")
    except Exception:
        pass

    print()


def main():
    parser = argparse.ArgumentParser(description='Importar JSONs de baseball a mlb.db')
    parser.add_argument('--source', choices=['raw'],
                        help='Fuente de JSONs (raw/)')
    parser.add_argument('--year', type=int, help='Año específico')
    parser.add_argument('--force', action='store_true',
                        help='Re-importar aunque ya exista')
    parser.add_argument('--stats', action='store_true',
                        help='Mostrar estadísticas de la DB')

    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_FILE))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")

    create_schema(conn)

    if args.stats:
        show_stats(conn)
        conn.close()
        return

    if not args.source:
        parser.print_help()
        conn.close()
        return

    if args.source == 'raw':
        logger.info(f"\n{'='*60}")
        logger.info(f"IMPORTANDO desde raw/ (JSONs de GitHub Actions)")
        logger.info(f"{'='*60}")
        import_from_source(RAW_DIR, conn, year=args.year, force=args.force)

    show_stats(conn)
    conn.close()
    logger.info("Importación finalizada")


if __name__ == '__main__':
    main()
