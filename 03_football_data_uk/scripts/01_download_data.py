"""
Script para descargar datos históricos de cuotas de Football-Data.co.uk

Descarga CSVs con resultados y cuotas de múltiples casas de apuestas.
Fuente: https://www.football-data.co.uk/data.php

Uso:
    python download_football_data.py
"""

import os
import requests
import pandas as pd
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URL base de Football-Data.co.uk
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Rutas relativas al directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

# Ligas a descargar con su ID en nuestra BD
LEAGUES = {
    'E0': {'name': 'Premier League', 'country': 'England', 'league_id': 39},
    'E1': {'name': 'Championship', 'country': 'England', 'league_id': 40},
    'D1': {'name': 'Bundesliga', 'country': 'Germany', 'league_id': 78},
    'I1': {'name': 'Serie A', 'country': 'Italy', 'league_id': 135},
    'SP1': {'name': 'La Liga', 'country': 'Spain', 'league_id': 140},
    'F1': {'name': 'Ligue 1', 'country': 'France', 'league_id': 61},
    'N1': {'name': 'Eredivisie', 'country': 'Netherlands', 'league_id': 88},
    'P1': {'name': 'Primeira Liga', 'country': 'Portugal', 'league_id': 94},
    'B1': {'name': 'Jupiler League', 'country': 'Belgium', 'league_id': 144},
    'T1': {'name': 'Super Lig', 'country': 'Turkey', 'league_id': 203},
}

# Temporadas a descargar (código de 4 dígitos: XXYY = 20XX-20YY)
SEASONS = [
    '1920',  # 2019-20
    '2021',  # 2020-21
    '2122',  # 2021-22
    '2223',  # 2022-23
    '2324',  # 2023-24
    '2425',  # 2024-25
]


def download_csv(league_code: str, season: str) -> pd.DataFrame:
    """
    Descarga CSV de una liga/temporada específica.

    Args:
        league_code: Código de la liga (ej: 'E0', 'SP1')
        season: Código de temporada (ej: '2324' para 2023-24)

    Returns:
        DataFrame con los datos o None si falla
    """
    url = f"{BASE_URL}/{season}/{league_code}.csv"

    try:
        # Intentar descargar
        response = requests.get(url, timeout=30, verify=False)

        if response.status_code == 200:
            # Leer CSV desde el contenido
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), encoding='utf-8', on_bad_lines='skip')
            return df
        else:
            logger.warning(f"  HTTP {response.status_code} para {url}")
            return None

    except Exception as e:
        logger.error(f"  Error descargando {url}: {str(e)}")
        return None


def process_dataframe(df: pd.DataFrame, league_code: str, season: str) -> pd.DataFrame:
    """
    Procesa y limpia el DataFrame descargado.

    Args:
        df: DataFrame original
        league_code: Código de la liga
        season: Código de temporada

    Returns:
        DataFrame procesado
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Agregar metadatos
    df['league_code'] = league_code
    df['season_code'] = season
    df['league_name'] = LEAGUES[league_code]['name']
    df['league_country'] = LEAGUES[league_code]['country']
    df['league_id'] = LEAGUES[league_code]['league_id']

    # Convertir fecha al formato estándar
    if 'Date' in df.columns:
        # El formato puede ser DD/MM/YYYY o DD/MM/YY
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        except:
            pass

    # Seleccionar columnas relevantes (si existen)
    columns_to_keep = [
        # Metadatos
        'league_code', 'season_code', 'league_name', 'league_country', 'league_id',
        # Match info
        'Date', 'Time', 'HomeTeam', 'AwayTeam',
        # Resultado
        'FTHG', 'FTAG', 'FTR',  # Full Time Home/Away Goals, Result
        'HTHG', 'HTAG', 'HTR',  # Half Time
        # Cuotas Bet365
        'B365H', 'B365D', 'B365A',
        # Cuotas Pinnacle (más precisas)
        'PSH', 'PSD', 'PSA',
        # Cuotas máximas del mercado
        'MaxH', 'MaxD', 'MaxA',
        # Cuotas promedio del mercado
        'AvgH', 'AvgD', 'AvgA',
        # Over/Under 2.5
        'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5',
        # Referencia adicional
        'Div', 'Referee'
    ]

    # Filtrar solo columnas que existen
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()

    # Renombrar columnas para consistencia
    rename_map = {
        'FTHG': 'goals_home',
        'FTAG': 'goals_away',
        'FTR': 'result',
        'B365H': 'odds_b365_home',
        'B365D': 'odds_b365_draw',
        'B365A': 'odds_b365_away',
        'PSH': 'odds_pinnacle_home',
        'PSD': 'odds_pinnacle_draw',
        'PSA': 'odds_pinnacle_away',
        'MaxH': 'odds_max_home',
        'MaxD': 'odds_max_draw',
        'MaxA': 'odds_max_away',
        'AvgH': 'odds_avg_home',
        'AvgD': 'odds_avg_draw',
        'AvgA': 'odds_avg_away',
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


def main():
    """Función principal."""
    logger.info("=" * 60)
    logger.info("DESCARGA DE DATOS DE FOOTBALL-DATA.CO.UK")
    logger.info("=" * 60)

    # Crear directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Estadísticas
    total_downloaded = 0
    total_rows = 0
    all_data = []

    # Descargar cada liga/temporada
    for league_code, league_info in LEAGUES.items():
        logger.info(f"\n--- {league_info['name']} ({league_info['country']}) ---")

        for season in SEASONS:
            season_str = f"20{season[:2]}-{season[2:]}"
            logger.info(f"  Descargando temporada {season_str}...")

            df = download_csv(league_code, season)

            if df is not None and not df.empty:
                df = process_dataframe(df, league_code, season)

                if not df.empty:
                    # Guardar CSV individual
                    filename = f"{OUTPUT_DIR}/{league_code}_{season}.csv"
                    df.to_csv(filename, index=False)

                    all_data.append(df)
                    total_downloaded += 1
                    total_rows += len(df)
                    logger.info(f"    ✓ {len(df)} partidos descargados")
            else:
                logger.info(f"    ✗ No disponible")

    # Combinar todos los datos en un solo archivo
    if all_data:
        logger.info("\n--- Combinando datos ---")
        combined_df = pd.concat(all_data, ignore_index=True)

        # Guardar archivo combinado
        combined_file = f"{OUTPUT_DIR}/all_leagues_combined.csv"
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"Archivo combinado guardado: {combined_file}")

        # Resumen por liga
        logger.info("\n" + "=" * 60)
        logger.info("RESUMEN")
        logger.info("=" * 60)

        summary = combined_df.groupby(['league_name', 'league_country']).agg({
            'Date': ['min', 'max', 'count']
        }).reset_index()
        summary.columns = ['Liga', 'País', 'Fecha Min', 'Fecha Max', 'Partidos']

        for _, row in summary.iterrows():
            logger.info(f"{row['Liga']}: {row['Partidos']} partidos ({row['Fecha Min']} - {row['Fecha Max']})")

        logger.info(f"\nTotal archivos descargados: {total_downloaded}")
        logger.info(f"Total partidos: {total_rows}")
        logger.info(f"Directorio de salida: {OUTPUT_DIR}/")
    else:
        logger.warning("No se descargaron datos")


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()
