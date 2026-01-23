"""
Script para analizar combinaciones de liga + umbral + tipo_apuesta.

FASE 1 del análisis de estrategia de apuestas.
Genera todas las combinaciones posibles y calcula accuracy en el período de CALIBRACIÓN (Jun-Jul 2025).

NO calcula EV ni ROI - solo accuracy.
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import urllib3

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

# Período de CALIBRACIÓN
CALIBRATION_START = '2025-06-01'
CALIBRATION_END = '2025-08-01'

# Configuración del análisis
MIN_PARTIDOS_LIGA = 20  # Mínimo partidos por liga en el período
UMBRALES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
TIPOS_APUESTA = ['Home', 'Draw', 'Away', 'Over25', 'BTTS']

# Rutas relativas al directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')

# Columnas a excluir del entrenamiento (para cargar features)
EXCLUDE_COLS = [
    'fixture_id', 'calculated_at', 'features_version',
    'result', 'total_goals', 'home_goals', 'away_goals',
    'btts', 'over15', 'over25', 'over35'
]


# ============================================
# CARGA DE DATOS
# ============================================

def fetch_calibration_data() -> pd.DataFrame:
    """Carga fixtures del período de calibración con features y cuotas."""
    logger.info(f"Cargando datos de calibración ({CALIBRATION_START} a {CALIBRATION_END})...")

    # Cargar fixtures con información completa
    url = f"{SUPABASE_URL}/rest/v1/fixtures"
    all_data = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            'select': 'fixture_id,date,league_id,league_name,league_country,home_team_name,away_team_name,goals_home,goals_away,odds_home,odds_draw,odds_away,status_short',
            'date': f'gte.{CALIBRATION_START}',
            'order': 'date.asc',
            'offset': offset,
            'limit': page_size
        }

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

    # Filtrar por período de calibración y partidos terminados
    df = df[(df['date'] >= CALIBRATION_START) & (df['date'] < CALIBRATION_END)]
    df = df[df['status_short'] == 'FT']

    logger.info(f"Cargados {len(df):,} partidos terminados en calibración")

    return df


def fetch_features_for_fixtures(fixture_ids: List[int]) -> pd.DataFrame:
    """Carga features para un conjunto de fixtures."""
    logger.info(f"Cargando features para {len(fixture_ids):,} fixtures...")

    url = f"{SUPABASE_URL}/rest/v1/fixture_features"
    all_data = []

    # Dividir en chunks de 100 IDs
    chunk_size = 100
    for i in range(0, len(fixture_ids), chunk_size):
        chunk_ids = fixture_ids[i:i+chunk_size]
        ids_str = ','.join(map(str, chunk_ids))

        params = {
            'select': '*',
            'fixture_id': f'in.({ids_str})'
        }

        response = requests.get(url, headers=HEADERS, params=params, verify=False, timeout=60)

        if response.status_code != 200:
            raise Exception(f"Error al obtener features: {response.text}")

        data = response.json()
        all_data.extend(data)

    df = pd.DataFrame(all_data)
    logger.info(f"Cargadas features para {len(df):,} fixtures")

    return df


# ============================================
# PREDICCIONES
# ============================================

def load_models_and_preprocessors():
    """Carga los modelos entrenados y preprocesadores."""
    logger.info("Cargando modelos y preprocesadores...")

    models = {}

    # Modelo de resultado (H/D/A)
    models['result'] = {
        'model': joblib.load(os.path.join(MODELS_DIR, 'xgb_result_model.pkl')),
        'imputer': joblib.load(os.path.join(MODELS_DIR, 'imputer_result.pkl')),
        'scaler': joblib.load(os.path.join(MODELS_DIR, 'scaler_result.pkl')),
        'label_encoder': joblib.load(os.path.join(MODELS_DIR, 'label_encoder_result.pkl')),
        'feature_cols': json.load(open(os.path.join(MODELS_DIR, 'feature_cols_result.json')))
    }

    # Modelo Over 2.5
    models['over25'] = {
        'model': joblib.load(os.path.join(MODELS_DIR, 'xgb_over25_model.pkl')),
        'imputer': joblib.load(os.path.join(MODELS_DIR, 'imputer_over25.pkl')),
        'scaler': joblib.load(os.path.join(MODELS_DIR, 'scaler_over25.pkl')),
        'feature_cols': json.load(open(os.path.join(MODELS_DIR, 'feature_cols_over25.json')))
    }

    # Modelo BTTS
    models['btts'] = {
        'model': joblib.load(os.path.join(MODELS_DIR, 'xgb_btts_model.pkl')),
        'imputer': joblib.load(os.path.join(MODELS_DIR, 'imputer_btts.pkl')),
        'scaler': joblib.load(os.path.join(MODELS_DIR, 'scaler_btts.pkl')),
        'feature_cols': json.load(open(os.path.join(MODELS_DIR, 'feature_cols_btts.json')))
    }

    logger.info("Modelos cargados correctamente")
    return models


def generate_predictions(features_df: pd.DataFrame, models: Dict) -> pd.DataFrame:
    """Genera predicciones para todos los partidos."""
    logger.info("Generando predicciones...")

    predictions = pd.DataFrame()
    predictions['fixture_id'] = features_df['fixture_id']

    # Predicciones de resultado (H/D/A)
    result_model = models['result']
    feature_cols = result_model['feature_cols']

    X = features_df[feature_cols].copy()
    X_imputed = result_model['imputer'].transform(X)
    X_scaled = result_model['scaler'].transform(X_imputed)

    proba = result_model['model'].predict_proba(X_scaled)
    classes = result_model['label_encoder'].classes_  # ['A', 'D', 'H']

    for i, cls in enumerate(classes):
        predictions[f'prob_{cls}'] = proba[:, i]

    predictions['pred_result'] = result_model['label_encoder'].inverse_transform(
        result_model['model'].predict(X_scaled)
    )

    # Predicciones Over 2.5
    over25_model = models['over25']
    feature_cols = over25_model['feature_cols']

    X = features_df[feature_cols].copy()
    X_imputed = over25_model['imputer'].transform(X)
    X_scaled = over25_model['scaler'].transform(X_imputed)

    proba = over25_model['model'].predict_proba(X_scaled)
    predictions['prob_over25'] = proba[:, 1]  # Probabilidad de True
    predictions['pred_over25'] = over25_model['model'].predict(X_scaled)

    # Predicciones BTTS
    btts_model = models['btts']
    feature_cols = btts_model['feature_cols']

    X = features_df[feature_cols].copy()
    X_imputed = btts_model['imputer'].transform(X)
    X_scaled = btts_model['scaler'].transform(X_imputed)

    proba = btts_model['model'].predict_proba(X_scaled)
    predictions['prob_btts'] = proba[:, 1]  # Probabilidad de True
    predictions['pred_btts'] = btts_model['model'].predict(X_scaled)

    logger.info(f"Generadas predicciones para {len(predictions):,} partidos")
    return predictions


# ============================================
# ANÁLISIS DE COMBINACIONES
# ============================================

def calculate_actual_results(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula los resultados reales de cada partido."""
    df = fixtures_df.copy()

    # Resultado (H/D/A)
    df['actual_result'] = df.apply(
        lambda x: 'H' if x['goals_home'] > x['goals_away']
                  else ('A' if x['goals_home'] < x['goals_away'] else 'D'),
        axis=1
    )

    # Over 2.5
    df['actual_over25'] = (df['goals_home'] + df['goals_away']) > 2.5

    # BTTS
    df['actual_btts'] = (df['goals_home'] > 0) & (df['goals_away'] > 0)

    return df


def analyze_combinations(merged_df: pd.DataFrame) -> List[Dict]:
    """Analiza todas las combinaciones de liga + umbral + tipo_apuesta."""
    logger.info("Analizando combinaciones...")

    results = []

    # Obtener ligas con suficientes partidos
    liga_counts = merged_df.groupby(['league_id', 'league_name', 'league_country']).size().reset_index(name='total_partidos')
    ligas_validas = liga_counts[liga_counts['total_partidos'] >= MIN_PARTIDOS_LIGA]

    logger.info(f"Ligas con >= {MIN_PARTIDOS_LIGA} partidos: {len(ligas_validas)}")

    for _, liga in ligas_validas.iterrows():
        league_id = liga['league_id']
        league_name = liga['league_name']
        league_country = liga['league_country']

        # Filtrar partidos de esta liga
        df_liga = merged_df[merged_df['league_id'] == league_id]

        for umbral in UMBRALES:
            for tipo_apuesta in TIPOS_APUESTA:

                # Determinar columnas según tipo de apuesta
                if tipo_apuesta == 'Home':
                    prob_col = 'prob_H'
                    actual_col = 'actual_result'
                    target_value = 'H'
                    odds_col = 'odds_home'
                elif tipo_apuesta == 'Draw':
                    prob_col = 'prob_D'
                    actual_col = 'actual_result'
                    target_value = 'D'
                    odds_col = 'odds_draw'
                elif tipo_apuesta == 'Away':
                    prob_col = 'prob_A'
                    actual_col = 'actual_result'
                    target_value = 'A'
                    odds_col = 'odds_away'
                elif tipo_apuesta == 'Over25':
                    prob_col = 'prob_over25'
                    actual_col = 'actual_over25'
                    target_value = True
                    odds_col = None  # No tenemos cuotas de over/under
                elif tipo_apuesta == 'BTTS':
                    prob_col = 'prob_btts'
                    actual_col = 'actual_btts'
                    target_value = True
                    odds_col = None  # No tenemos cuotas de BTTS

                # Filtrar por umbral de probabilidad
                df_filtered = df_liga[df_liga[prob_col] >= umbral]

                n_partidos = len(df_filtered)

                if n_partidos == 0:
                    continue

                # Calcular accuracy
                if tipo_apuesta in ['Home', 'Draw', 'Away']:
                    aciertos = (df_filtered[actual_col] == target_value).sum()
                else:
                    aciertos = (df_filtered[actual_col] == target_value).sum()

                accuracy = aciertos / n_partidos if n_partidos > 0 else 0

                # Contar partidos con cuotas
                n_con_cuotas = 0
                if odds_col:
                    n_con_cuotas = df_filtered[odds_col].notna().sum()

                results.append({
                    'league_id': league_id,
                    'league_name': league_name,
                    'league_country': league_country,
                    'tipo_apuesta': tipo_apuesta,
                    'umbral': umbral,
                    'n_partidos': n_partidos,
                    'aciertos': aciertos,
                    'accuracy': round(accuracy, 4),
                    'n_con_cuotas': n_con_cuotas
                })

    logger.info(f"Generadas {len(results):,} combinaciones")
    return results


# ============================================
# MAIN
# ============================================

def main():
    """Ejecuta el análisis de calibración."""
    logger.info("=" * 60)
    logger.info("FASE 1: Análisis de Calibración (Jun-Jul 2025)")
    logger.info("=" * 60)

    # 1. Cargar datos de calibración
    fixtures_df = fetch_calibration_data()

    # 2. Cargar features
    fixture_ids = fixtures_df['fixture_id'].tolist()
    features_df = fetch_features_for_fixtures(fixture_ids)

    # 3. Cargar modelos
    models = load_models_and_preprocessors()

    # 4. Generar predicciones
    predictions_df = generate_predictions(features_df, models)

    # 5. Calcular resultados reales
    fixtures_df = calculate_actual_results(fixtures_df)

    # 6. Merge fixtures con predicciones
    merged_df = fixtures_df.merge(predictions_df, on='fixture_id', how='inner')
    logger.info(f"Partidos con predicciones: {len(merged_df):,}")

    # 7. Analizar todas las combinaciones
    results = analyze_combinations(merged_df)

    # 8. Guardar resultados
    results_df = pd.DataFrame(results)

    # Ordenar por accuracy descendente
    results_df = results_df.sort_values('accuracy', ascending=False)

    # Guardar en Excel
    output_path = os.path.join(OUTPUTS_DIR, 'calibration_results.xlsx')
    results_df.to_excel(output_path, index=False)
    logger.info(f"Resultados guardados en: {output_path}")

    # Resumen
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)
    logger.info(f"Total combinaciones: {len(results_df):,}")
    logger.info(f"Combinaciones con accuracy >= 55%: {len(results_df[results_df['accuracy'] >= 0.55]):,}")
    logger.info(f"Combinaciones con accuracy >= 60%: {len(results_df[results_df['accuracy'] >= 0.60]):,}")
    logger.info(f"Combinaciones con accuracy >= 65%: {len(results_df[results_df['accuracy'] >= 0.65]):,}")

    # Top 10 combinaciones
    logger.info("\nTop 10 combinaciones por accuracy:")
    top10 = results_df.head(10)
    for _, row in top10.iterrows():
        logger.info(f"  {row['league_name']} ({row['league_country']}) - {row['tipo_apuesta']} >= {row['umbral']}: "
                   f"{row['accuracy']:.1%} ({row['aciertos']}/{row['n_partidos']})")

    return results_df


if __name__ == "__main__":
    main()
