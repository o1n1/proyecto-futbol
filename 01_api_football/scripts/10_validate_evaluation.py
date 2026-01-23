"""
Script para validar estrategias en el período de EVALUATION (Ago-Oct 2025).

FASE 3 del análisis de estrategia de apuestas.
Aplica las estrategias seleccionadas a datos que NO se usaron para calibración.

Compara accuracy de calibración vs evaluation para detectar degradación.
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import joblib
import logging
from typing import Dict, List
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

# Período de EVALUATION
EVALUATION_START = '2025-08-01'
EVALUATION_END = '2025-10-05'

# Rutas relativas al directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')

# Umbral de degradación
MAX_DEGRADATION = 0.10  # Si accuracy baja más de 10%, marcar como degradada


# ============================================
# CARGA DE DATOS
# ============================================

def fetch_evaluation_data() -> pd.DataFrame:
    """Carga fixtures del período de evaluación."""
    logger.info(f"Cargando datos de evaluación ({EVALUATION_START} a {EVALUATION_END})...")

    url = f"{SUPABASE_URL}/rest/v1/fixtures"
    all_data = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            'select': 'fixture_id,date,league_id,league_name,league_country,home_team_name,away_team_name,goals_home,goals_away,odds_home,odds_draw,odds_away,status_short',
            'date': f'gte.{EVALUATION_START}',
            'order': 'date.asc',
            'offset': offset,
            'limit': page_size
        }

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

    # Filtrar por período y partidos terminados
    df = df[(df['date'] >= EVALUATION_START) & (df['date'] < EVALUATION_END)]
    df = df[df['status_short'] == 'FT']

    logger.info(f"Cargados {len(df):,} partidos terminados en evaluación")
    return df


def fetch_features_for_fixtures(fixture_ids: List[int]) -> pd.DataFrame:
    """Carga features para un conjunto de fixtures."""
    logger.info(f"Cargando features para {len(fixture_ids):,} fixtures...")

    url = f"{SUPABASE_URL}/rest/v1/fixture_features"
    all_data = []

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
            raise Exception(f"Error: {response.text}")

        data = response.json()
        all_data.extend(data)

    df = pd.DataFrame(all_data)
    return df


def load_models():
    """Carga los modelos entrenados."""
    models = {}

    # Modelo resultado
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

    return models


def generate_predictions(features_df: pd.DataFrame, models: Dict) -> pd.DataFrame:
    """Genera predicciones para los partidos."""
    predictions = pd.DataFrame()
    predictions['fixture_id'] = features_df['fixture_id']

    # Predicciones resultado
    result_model = models['result']
    X = features_df[result_model['feature_cols']].copy()
    X_imputed = result_model['imputer'].transform(X)
    X_scaled = result_model['scaler'].transform(X_imputed)

    proba = result_model['model'].predict_proba(X_scaled)
    classes = result_model['label_encoder'].classes_

    for i, cls in enumerate(classes):
        predictions[f'prob_{cls}'] = proba[:, i]

    # Predicciones Over 2.5
    over25_model = models['over25']
    X = features_df[over25_model['feature_cols']].copy()
    X_imputed = over25_model['imputer'].transform(X)
    X_scaled = over25_model['scaler'].transform(X_imputed)
    predictions['prob_over25'] = over25_model['model'].predict_proba(X_scaled)[:, 1]

    # Predicciones BTTS
    btts_model = models['btts']
    X = features_df[btts_model['feature_cols']].copy()
    X_imputed = btts_model['imputer'].transform(X)
    X_scaled = btts_model['scaler'].transform(X_imputed)
    predictions['prob_btts'] = btts_model['model'].predict_proba(X_scaled)[:, 1]

    return predictions


def calculate_actual_results(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula resultados reales."""
    df = fixtures_df.copy()

    df['actual_result'] = df.apply(
        lambda x: 'H' if x['goals_home'] > x['goals_away']
                  else ('A' if x['goals_home'] < x['goals_away'] else 'D'),
        axis=1
    )
    df['actual_over25'] = (df['goals_home'] + df['goals_away']) > 2.5
    df['actual_btts'] = (df['goals_home'] > 0) & (df['goals_away'] > 0)

    return df


# ============================================
# VALIDACIÓN
# ============================================

def validate_strategies(merged_df: pd.DataFrame, strategies_df: pd.DataFrame) -> pd.DataFrame:
    """Valida cada estrategia en el período de evaluación."""
    logger.info("Validando estrategias en período de evaluación...")

    results = []

    for _, strategy in strategies_df.iterrows():
        league_id = strategy['league_id']
        tipo_apuesta = strategy['tipo_apuesta']
        umbral = strategy['umbral']
        accuracy_calib = strategy['accuracy']

        # Filtrar partidos de esta liga
        df_liga = merged_df[merged_df['league_id'] == league_id]

        # Determinar columnas
        if tipo_apuesta == 'Home':
            prob_col, actual_col, target_value = 'prob_H', 'actual_result', 'H'
            odds_col = 'odds_home'
        elif tipo_apuesta == 'Draw':
            prob_col, actual_col, target_value = 'prob_D', 'actual_result', 'D'
            odds_col = 'odds_draw'
        elif tipo_apuesta == 'Away':
            prob_col, actual_col, target_value = 'prob_A', 'actual_result', 'A'
            odds_col = 'odds_away'
        elif tipo_apuesta == 'Over25':
            prob_col, actual_col, target_value = 'prob_over25', 'actual_over25', True
            odds_col = None
        elif tipo_apuesta == 'BTTS':
            prob_col, actual_col, target_value = 'prob_btts', 'actual_btts', True
            odds_col = None

        # Filtrar por umbral
        df_filtered = df_liga[df_liga[prob_col] >= umbral]
        n_partidos_eval = len(df_filtered)

        if n_partidos_eval == 0:
            accuracy_eval = None
            degradation = None
            is_degraded = True
        else:
            aciertos = (df_filtered[actual_col] == target_value).sum()
            accuracy_eval = aciertos / n_partidos_eval
            degradation = accuracy_calib - accuracy_eval
            is_degraded = degradation > MAX_DEGRADATION

        # Contar con cuotas
        n_con_cuotas = 0
        if odds_col and n_partidos_eval > 0:
            n_con_cuotas = df_filtered[odds_col].notna().sum()

        results.append({
            'strategy_id': strategy['strategy_id'],
            'league_id': league_id,
            'league_name': strategy['league_name'],
            'league_country': strategy['league_country'],
            'tipo_apuesta': tipo_apuesta,
            'umbral': umbral,
            'n_partidos_calib': strategy['n_partidos'],
            'accuracy_calib': accuracy_calib,
            'n_partidos_eval': n_partidos_eval,
            'accuracy_eval': round(accuracy_eval, 4) if accuracy_eval else None,
            'degradation': round(degradation, 4) if degradation else None,
            'is_degraded': is_degraded,
            'n_con_cuotas_eval': n_con_cuotas
        })

    return pd.DataFrame(results)


# ============================================
# MAIN
# ============================================

def main():
    """Ejecuta la validación de estrategias."""
    logger.info("=" * 60)
    logger.info("FASE 3: Validación en Período de Evaluación (Ago-Oct 2025)")
    logger.info("=" * 60)

    # 1. Cargar estrategias seleccionadas
    strategies_path = os.path.join(OUTPUTS_DIR, 'selected_strategies.xlsx')
    strategies_df = pd.read_excel(strategies_path)
    logger.info(f"Cargadas {len(strategies_df):,} estrategias seleccionadas")

    # 2. Cargar datos de evaluación
    fixtures_df = fetch_evaluation_data()

    # 3. Cargar features
    fixture_ids = fixtures_df['fixture_id'].tolist()
    features_df = fetch_features_for_fixtures(fixture_ids)

    # 4. Cargar modelos y generar predicciones
    models = load_models()
    predictions_df = generate_predictions(features_df, models)

    # 5. Calcular resultados reales
    fixtures_df = calculate_actual_results(fixtures_df)

    # 6. Merge
    merged_df = fixtures_df.merge(predictions_df, on='fixture_id', how='inner')
    logger.info(f"Partidos con predicciones: {len(merged_df):,}")

    # 7. Validar estrategias
    validated_df = validate_strategies(merged_df, strategies_df)

    # 8. Filtrar estrategias no degradadas
    validated_df = validated_df.sort_values('accuracy_eval', ascending=False)

    # 9. Guardar
    output_path = os.path.join(OUTPUTS_DIR, 'validated_strategies.xlsx')
    validated_df.to_excel(output_path, index=False)
    logger.info(f"Resultados guardados en: {output_path}")

    # Resumen
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)

    total = len(validated_df)
    degraded = validated_df['is_degraded'].sum()
    valid = total - degraded

    logger.info(f"Total estrategias validadas: {total:,}")
    logger.info(f"Estrategias degradadas (bajan >{MAX_DEGRADATION:.0%}): {degraded:,}")
    logger.info(f"Estrategias válidas: {valid:,}")

    # Top 10 no degradadas
    valid_strategies = validated_df[~validated_df['is_degraded']]
    if len(valid_strategies) > 0:
        logger.info("\nTop 10 estrategias válidas:")
        top10 = valid_strategies.head(10)
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            acc_c = row['accuracy_calib']
            acc_e = row['accuracy_eval'] if row['accuracy_eval'] else 0
            logger.info(f"  {i:2d}. {row['league_name'][:20]:20s} | {row['tipo_apuesta']:6s} >= {row['umbral']:.2f} | "
                       f"Calib: {acc_c:.1%} | Eval: {acc_e:.1%} | N: {row['n_partidos_eval']}")

    return validated_df


if __name__ == "__main__":
    main()
