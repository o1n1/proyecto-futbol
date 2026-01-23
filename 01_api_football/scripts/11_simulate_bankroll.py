"""
Script para simular bankroll día a día con las estrategias validadas.

FASE 4 del análisis de estrategia de apuestas.
Simula apuestas en el período de EVALUATION (Ago-Oct 2025).

Reglas:
1. El stake se calcula AL INICIO del día con el bankroll del día anterior
2. El mismo stake fijo se usa para TODAS las apuestas del día
3. Si bankroll llega a 0 o menos → Simulación se detiene
4. No se puede apostar más del 100% del bankroll disponible
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import joblib
import logging
from datetime import datetime, timedelta
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

# Período de EVALUATION
EVALUATION_START = '2025-08-01'
EVALUATION_END = '2025-10-05'

# Configuración de bankroll
BANKROLL_INICIAL = 1000
STAKE_PCT = 0.02  # 2% del bankroll por apuesta

# Rutas relativas al directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')


# ============================================
# CARGA DE DATOS
# ============================================

def fetch_evaluation_data() -> pd.DataFrame:
    """Carga fixtures del período de evaluación con fechas."""
    logger.info(f"Cargando datos de evaluación ({EVALUATION_START} a {EVALUATION_END})...")

    url = f"{SUPABASE_URL}/rest/v1/fixtures"
    all_data = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            'select': 'fixture_id,date,league_id,league_name,goals_home,goals_away,odds_home,odds_draw,odds_away,status_short',
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
    df['date_only'] = df['date'].dt.date

    # Filtrar
    df = df[(df['date'] >= EVALUATION_START) & (df['date'] < EVALUATION_END)]
    df = df[df['status_short'] == 'FT']

    logger.info(f"Cargados {len(df):,} partidos")
    return df


def fetch_features_for_fixtures(fixture_ids: List[int]) -> pd.DataFrame:
    """Carga features para fixtures."""
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

    return pd.DataFrame(all_data)


def load_models():
    """Carga modelos entrenados."""
    models = {}

    models['result'] = {
        'model': joblib.load(os.path.join(MODELS_DIR, 'xgb_result_model.pkl')),
        'imputer': joblib.load(os.path.join(MODELS_DIR, 'imputer_result.pkl')),
        'scaler': joblib.load(os.path.join(MODELS_DIR, 'scaler_result.pkl')),
        'label_encoder': joblib.load(os.path.join(MODELS_DIR, 'label_encoder_result.pkl')),
        'feature_cols': json.load(open(os.path.join(MODELS_DIR, 'feature_cols_result.json')))
    }

    models['over25'] = {
        'model': joblib.load(os.path.join(MODELS_DIR, 'xgb_over25_model.pkl')),
        'imputer': joblib.load(os.path.join(MODELS_DIR, 'imputer_over25.pkl')),
        'scaler': joblib.load(os.path.join(MODELS_DIR, 'scaler_over25.pkl')),
        'feature_cols': json.load(open(os.path.join(MODELS_DIR, 'feature_cols_over25.json')))
    }

    models['btts'] = {
        'model': joblib.load(os.path.join(MODELS_DIR, 'xgb_btts_model.pkl')),
        'imputer': joblib.load(os.path.join(MODELS_DIR, 'imputer_btts.pkl')),
        'scaler': joblib.load(os.path.join(MODELS_DIR, 'scaler_btts.pkl')),
        'feature_cols': json.load(open(os.path.join(MODELS_DIR, 'feature_cols_btts.json')))
    }

    return models


def generate_predictions(features_df: pd.DataFrame, models: Dict) -> pd.DataFrame:
    """Genera predicciones."""
    predictions = pd.DataFrame()
    predictions['fixture_id'] = features_df['fixture_id']

    # Resultado
    m = models['result']
    X = features_df[m['feature_cols']].copy()
    X_imp = m['imputer'].transform(X)
    X_scl = m['scaler'].transform(X_imp)
    proba = m['model'].predict_proba(X_scl)
    for i, cls in enumerate(m['label_encoder'].classes_):
        predictions[f'prob_{cls}'] = proba[:, i]

    # Over 2.5
    m = models['over25']
    X = features_df[m['feature_cols']].copy()
    X_imp = m['imputer'].transform(X)
    X_scl = m['scaler'].transform(X_imp)
    predictions['prob_over25'] = m['model'].predict_proba(X_scl)[:, 1]

    # BTTS
    m = models['btts']
    X = features_df[m['feature_cols']].copy()
    X_imp = m['imputer'].transform(X)
    X_scl = m['scaler'].transform(X_imp)
    predictions['prob_btts'] = m['model'].predict_proba(X_scl)[:, 1]

    return predictions


def calculate_actual_results(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula resultados reales."""
    df = df.copy()
    df['actual_result'] = df.apply(
        lambda x: 'H' if x['goals_home'] > x['goals_away']
                  else ('A' if x['goals_home'] < x['goals_away'] else 'D'),
        axis=1
    )
    df['actual_over25'] = (df['goals_home'] + df['goals_away']) > 2.5
    df['actual_btts'] = (df['goals_home'] > 0) & (df['goals_away'] > 0)
    return df


# ============================================
# SIMULACIÓN
# ============================================

def get_apuestas_dia(partidos_dia: pd.DataFrame, strategies: pd.DataFrame) -> List[Dict]:
    """Identifica las apuestas a realizar en un día según las estrategias."""
    apuestas = []

    for _, strategy in strategies.iterrows():
        league_id = strategy['league_id']
        tipo_apuesta = strategy['tipo_apuesta']
        umbral = strategy['umbral']

        # Filtrar partidos de esta liga
        partidos_liga = partidos_dia[partidos_dia['league_id'] == league_id]

        for _, partido in partidos_liga.iterrows():
            # Determinar columnas según tipo de apuesta
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
            else:
                continue

            # Verificar si cumple umbral
            prob = partido.get(prob_col, 0)
            if prob >= umbral:
                # Solo apostar si tenemos cuota
                cuota = partido.get(odds_col) if odds_col else None

                if cuota and not pd.isna(cuota):
                    actual = partido[actual_col]
                    acierto = (actual == target_value)

                    apuestas.append({
                        'fixture_id': partido['fixture_id'],
                        'strategy_id': strategy['strategy_id'],
                        'league_name': strategy['league_name'],
                        'tipo_apuesta': tipo_apuesta,
                        'prob': prob,
                        'umbral': umbral,
                        'cuota': cuota,
                        'acierto': acierto
                    })

    return apuestas


def simulate_bankroll(merged_df: pd.DataFrame, strategies: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Simula el bankroll día a día."""
    logger.info("Simulando bankroll día a día...")

    bankroll = BANKROLL_INICIAL
    daily_records = []
    all_bets = []

    # Obtener fechas únicas ordenadas
    fechas = sorted(merged_df['date_only'].unique())
    logger.info(f"Simulando {len(fechas)} días de apuestas")

    for fecha in fechas:
        if bankroll <= 0:
            logger.warning(f"QUIEBRA en {fecha}. Bankroll: {bankroll:.2f}")
            break

        # Partidos del día
        partidos_dia = merged_df[merged_df['date_only'] == fecha]

        # Calcular stake del día (fijo para todas las apuestas)
        stake_dia = bankroll * STAKE_PCT

        # Obtener apuestas según estrategias
        apuestas_dia = get_apuestas_dia(partidos_dia, strategies)

        if len(apuestas_dia) == 0:
            continue

        # Procesar apuestas
        ganancia_dia = 0
        ganadas = 0
        perdidas = 0

        for apuesta in apuestas_dia:
            if apuesta['acierto']:
                ganancia = stake_dia * (apuesta['cuota'] - 1)
                ganancia_dia += ganancia
                ganadas += 1
            else:
                ganancia_dia -= stake_dia
                perdidas += 1

            # Registrar apuesta
            all_bets.append({
                'fecha': fecha,
                'fixture_id': apuesta['fixture_id'],
                'strategy_id': apuesta['strategy_id'],
                'tipo_apuesta': apuesta['tipo_apuesta'],
                'prob': apuesta['prob'],
                'cuota': apuesta['cuota'],
                'stake': stake_dia,
                'acierto': apuesta['acierto'],
                'ganancia': stake_dia * (apuesta['cuota'] - 1) if apuesta['acierto'] else -stake_dia
            })

        # Actualizar bankroll
        bankroll_anterior = bankroll
        bankroll += ganancia_dia

        # Registrar día
        daily_records.append({
            'fecha': fecha,
            'n_apuestas': len(apuestas_dia),
            'ganadas': ganadas,
            'perdidas': perdidas,
            'stake_dia': stake_dia,
            'ganancia_dia': ganancia_dia,
            'bankroll_anterior': bankroll_anterior,
            'bankroll_nuevo': bankroll,
            'roi_dia': (ganancia_dia / (stake_dia * len(apuestas_dia)) * 100) if len(apuestas_dia) > 0 else 0
        })

    return daily_records, all_bets


def calculate_metrics(daily_records: List[Dict], all_bets: List[Dict]) -> Dict:
    """Calcula métricas finales de la simulación."""
    if not daily_records:
        return {}

    df_daily = pd.DataFrame(daily_records)
    df_bets = pd.DataFrame(all_bets)

    bankroll_final = df_daily['bankroll_nuevo'].iloc[-1]
    roi_total = ((bankroll_final - BANKROLL_INICIAL) / BANKROLL_INICIAL) * 100

    # Max drawdown
    df_daily['max_bankroll'] = df_daily['bankroll_nuevo'].cummax()
    df_daily['drawdown'] = (df_daily['max_bankroll'] - df_daily['bankroll_nuevo']) / df_daily['max_bankroll'] * 100
    max_drawdown = df_daily['drawdown'].max()

    # Win rate
    total_apuestas = len(df_bets)
    total_ganadas = df_bets['acierto'].sum()
    win_rate = (total_ganadas / total_apuestas * 100) if total_apuestas > 0 else 0

    # Sharpe ratio (simplificado)
    if len(df_daily) > 1:
        daily_returns = df_daily['ganancia_dia'] / df_daily['bankroll_anterior']
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    else:
        sharpe = 0

    return {
        'bankroll_inicial': BANKROLL_INICIAL,
        'bankroll_final': round(bankroll_final, 2),
        'roi_total': round(roi_total, 2),
        'total_dias': len(df_daily),
        'total_apuestas': total_apuestas,
        'total_ganadas': int(total_ganadas),
        'total_perdidas': int(total_apuestas - total_ganadas),
        'win_rate': round(win_rate, 2),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe_ratio': round(sharpe, 2)
    }


# ============================================
# MAIN
# ============================================

def main():
    """Ejecuta la simulación de bankroll."""
    logger.info("=" * 60)
    logger.info("FASE 4: Simulación de Bankroll Día a Día")
    logger.info("=" * 60)
    logger.info(f"Bankroll inicial: {BANKROLL_INICIAL}")
    logger.info(f"Stake por apuesta: {STAKE_PCT:.1%}")

    # 1. Cargar estrategias validadas (solo no degradadas)
    strategies_path = os.path.join(OUTPUTS_DIR, 'validated_strategies.xlsx')
    strategies_df = pd.read_excel(strategies_path)

    # Filtrar solo estrategias no degradadas
    strategies_df = strategies_df[~strategies_df['is_degraded']]
    logger.info(f"Estrategias no degradadas: {len(strategies_df):,}")

    if len(strategies_df) == 0:
        logger.warning("No hay estrategias válidas para simular!")
        return

    # 2. Cargar datos
    fixtures_df = fetch_evaluation_data()
    fixture_ids = fixtures_df['fixture_id'].tolist()
    features_df = fetch_features_for_fixtures(fixture_ids)

    # 3. Cargar modelos y generar predicciones
    models = load_models()
    predictions_df = generate_predictions(features_df, models)

    # 4. Calcular resultados reales
    fixtures_df = calculate_actual_results(fixtures_df)

    # 5. Merge
    merged_df = fixtures_df.merge(predictions_df, on='fixture_id', how='inner')

    # 6. Simular
    daily_records, all_bets = simulate_bankroll(merged_df, strategies_df)

    # 7. Calcular métricas
    metrics = calculate_metrics(daily_records, all_bets)

    # 8. Guardar resultados
    if daily_records:
        pd.DataFrame(daily_records).to_excel(
            os.path.join(OUTPUTS_DIR, 'simulation_daily.xlsx'), index=False
        )
    if all_bets:
        pd.DataFrame(all_bets).to_excel(
            os.path.join(OUTPUTS_DIR, 'simulation_bets.xlsx'), index=False
        )

    # Guardar métricas
    with open(os.path.join(OUTPUTS_DIR, 'simulation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Resumen
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS DE LA SIMULACIÓN")
    logger.info("=" * 60)

    if metrics:
        logger.info(f"Bankroll inicial:  ${metrics['bankroll_inicial']:,.2f}")
        logger.info(f"Bankroll final:    ${metrics['bankroll_final']:,.2f}")
        logger.info(f"ROI Total:         {metrics['roi_total']:+.2f}%")
        logger.info(f"Total días:        {metrics['total_dias']}")
        logger.info(f"Total apuestas:    {metrics['total_apuestas']}")
        logger.info(f"Ganadas/Perdidas:  {metrics['total_ganadas']}/{metrics['total_perdidas']}")
        logger.info(f"Win Rate:          {metrics['win_rate']:.1f}%")
        logger.info(f"Max Drawdown:      {metrics['max_drawdown']:.1f}%")
        logger.info(f"Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")

    logger.info(f"\nResultados guardados en: {OUTPUTS_DIR}/")

    return metrics


if __name__ == "__main__":
    main()
