"""
Script de Predicción en Batch
Genera predicciones para todos los fixtures con features calculadas.

Uso:
    python predict_batch.py [--future-only] [--min-date YYYY-MM-DD]

Opciones:
    --future-only: Solo predice partidos futuros (sin resultado)
    --min-date: Solo predice partidos desde esta fecha
"""

import pandas as pd
import numpy as np
import requests
import json
import joblib
import logging
import argparse
from datetime import datetime
import os

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración Supabase
SUPABASE_URL = "https://ykqaplnfrhvjqkvejudg.supabase.co"
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlrcWFwbG5mcmh2anFrdmVqdWRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg2NjY4NjgsImV4cCI6MjA4NDI0Mjg2OH0.abeJY6QxUn4gT5GYJmoD2xJ7uPVNEwAVAxJ0wE5bMvM')

# Rutas relativas al directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')


class Predictor:
    """Clase para generar predicciones usando modelos entrenados."""

    def __init__(self):
        self.models = {}
        self.imputers = {}
        self.scalers = {}
        self.feature_cols = {}
        self.label_encoder = None
        self.loaded = False

    def load_models(self):
        """Carga todos los modelos entrenados."""
        logger.info("Cargando modelos...")

        # Modelo de resultado
        try:
            self.models['result'] = joblib.load(f'{MODELS_DIR}/xgb_result_model.pkl')
            self.imputers['result'] = joblib.load(f'{MODELS_DIR}/imputer_result.pkl')
            self.scalers['result'] = joblib.load(f'{MODELS_DIR}/scaler_result.pkl')
            self.label_encoder = joblib.load(f'{MODELS_DIR}/label_encoder_result.pkl')
            with open(f'{MODELS_DIR}/feature_cols_result.json', 'r') as f:
                self.feature_cols['result'] = json.load(f)
            logger.info(f"  - Modelo Result cargado ({len(self.feature_cols['result'])} features)")
        except Exception as e:
            logger.error(f"Error cargando modelo result: {e}")
            raise

        # Modelo Over 2.5
        try:
            self.models['over25'] = joblib.load(f'{MODELS_DIR}/xgb_over25_model.pkl')
            self.imputers['over25'] = joblib.load(f'{MODELS_DIR}/imputer_over25.pkl')
            self.scalers['over25'] = joblib.load(f'{MODELS_DIR}/scaler_over25.pkl')
            with open(f'{MODELS_DIR}/feature_cols_over25.json', 'r') as f:
                self.feature_cols['over25'] = json.load(f)
            logger.info(f"  - Modelo Over25 cargado ({len(self.feature_cols['over25'])} features)")
        except Exception as e:
            logger.warning(f"Modelo over25 no disponible: {e}")

        # Modelo BTTS
        try:
            self.models['btts'] = joblib.load(f'{MODELS_DIR}/xgb_btts_model.pkl')
            self.imputers['btts'] = joblib.load(f'{MODELS_DIR}/imputer_btts.pkl')
            self.scalers['btts'] = joblib.load(f'{MODELS_DIR}/scaler_btts.pkl')
            with open(f'{MODELS_DIR}/feature_cols_btts.json', 'r') as f:
                self.feature_cols['btts'] = json.load(f)
            logger.info(f"  - Modelo BTTS cargado ({len(self.feature_cols['btts'])} features)")
        except Exception as e:
            logger.warning(f"Modelo btts no disponible: {e}")

        self.loaded = True
        logger.info("Modelos cargados exitosamente")

    def predict_result(self, features_df: pd.DataFrame) -> dict:
        """Genera predicciones de resultado (H/D/A)."""
        if 'result' not in self.models:
            return {}

        # Seleccionar y preparar features
        X = features_df[self.feature_cols['result']].copy()
        X = self.imputers['result'].transform(X)
        X = self.scalers['result'].transform(X)

        # Predecir
        y_pred = self.models['result'].predict(X)
        y_proba = self.models['result'].predict_proba(X)

        # Decodificar clases
        classes = self.label_encoder.classes_  # ['A', 'D', 'H']

        return {
            'pred_result': self.label_encoder.inverse_transform(y_pred),
            'prob_away': y_proba[:, list(classes).index('A')],
            'prob_draw': y_proba[:, list(classes).index('D')],
            'prob_home': y_proba[:, list(classes).index('H')]
        }

    def predict_over25(self, features_df: pd.DataFrame) -> dict:
        """Genera predicciones de Over 2.5."""
        if 'over25' not in self.models:
            return {}

        X = features_df[self.feature_cols['over25']].copy()
        X = self.imputers['over25'].transform(X)
        X = self.scalers['over25'].transform(X)

        y_pred = self.models['over25'].predict(X)
        y_proba = self.models['over25'].predict_proba(X)

        return {
            'pred_over25': y_pred,
            'prob_over25': y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        }

    def predict_btts(self, features_df: pd.DataFrame) -> dict:
        """Genera predicciones de BTTS."""
        if 'btts' not in self.models:
            return {}

        X = features_df[self.feature_cols['btts']].copy()
        X = self.imputers['btts'].transform(X)
        X = self.scalers['btts'].transform(X)

        y_pred = self.models['btts'].predict(X)
        y_proba = self.models['btts'].predict_proba(X)

        return {
            'pred_btts': y_pred,
            'prob_btts': y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        }

    def predict_all(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Genera todas las predicciones."""
        if not self.loaded:
            self.load_models()

        results = features_df[['fixture_id']].copy()

        # Predicciones de resultado
        result_preds = self.predict_result(features_df)
        if result_preds:
            results['pred_result'] = result_preds['pred_result']
            results['prob_home'] = result_preds['prob_home']
            results['prob_draw'] = result_preds['prob_draw']
            results['prob_away'] = result_preds['prob_away']

        # Predicciones de Over 2.5
        over25_preds = self.predict_over25(features_df)
        if over25_preds:
            results['pred_over25'] = over25_preds['pred_over25']
            results['prob_over25'] = over25_preds['prob_over25']

        # Predicciones de BTTS
        btts_preds = self.predict_btts(features_df)
        if btts_preds:
            results['pred_btts'] = btts_preds['pred_btts']
            results['prob_btts'] = btts_preds['prob_btts']

        return results


class SupabaseClient:
    """Cliente para Supabase."""

    def __init__(self):
        self.headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type': 'application/json'
        }

    def get_features(self, min_date: str = None, future_only: bool = False) -> pd.DataFrame:
        """Obtiene features de Supabase."""
        logger.info("Obteniendo features de Supabase...")

        all_data = []
        offset = 0
        limit = 1000

        # Construir query base
        base_url = f"{SUPABASE_URL}/rest/v1/fixture_features"

        while True:
            params = {
                'select': '*',
                'order': 'fixture_id.asc',
                'offset': offset,
                'limit': limit
            }

            if min_date:
                params['calculated_at'] = f'gte.{min_date}'

            response = requests.get(
                base_url,
                headers=self.headers,
                params=params,
                verify=False
            )

            if response.status_code != 200:
                logger.error(f"Error obteniendo features: {response.status_code}")
                break

            data = response.json()
            if not data:
                break

            all_data.extend(data)
            logger.info(f"  Obtenidos {len(all_data)} registros...")

            if len(data) < limit:
                break

            offset += limit

        df = pd.DataFrame(all_data)
        logger.info(f"Total features obtenidas: {len(df)}")

        if future_only and 'result' in df.columns:
            df = df[df['result'].isna()]
            logger.info(f"Filtrado a partidos futuros: {len(df)}")

        return df

    def get_fixtures_info(self, fixture_ids: list) -> pd.DataFrame:
        """Obtiene información básica de fixtures."""
        logger.info(f"Obteniendo info de {len(fixture_ids)} fixtures...")

        all_data = []

        # Procesar en batches de 100
        for i in range(0, len(fixture_ids), 100):
            batch_ids = fixture_ids[i:i + 100]
            ids_str = ','.join(str(id) for id in batch_ids)

            response = requests.get(
                f"{SUPABASE_URL}/rest/v1/fixtures",
                headers=self.headers,
                params={
                    'select': 'fixture_id,date,home_team_name,away_team_name,league_name,league_country,odds_home,odds_draw,odds_away,goals_home,goals_away,status_short',
                    'fixture_id': f'in.({ids_str})'
                },
                verify=False
            )

            if response.status_code == 200:
                all_data.extend(response.json())

        return pd.DataFrame(all_data)


def main():
    parser = argparse.ArgumentParser(description='Genera predicciones en batch')
    parser.add_argument('--future-only', action='store_true', help='Solo partidos futuros')
    parser.add_argument('--min-date', type=str, help='Fecha mínima (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=os.path.join(OUTPUTS_DIR, 'predictions.xlsx'), help='Archivo de salida')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PREDICCIÓN EN BATCH")
    logger.info("=" * 60)

    # Inicializar
    db = SupabaseClient()
    predictor = Predictor()
    predictor.load_models()

    # Obtener features
    features_df = db.get_features(min_date=args.min_date, future_only=args.future_only)

    if features_df.empty:
        logger.info("No hay fixtures para predecir")
        return

    # Generar predicciones
    logger.info(f"Generando predicciones para {len(features_df)} fixtures...")
    predictions = predictor.predict_all(features_df)

    # Obtener info adicional de fixtures
    fixtures_info = db.get_fixtures_info(predictions['fixture_id'].tolist())

    # Combinar
    results = predictions.merge(fixtures_info, on='fixture_id', how='left')

    # Ordenar por fecha
    results['date'] = pd.to_datetime(results['date']).dt.tz_localize(None)
    results = results.sort_values('date')

    # Calcular métricas si hay resultados reales
    if 'goals_home' in results.columns and not results['goals_home'].isna().all():
        # Calcular resultado real
        def get_actual_result(row):
            if pd.isna(row['goals_home']) or pd.isna(row['goals_away']):
                return None
            if row['goals_home'] > row['goals_away']:
                return 'H'
            elif row['goals_home'] < row['goals_away']:
                return 'A'
            else:
                return 'D'

        results['actual_result'] = results.apply(get_actual_result, axis=1)

        # Calcular accuracy
        finished = results[results['actual_result'].notna()]
        if len(finished) > 0:
            correct = (finished['pred_result'] == finished['actual_result']).sum()
            accuracy = correct / len(finished)
            logger.info(f"\nAccuracy en partidos terminados: {accuracy:.2%} ({correct}/{len(finished)})")

    # Guardar resultados
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Formatear columnas de probabilidad
    prob_cols = ['prob_home', 'prob_draw', 'prob_away', 'prob_over25', 'prob_btts']
    for col in prob_cols:
        if col in results.columns:
            results[col] = results[col].round(4)

    # Seleccionar columnas para el output
    output_cols = ['fixture_id', 'date', 'league_name', 'league_country',
                   'home_team_name', 'away_team_name',
                   'pred_result', 'prob_home', 'prob_draw', 'prob_away',
                   'pred_over25', 'prob_over25', 'pred_btts', 'prob_btts',
                   'odds_home', 'odds_draw', 'odds_away',
                   'goals_home', 'goals_away', 'status_short']

    output_cols = [c for c in output_cols if c in results.columns]
    results_output = results[output_cols]

    # Guardar Excel
    results_output.to_excel(args.output, index=False)
    logger.info(f"\nPredicciones guardadas en: {args.output}")

    # Resumen
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)
    logger.info(f"Total predicciones: {len(results)}")

    if 'status_short' in results.columns:
        terminados = (results['status_short'] == 'FT').sum()
        pendientes = (results['status_short'] == 'NS').sum()
        logger.info(f"Partidos terminados: {terminados}")
        logger.info(f"Partidos pendientes: {pendientes}")

    if 'pred_result' in results.columns:
        logger.info(f"\nDistribución predicciones:")
        logger.info(results['pred_result'].value_counts().to_string())

    return results


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()
