#!/usr/bin/env python3
"""
05_train_hybrid_model.py
Entrena modelos de predicción usando los datos híbridos (features + cuotas).

Modelos:
- XGBoost para resultado (H/D/A)
- LightGBM para resultado (H/D/A)
- Comparación con modelo anterior (solo features)

División temporal:
- TRAIN: 2020-01-01 → 2024-12-31
- VALIDATION: 2025-01-01 → 2025-05-31
- TEST: 2025-06-01 → 2025-12-31

Uso:
    python 05_train_hybrid_model.py
    python 05_train_hybrid_model.py --compare  # Comparar con modelo original
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, log_loss
)

# Suprimir warnings
warnings.filterwarnings('ignore')

# XGBoost y LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost no instalado. Instalar con: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM no instalado. Instalar con: pip install lightgbm")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
HYBRID_DB_PATH = os.path.join(DATA_DIR, 'hybrid.db')

# Fechas de división
TRAIN_END = '2024-12-31'
VAL_END = '2025-05-31'

# Features a excluir (metadata, targets, etc.)
EXCLUDE_FEATURES = [
    'fixture_id', 'date', 'league_name', 'league_country',
    'home_team_name', 'away_team_name', 'goals_home', 'goals_away',
    'result', 'home_goals', 'away_goals', 'total_goals', 'btts',
    'over15', 'over25', 'over35', 'calculated_at', 'features_version',
    'mapping_score', 'mapping_method'
]


class HybridModelTrainer:
    """Entrena modelos con datos híbridos."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.df = None
        self.feature_cols = None
        self.label_encoder = None

    def connect(self):
        """Conecta a la base de datos."""
        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"Conectado a {self.db_path}")

    def close(self):
        """Cierra la conexión."""
        if self.conn:
            self.conn.close()

    def load_data(self) -> pd.DataFrame:
        """Carga los datos de entrenamiento."""
        logger.info("Cargando datos de training_data...")

        self.df = pd.read_sql('SELECT * FROM training_data ORDER BY date', self.conn)
        logger.info(f"Cargados {len(self.df):,} registros")

        # Identificar features
        self.feature_cols = [col for col in self.df.columns if col not in EXCLUDE_FEATURES]
        logger.info(f"Features identificadas: {len(self.feature_cols)}")

        return self.df

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divide los datos temporalmente."""
        train = self.df[self.df['date'] <= TRAIN_END].copy()
        val = self.df[(self.df['date'] > TRAIN_END) & (self.df['date'] <= VAL_END)].copy()
        test = self.df[self.df['date'] > VAL_END].copy()

        logger.info(f"\nDivisión temporal:")
        logger.info(f"  TRAIN: {len(train):,} ({train['date'].min()} → {train['date'].max()})")
        logger.info(f"  VAL: {len(val):,} ({val['date'].min()} → {val['date'].max()})")
        logger.info(f"  TEST: {len(test):,} ({test['date'].min()} → {test['date'].max()})")

        return train, val, test

    def prepare_features(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Prepara features para entrenamiento."""

        X_train = train[self.feature_cols].copy()
        X_val = val[self.feature_cols].copy()
        X_test = test[self.feature_cols].copy()

        y_train = train['result'].copy()
        y_val = val['result'].copy()
        y_test = test['result'].copy()

        # Imputar valores faltantes
        logger.info("Imputando valores faltantes...")
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        X_test_imp = imputer.transform(X_test)

        # Escalar
        logger.info("Escalando features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # Encodear labels
        self.label_encoder = LabelEncoder()
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        y_test_enc = self.label_encoder.transform(y_test)

        logger.info(f"Clases: {list(self.label_encoder.classes_)}")

        return (
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_enc, y_val_enc, y_test_enc,
            imputer, scaler
        )

    def train_xgboost(self, X_train, X_val, y_train, y_val) -> XGBClassifier:
        """Entrena modelo XGBoost."""
        if not HAS_XGB:
            return None

        logger.info("\nEntrenando XGBoost...")

        # Calcular sample weights para balance de clases
        class_counts = np.bincount(y_train)
        total = len(y_train)
        weights = total / (len(class_counts) * class_counts)
        sample_weights = weights[y_train]

        model = XGBClassifier(
            max_depth=4,
            min_child_weight=5,
            n_estimators=500,
            gamma=1.0,
            reg_alpha=1.0,
            reg_lambda=1.5,
            subsample=0.7,
            colsample_bytree=0.7,
            learning_rate=0.05,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='mlogloss'
        )

        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        logger.info(f"Mejor iteración: {model.best_iteration}")
        return model

    def train_lightgbm(self, X_train, X_val, y_train, y_val) -> LGBMClassifier:
        """Entrena modelo LightGBM."""
        if not HAS_LGB:
            return None

        logger.info("\nEntrenando LightGBM...")

        model = LGBMClassifier(
            max_depth=4,
            num_leaves=15,
            min_data_in_leaf=50,
            n_estimators=500,
            lambda_l1=1.0,
            lambda_l2=1.5,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=1,
            learning_rate=0.05,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                # Early stopping callback
            ]
        )

        return model

    def evaluate_model(self, model, X, y_true, set_name: str) -> Dict:
        """Evalúa un modelo."""
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'log_loss': log_loss(y_true, y_proba)
        }

        logger.info(f"\n{set_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")

        return metrics

    def get_feature_importance(self, model, top_n: int = 20) -> List[Tuple[str, float]]:
        """Obtiene las features más importantes."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            return [(self.feature_cols[i], importances[i]) for i in indices]
        return []

    def save_model(self, model, imputer, scaler, model_name: str, metrics: Dict):
        """Guarda modelo y artefactos."""
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Guardar modelo
        model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Modelo guardado: {model_path}")

        # Guardar imputer
        imputer_path = os.path.join(MODELS_DIR, f'imputer_hybrid.pkl')
        joblib.dump(imputer, imputer_path)

        # Guardar scaler
        scaler_path = os.path.join(MODELS_DIR, f'scaler_hybrid.pkl')
        joblib.dump(scaler, scaler_path)

        # Guardar label encoder
        encoder_path = os.path.join(MODELS_DIR, f'label_encoder_hybrid.pkl')
        joblib.dump(self.label_encoder, encoder_path)

        # Guardar feature columns
        cols_path = os.path.join(MODELS_DIR, f'feature_cols_hybrid.json')
        with open(cols_path, 'w') as f:
            json.dump(self.feature_cols, f)

        # Guardar métricas
        metrics_path = os.path.join(MODELS_DIR, f'metrics_{model_name}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def run_training(self):
        """Ejecuta el entrenamiento completo."""

        # Cargar datos
        self.load_data()

        # Dividir
        train, val, test = self.split_data()

        # Preparar features
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         imputer, scaler) = self.prepare_features(train, val, test)

        results = {}

        # Entrenar XGBoost
        if HAS_XGB:
            xgb_model = self.train_xgboost(X_train, X_val, y_train, y_val)

            if xgb_model:
                logger.info("\n" + "=" * 40)
                logger.info("EVALUACIÓN XGBOOST")
                logger.info("=" * 40)

                train_metrics = self.evaluate_model(xgb_model, X_train, y_train, "TRAIN")
                val_metrics = self.evaluate_model(xgb_model, X_val, y_val, "VALIDATION")
                test_metrics = self.evaluate_model(xgb_model, X_test, y_test, "TEST")

                results['xgb'] = {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                }

                # Feature importance
                logger.info("\nTop 20 Features:")
                for feat, imp in self.get_feature_importance(xgb_model):
                    logger.info(f"  {feat}: {imp:.4f}")

                # Guardar
                self.save_model(xgb_model, imputer, scaler, 'xgb_hybrid', results['xgb'])

        # Entrenar LightGBM
        if HAS_LGB:
            lgb_model = self.train_lightgbm(X_train, X_val, y_train, y_val)

            if lgb_model:
                logger.info("\n" + "=" * 40)
                logger.info("EVALUACIÓN LIGHTGBM")
                logger.info("=" * 40)

                train_metrics = self.evaluate_model(lgb_model, X_train, y_train, "TRAIN")
                val_metrics = self.evaluate_model(lgb_model, X_val, y_val, "VALIDATION")
                test_metrics = self.evaluate_model(lgb_model, X_test, y_test, "TEST")

                results['lgb'] = {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                }

                self.save_model(lgb_model, imputer, scaler, 'lgb_hybrid', results['lgb'])

        # Resumen final
        logger.info("\n" + "=" * 60)
        logger.info("RESUMEN FINAL")
        logger.info("=" * 60)

        for model_name, model_results in results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Test Accuracy: {model_results['test']['accuracy']:.4f}")
            logger.info(f"  Test Balanced Acc: {model_results['test']['balanced_accuracy']:.4f}")
            gap = model_results['train']['accuracy'] - model_results['test']['accuracy']
            logger.info(f"  Overfitting Gap: {gap:.4f}")

        # Comparación con features de cuotas
        logger.info("\n" + "=" * 60)
        logger.info("ANÁLISIS DE FEATURES DE CUOTAS")
        logger.info("=" * 60)

        odds_features = [f for f in self.feature_cols if 'odds' in f.lower() or 'implied' in f.lower()]
        logger.info(f"Features de cuotas usadas: {len(odds_features)}")
        for f in odds_features:
            logger.info(f"  - {f}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo híbrido')
    parser.add_argument('--compare', action='store_true',
                        help='Comparar con modelo original')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO DE MODELO HÍBRIDO")
    logger.info("=" * 60)

    if not HAS_XGB and not HAS_LGB:
        logger.error("Necesitas instalar XGBoost o LightGBM")
        sys.exit(1)

    trainer = HybridModelTrainer(HYBRID_DB_PATH)

    try:
        trainer.connect()
        trainer.run_training()
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
