"""
Script para entrenar modelos de predicción de resultados de fútbol.

Características:
- División temporal de datos (2020-2024 train, 2025 test)
- Ensemble de XGBoost + LightGBM
- Manejo de NULLs con imputación
- Early stopping para evitar overfitting
- Métricas: Accuracy, Macro F1, Log Loss

Targets:
- result: H (Home Win), A (Away Win), D (Draw)
- over25: True/False (más de 2.5 goles)
- btts: True/False (ambos equipos marcan)
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import urllib3

# Modelos
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

# Preprocesamiento
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Evaluación
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    classification_report, confusion_matrix,
    balanced_accuracy_score, precision_recall_fscore_support
)

# Balanceo de clases
from sklearn.utils.class_weight import compute_class_weight

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

# Fechas de corte para división temporal
TRAIN_END_DATE = '2024-12-31'
VAL_END_DATE = '2025-05-31'

# Columnas a excluir del entrenamiento
EXCLUDE_COLS = [
    'fixture_id', 'calculated_at', 'features_version',
    # Targets
    'result', 'total_goals', 'home_goals', 'away_goals',
    'btts', 'over15', 'over25', 'over35'
]

# Rutas relativas al directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')


# ============================================
# CARGA DE DATOS
# ============================================

def fetch_features_from_supabase() -> pd.DataFrame:
    """Carga todas las features desde Supabase con paginación."""
    logger.info("Cargando features desde Supabase...")

    url = f"{SUPABASE_URL}/rest/v1/fixture_features"
    all_data = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            'select': '*',
            'order': 'fixture_id.asc',
            'offset': offset,
            'limit': page_size
        }

        response = requests.get(url, headers=HEADERS, params=params, verify=False, timeout=60)

        if response.status_code != 200:
            raise Exception(f"Error al obtener features: {response.text}")

        data = response.json()
        if not data:
            break

        all_data.extend(data)
        offset += page_size

        if len(data) < page_size:
            break

    df = pd.DataFrame(all_data)
    logger.info(f"Cargados {len(df):,} registros de features")

    return df


def fetch_fixture_dates() -> pd.DataFrame:
    """Obtiene las fechas de los fixtures para división temporal."""
    logger.info("Cargando fechas de fixtures...")

    url = f"{SUPABASE_URL}/rest/v1/fixtures"
    all_data = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            'select': 'fixture_id,date',
            'order': 'fixture_id.asc',
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

    return df


# ============================================
# PREPROCESAMIENTO
# ============================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Obtiene las columnas de features (excluyendo targets y metadatos)."""
    return [col for col in df.columns if col not in EXCLUDE_COLS]


def preprocess_data(df: pd.DataFrame, target: str = 'result') -> Tuple:
    """
    Preprocesa los datos para entrenamiento.

    Returns:
        X: Features
        y: Target encoded
        feature_cols: Lista de nombres de features
        imputer: Imputer ajustado
        scaler: Scaler ajustado
        label_encoder: LabelEncoder ajustado (solo para result)
    """
    logger.info(f"Preprocesando datos para target: {target}")

    # Filtrar solo partidos con target disponible (terminados)
    df_valid = df[df[target].notna()].copy()
    logger.info(f"Partidos con {target} disponible: {len(df_valid):,}")

    # Seleccionar features
    feature_cols = get_feature_columns(df)
    X = df_valid[feature_cols].copy()

    # Target
    if target == 'result':
        le = LabelEncoder()
        y = le.fit_transform(df_valid[target])  # A=0, D=1, H=2
        logger.info(f"Clases: {dict(zip(le.classes_, range(len(le.classes_))))}")
    else:
        le = None
        y = df_valid[target].astype(int).values

    # Separar columnas numéricas y booleanas
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()

    # Convertir booleanos a int
    for col in bool_cols:
        X[col] = X[col].astype(float)

    # Imputación con mediana
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    logger.info(f"Features finales: {len(feature_cols)}")
    logger.info(f"Samples: {len(y):,}")

    return X_scaled, y, feature_cols, imputer, scaler, le, df_valid['fixture_id'].values


def temporal_split(fixture_ids: np.ndarray, dates_df: pd.DataFrame,
                   X: np.ndarray, y: np.ndarray) -> Tuple:
    """
    Divide los datos temporalmente.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Dividiendo datos temporalmente...")

    # Crear DataFrame con fixture_ids
    df_split = pd.DataFrame({'fixture_id': fixture_ids})
    df_split = df_split.merge(dates_df, on='fixture_id', how='left')

    # Máscaras temporales
    train_mask = df_split['date'] <= TRAIN_END_DATE
    val_mask = (df_split['date'] > TRAIN_END_DATE) & (df_split['date'] <= VAL_END_DATE)
    test_mask = df_split['date'] > VAL_END_DATE

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    logger.info(f"TRAIN: {len(y_train):,} ({len(y_train)/len(y)*100:.1f}%) - hasta {TRAIN_END_DATE}")
    logger.info(f"VAL:   {len(y_val):,} ({len(y_val)/len(y)*100:.1f}%) - {TRAIN_END_DATE} a {VAL_END_DATE}")
    logger.info(f"TEST:  {len(y_test):,} ({len(y_test)/len(y)*100:.1f}%) - desde {VAL_END_DATE}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================
# ENTRENAMIENTO
# ============================================

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  n_classes: int = 3) -> XGBClassifier:
    """Entrena modelo XGBoost con early stopping y class weighting."""
    logger.info("Entrenando XGBoost...")

    # Calcular sample weights para balanceo de clases
    class_weights = compute_class_weight('balanced',
                                          classes=np.unique(y_train),
                                          y=y_train)
    sample_weight = np.array([class_weights[y] for y in y_train])

    logger.info(f"Class weights calculados: {dict(zip(np.unique(y_train), class_weights.round(3)))}")

    xgb = XGBClassifier(
        # Arquitectura - AJUSTADA para reducir overfitting
        max_depth=4,              # v2.1: Reducido de 5 → menos overfitting
        min_child_weight=5,       # v2.1: Aumentado de 3 → más restrictivo
        n_estimators=500,         # v2.1: Reducido de 1000 (early stopping lo cortará)
        gamma=1.0,                # v2.1: Aumentado de 0.5 → más regularización

        # Regularización - MÁS FUERTE
        reg_alpha=1.0,            # v2.1: Aumentado de 0.5
        reg_lambda=1.5,           # v2.1: Aumentado de 1.0
        subsample=0.7,            # v2.1: Reducido de 0.8
        colsample_bytree=0.7,     # v2.1: Reducido de 0.8

        # Learning
        learning_rate=0.05,       # v2.1: Aumentado de 0.03 (converge más rápido con early stop)

        # Early stopping
        early_stopping_rounds=50, # v2.1: NUEVO - detiene si no mejora en 50 rounds

        # Otros
        objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
        num_class=n_classes if n_classes > 2 else None,
        eval_metric='mlogloss' if n_classes > 2 else 'logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Best iteration solo disponible con early stopping
    if hasattr(xgb, 'best_iteration') and xgb.get_booster().best_iteration > 0:
        logger.info(f"XGBoost - Best iteration: {xgb.best_iteration}")
    else:
        logger.info(f"XGBoost - Entrenado con {xgb.n_estimators} estimators")

    return xgb


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   n_classes: int = 3) -> LGBMClassifier:
    """Entrena modelo LightGBM con early stopping y balanceo de clases."""
    logger.info("Entrenando LightGBM...")

    # Importar callbacks de LightGBM
    from lightgbm import early_stopping, log_evaluation

    lgb = LGBMClassifier(
        # Arquitectura - AJUSTADA para reducir overfitting
        max_depth=4,              # v2.1: Reducido de 6
        num_leaves=15,            # v2.1: Reducido de 25 (2^4 - 1)
        min_data_in_leaf=50,      # v2.1: Aumentado de 30
        n_estimators=500,         # v2.1: Reducido de 1000

        # Regularización - MÁS FUERTE
        lambda_l1=1.0,            # v2.1: Aumentado de 0.5
        lambda_l2=1.5,            # v2.1: Aumentado de 1.0
        feature_fraction=0.7,     # v2.1: Reducido de 0.8
        bagging_fraction=0.7,     # v2.1: Reducido de 0.8
        bagging_freq=1,

        # Learning
        learning_rate=0.05,       # v2.1: Aumentado de 0.03

        # Balanceo de clases
        is_unbalance=True,

        # Otros
        objective='multiclass' if n_classes > 2 else 'binary',
        num_class=n_classes if n_classes > 2 else None,
        metric='multi_logloss' if n_classes > 2 else 'binary_logloss',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),  # v2.1: Early stopping
            log_evaluation(period=50)
        ]
    )

    # Best iteration
    if hasattr(lgb, 'best_iteration_') and lgb.best_iteration_ is not None:
        logger.info(f"LightGBM - Best iteration: {lgb.best_iteration_}")
    else:
        logger.info(f"LightGBM - Entrenado con {lgb.n_estimators} estimators")

    return lgb


def create_ensemble(xgb_model: XGBClassifier, lgb_model: LGBMClassifier) -> VotingClassifier:
    """Crea ensemble de votación suave."""
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        voting='soft',
        weights=[0.6, 0.4]  # Dar más peso a XGBoost
    )
    return ensemble


# ============================================
# EVALUACIÓN
# ============================================

def evaluate_model(model, X: np.ndarray, y: np.ndarray,
                   label_encoder: Optional[LabelEncoder] = None,
                   set_name: str = "Test") -> Dict:
    """Evalúa el modelo y retorna métricas detalladas."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Métricas principales
    accuracy = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)  # ← NUEVA MÉTRICA PRINCIPAL
    macro_f1 = f1_score(y, y_pred, average='macro')
    weighted_f1 = f1_score(y, y_pred, average='weighted')
    logloss = log_loss(y, y_proba)

    # Métricas por clase
    precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred)

    logger.info(f"\n{'='*50}")
    logger.info(f"MÉTRICAS EN {set_name.upper()}")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy:         {accuracy:.4f}")
    logger.info(f"Balanced Acc:     {balanced_acc:.4f}  ← MÉTRICA PRINCIPAL")
    logger.info(f"Macro F1:         {macro_f1:.4f}")
    logger.info(f"Weighted F1:      {weighted_f1:.4f}")
    logger.info(f"Log Loss:         {logloss:.4f}")

    # Métricas por clase (importante para ver recall de Draw)
    if label_encoder is not None:
        target_names = label_encoder.classes_
        logger.info(f"\nMétricas por clase:")
        for i, name in enumerate(target_names):
            logger.info(f"  {name}: Precision={precision[i]:.3f} Recall={recall[i]:.3f} F1={fscore[i]:.3f} Support={support[i]}")
    else:
        target_names = None
        logger.info(f"\nMétricas por clase:")
        for i in range(len(precision)):
            logger.info(f"  Clase {i}: Precision={precision[i]:.3f} Recall={recall[i]:.3f} F1={fscore[i]:.3f} Support={support[i]}")

    # Classification report completo
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'log_loss': logloss,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': fscore.tolist(),
        'confusion_matrix': cm.tolist()
    }


def check_overfitting(train_metrics: Dict, val_metrics: Dict, test_metrics: Dict):
    """Verifica si hay overfitting."""
    logger.info("\n" + "="*50)
    logger.info("ANÁLISIS DE OVERFITTING")
    logger.info("="*50)

    train_acc = train_metrics['accuracy']
    val_acc = val_metrics['accuracy']
    test_acc = test_metrics['accuracy']

    train_val_gap = train_acc - val_acc
    train_test_gap = train_acc - test_acc
    val_test_gap = val_acc - test_acc

    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Val Accuracy:   {val_acc:.4f}")
    logger.info(f"Test Accuracy:  {test_acc:.4f}")
    logger.info(f"")
    logger.info(f"Gap Train-Val:  {train_val_gap:.4f} {'⚠️ OVERFITTING' if train_val_gap > 0.05 else '✅ OK'}")
    logger.info(f"Gap Train-Test: {train_test_gap:.4f} {'⚠️ OVERFITTING' if train_test_gap > 0.08 else '✅ OK'}")
    logger.info(f"Gap Val-Test:   {val_test_gap:.4f} {'⚠️ DRIFT' if abs(val_test_gap) > 0.03 else '✅ OK'}")


def get_feature_importance(model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Obtiene importancia de features."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_booster'):
        booster = model.get_booster()
        importance_dict = booster.get_score(importance_type='gain')
        importances = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            if name in importance_dict:
                importances[i] = importance_dict[name]
    else:
        return None

    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return df_importance.head(top_n)


# ============================================
# GUARDAR MODELOS
# ============================================

def save_models(xgb_model: XGBClassifier, lgb_model: LGBMClassifier,
                imputer: SimpleImputer, scaler: StandardScaler,
                label_encoder: Optional[LabelEncoder],
                feature_cols: List[str], metrics: Dict,
                target: str = 'result'):
    """Guarda modelos y artefactos."""

    # Crear directorio si no existe
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Guardar modelos
    joblib.dump(xgb_model, f'{MODELS_DIR}/xgb_{target}_model.pkl')
    joblib.dump(lgb_model, f'{MODELS_DIR}/lgb_{target}_model.pkl')

    # Guardar preprocesadores
    joblib.dump(imputer, f'{MODELS_DIR}/imputer_{target}.pkl')
    joblib.dump(scaler, f'{MODELS_DIR}/scaler_{target}.pkl')

    if label_encoder is not None:
        joblib.dump(label_encoder, f'{MODELS_DIR}/label_encoder_{target}.pkl')

    # Guardar lista de features
    with open(f'{MODELS_DIR}/feature_cols_{target}.json', 'w') as f:
        json.dump(feature_cols, f)

    # Guardar métricas
    metrics['trained_at'] = datetime.now().isoformat()
    metrics['target'] = target
    with open(f'{MODELS_DIR}/metrics_{target}.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nModelos guardados en {MODELS_DIR}/")


# ============================================
# PIPELINE PRINCIPAL
# ============================================

def train_result_model(df_features: pd.DataFrame, dates_df: pd.DataFrame) -> Dict:
    """Pipeline completo para entrenar modelo de resultado (H/D/A)."""
    logger.info("\n" + "="*60)
    logger.info("ENTRENAMIENTO: PREDICCIÓN DE RESULTADO (H/D/A)")
    logger.info("="*60)

    # 1. Preprocesamiento
    X, y, feature_cols, imputer, scaler, le, fixture_ids = preprocess_data(df_features, target='result')

    # 2. División temporal
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
        fixture_ids, dates_df, X, y
    )

    # 3. Entrenar modelos
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, n_classes=3)
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, n_classes=3)

    # 4. Evaluación
    logger.info("\n--- XGBoost ---")
    xgb_train_metrics = evaluate_model(xgb_model, X_train, y_train, le, "Train")
    xgb_val_metrics = evaluate_model(xgb_model, X_val, y_val, le, "Validation")
    xgb_test_metrics = evaluate_model(xgb_model, X_test, y_test, le, "Test")

    logger.info("\n--- LightGBM ---")
    lgb_train_metrics = evaluate_model(lgb_model, X_train, y_train, le, "Train")
    lgb_val_metrics = evaluate_model(lgb_model, X_val, y_val, le, "Validation")
    lgb_test_metrics = evaluate_model(lgb_model, X_test, y_test, le, "Test")

    # 5. Verificar overfitting
    logger.info("\n--- Análisis XGBoost ---")
    check_overfitting(xgb_train_metrics, xgb_val_metrics, xgb_test_metrics)

    logger.info("\n--- Análisis LightGBM ---")
    check_overfitting(lgb_train_metrics, lgb_val_metrics, lgb_test_metrics)

    # 6. Feature importance
    logger.info("\n" + "="*50)
    logger.info("TOP 20 FEATURES (XGBoost)")
    logger.info("="*50)
    importance_df = get_feature_importance(xgb_model, feature_cols, top_n=20)
    if importance_df is not None:
        for _, row in importance_df.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # 7. Guardar modelos
    metrics = {
        'xgb_test': xgb_test_metrics,
        'lgb_test': lgb_test_metrics,
        'train_size': len(y_train),
        'val_size': len(y_val),
        'test_size': len(y_test)
    }

    save_models(xgb_model, lgb_model, imputer, scaler, le, feature_cols, metrics, 'result')

    return metrics


def train_over25_model(df_features: pd.DataFrame, dates_df: pd.DataFrame) -> Dict:
    """Pipeline para entrenar modelo de Over 2.5 goles."""
    logger.info("\n" + "="*60)
    logger.info("ENTRENAMIENTO: PREDICCIÓN OVER 2.5")
    logger.info("="*60)

    # 1. Preprocesamiento
    X, y, feature_cols, imputer, scaler, _, fixture_ids = preprocess_data(df_features, target='over25')

    # 2. División temporal
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
        fixture_ids, dates_df, X, y
    )

    # 3. Entrenar modelos (binario)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, n_classes=2)
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, n_classes=2)

    # 4. Evaluación
    logger.info("\n--- XGBoost ---")
    xgb_test_metrics = evaluate_model(xgb_model, X_test, y_test, None, "Test")

    logger.info("\n--- LightGBM ---")
    lgb_test_metrics = evaluate_model(lgb_model, X_test, y_test, None, "Test")

    # 5. Guardar modelos
    metrics = {
        'xgb_test': xgb_test_metrics,
        'lgb_test': lgb_test_metrics,
        'train_size': len(y_train),
        'val_size': len(y_val),
        'test_size': len(y_test)
    }

    save_models(xgb_model, lgb_model, imputer, scaler, None, feature_cols, metrics, 'over25')

    return metrics


def train_btts_model(df_features: pd.DataFrame, dates_df: pd.DataFrame) -> Dict:
    """Pipeline para entrenar modelo de BTTS."""
    logger.info("\n" + "="*60)
    logger.info("ENTRENAMIENTO: PREDICCIÓN BTTS")
    logger.info("="*60)

    # 1. Preprocesamiento
    X, y, feature_cols, imputer, scaler, _, fixture_ids = preprocess_data(df_features, target='btts')

    # 2. División temporal
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
        fixture_ids, dates_df, X, y
    )

    # 3. Entrenar modelos (binario)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, n_classes=2)
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, n_classes=2)

    # 4. Evaluación
    logger.info("\n--- XGBoost ---")
    xgb_test_metrics = evaluate_model(xgb_model, X_test, y_test, None, "Test")

    logger.info("\n--- LightGBM ---")
    lgb_test_metrics = evaluate_model(lgb_model, X_test, y_test, None, "Test")

    # 5. Guardar modelos
    metrics = {
        'xgb_test': xgb_test_metrics,
        'lgb_test': lgb_test_metrics,
        'train_size': len(y_train),
        'val_size': len(y_val),
        'test_size': len(y_test)
    }

    save_models(xgb_model, lgb_model, imputer, scaler, None, feature_cols, metrics, 'btts')

    return metrics


# ============================================
# MAIN
# ============================================

def main():
    """Ejecuta el pipeline completo de entrenamiento."""
    logger.info("="*60)
    logger.info("INICIO DEL ENTRENAMIENTO DE MODELOS")
    logger.info("="*60)

    start_time = datetime.now()

    # 1. Cargar datos
    df_features = fetch_features_from_supabase()
    dates_df = fetch_fixture_dates()

    # 2. Entrenar modelo de resultado (H/D/A)
    result_metrics = train_result_model(df_features, dates_df)

    # 3. Entrenar modelo de Over 2.5
    over25_metrics = train_over25_model(df_features, dates_df)

    # 4. Entrenar modelo de BTTS
    btts_metrics = train_btts_model(df_features, dates_df)

    # Resumen final
    elapsed = datetime.now() - start_time
    logger.info("\n" + "="*60)
    logger.info("RESUMEN FINAL")
    logger.info("="*60)
    logger.info(f"Tiempo total: {elapsed}")
    logger.info(f"\nModelo RESULT (H/D/A):")
    logger.info(f"  XGBoost:")
    logger.info(f"    Accuracy:         {result_metrics['xgb_test']['accuracy']:.4f}")
    logger.info(f"    Balanced Acc:     {result_metrics['xgb_test']['balanced_accuracy']:.4f}")
    logger.info(f"    Macro F1:         {result_metrics['xgb_test']['macro_f1']:.4f}")
    logger.info(f"  LightGBM:")
    logger.info(f"    Accuracy:         {result_metrics['lgb_test']['accuracy']:.4f}")
    logger.info(f"    Balanced Acc:     {result_metrics['lgb_test']['balanced_accuracy']:.4f}")
    logger.info(f"    Macro F1:         {result_metrics['lgb_test']['macro_f1']:.4f}")
    logger.info(f"\nModelo OVER 2.5:")
    logger.info(f"  XGBoost Test Accuracy: {over25_metrics['xgb_test']['accuracy']:.4f}")
    logger.info(f"  LightGBM Test Accuracy: {over25_metrics['lgb_test']['accuracy']:.4f}")
    logger.info(f"\nModelo BTTS:")
    logger.info(f"  XGBoost Test Accuracy: {btts_metrics['xgb_test']['accuracy']:.4f}")
    logger.info(f"  LightGBM Test Accuracy: {btts_metrics['lgb_test']['accuracy']:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
