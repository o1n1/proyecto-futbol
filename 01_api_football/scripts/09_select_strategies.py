"""
Script para seleccionar estrategias rentables basándose en accuracy.

FASE 2 del análisis de estrategia de apuestas.
Filtra las combinaciones de calibration_results.xlsx por criterios de accuracy.

NO filtra por EV ni cuotas - solo accuracy.
"""

import pandas as pd
import os
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURACIÓN
# ============================================

# Rutas relativas al directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')

# Criterios de filtrado
FILTROS = {
    'min_partidos': 10,      # Mínimo partidos después de aplicar umbral
    'min_accuracy': 0.55,    # Accuracy mínima (ajustable)
}


# ============================================
# FILTRADO
# ============================================

def load_calibration_results() -> pd.DataFrame:
    """Carga los resultados de calibración."""
    input_path = os.path.join(OUTPUTS_DIR, 'calibration_results.xlsx')
    df = pd.read_excel(input_path)
    logger.info(f"Cargadas {len(df):,} combinaciones de calibración")
    return df


def filter_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra estrategias por criterios de accuracy."""
    logger.info(f"Filtrando con: min_partidos={FILTROS['min_partidos']}, min_accuracy={FILTROS['min_accuracy']}")

    # Aplicar filtros
    df_filtered = df[
        (df['n_partidos'] >= FILTROS['min_partidos']) &
        (df['accuracy'] >= FILTROS['min_accuracy'])
    ].copy()

    logger.info(f"Estrategias que pasan filtros: {len(df_filtered):,}")

    return df_filtered


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de análisis adicionales."""
    df = df.copy()

    # Crear identificador único de estrategia
    df['strategy_id'] = df.apply(
        lambda x: f"{x['league_id']}_{x['tipo_apuesta']}_{x['umbral']}",
        axis=1
    )

    # Marcar nivel de confianza basado en n_partidos
    def confidence_level(n):
        if n >= 30:
            return 'Alto'
        elif n >= 20:
            return 'Medio'
        else:
            return 'Bajo'

    df['confidence_level'] = df['n_partidos'].apply(confidence_level)

    return df


# ============================================
# MAIN
# ============================================

def main():
    """Ejecuta la selección de estrategias."""
    logger.info("=" * 60)
    logger.info("FASE 2: Selección de Estrategias")
    logger.info("=" * 60)

    # 1. Cargar resultados de calibración
    df = load_calibration_results()

    # 2. Filtrar por criterios
    df_filtered = filter_strategies(df)

    if len(df_filtered) == 0:
        logger.warning("No hay estrategias que pasen los filtros!")
        return None

    # 3. Agregar columnas de análisis
    df_filtered = add_analysis_columns(df_filtered)

    # 4. Ordenar por accuracy
    df_filtered = df_filtered.sort_values(['accuracy', 'n_partidos'], ascending=[False, False])

    # 5. Guardar
    output_path = os.path.join(OUTPUTS_DIR, 'selected_strategies.xlsx')
    df_filtered.to_excel(output_path, index=False)
    logger.info(f"Estrategias guardadas en: {output_path}")

    # Resumen
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)
    logger.info(f"Total estrategias seleccionadas: {len(df_filtered):,}")

    # Por tipo de apuesta
    logger.info("\nPor tipo de apuesta:")
    for tipo in df_filtered['tipo_apuesta'].unique():
        count = len(df_filtered[df_filtered['tipo_apuesta'] == tipo])
        logger.info(f"  {tipo}: {count}")

    # Por nivel de confianza
    logger.info("\nPor nivel de confianza:")
    for nivel in ['Alto', 'Medio', 'Bajo']:
        count = len(df_filtered[df_filtered['confidence_level'] == nivel])
        logger.info(f"  {nivel}: {count}")

    # Top 20 estrategias
    logger.info("\nTop 20 estrategias:")
    top20 = df_filtered.head(20)
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        logger.info(f"  {i:2d}. {row['league_name'][:25]:25s} | {row['tipo_apuesta']:6s} >= {row['umbral']:.2f} | "
                   f"Acc: {row['accuracy']:.1%} | N: {row['n_partidos']:3d} | Conf: {row['confidence_level']}")

    return df_filtered


if __name__ == "__main__":
    main()
