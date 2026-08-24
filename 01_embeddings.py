"""Fase 3 - Paso 1: generacion de embeddings semanticos de ambas agendas.

Este script es el punto de partida del analisis de brechas de agenda: convierte
el corpus institucional (Plan Decenal de Salud Publica y politicas asociadas) y
el corpus social (titulares de prensa filtrados por salud) al mismo espacio
vectorial semantico, para poder compararlos posteriormente mediante similitud
del coseno y modelado de topicos (BERTopic).

Dependencias necesarias:
- pandas
- pyarrow
- numpy
- sentence-transformers (y su dependencia torch)

Comando de instalacion:
pip install pandas pyarrow numpy sentence-transformers

Entradas:
- data/processed/corpus_institucional.csv (columna de texto: "texto")
- data/processed/corpus_social_salud.csv   (columna de texto: "titulo")

Salidas:
- data/processed/emb_institucional.npy  (matriz de embeddings, float32)
- data/processed/emb_social.npy         (matriz de embeddings, float32)
- data/processed/orden_institucional.parquet (id_doc + texto, mismo orden que
  emb_institucional.npy)
- data/processed/orden_social.parquet        (id_art + titulo, mismo orden que
  emb_social.npy)
- outputs/reporte_embeddings.txt (resumen del paso ejecutado)

El orden de las filas en cada archivo "orden_*" coincide exactamente con el
orden de las filas de su matriz de embeddings correspondiente, lo que permite
alinear ambos corpus en pasos posteriores del pipeline (por ejemplo, al medir
similitud del coseno entre agendas) sin depender de un merge por indice.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

NOMBRE_MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE_SOCIAL = 64
BATCH_SIZE_INSTITUCIONAL = 32

COLUMNA_TEXTO_SOCIAL = "titulo"
COLUMNA_ID_SOCIAL = "id_art"
COLUMNA_TEXTO_INSTITUCIONAL = "texto"
COLUMNA_ID_INSTITUCIONAL = "id_doc"


def configurar_logger() -> logging.Logger:
    """Configura el logger principal del script."""
    logger = logging.getLogger("fase3_embeddings")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    manejador = logging.StreamHandler()
    manejador.setLevel(logging.INFO)
    manejador.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(manejador)
    return logger


def cargar_corpus(
    ruta_csv: Path,
    columna_id: str,
    columna_texto: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Carga un corpus desde CSV y descarta filas sin texto valido."""
    logger.info("Cargando corpus desde %s", ruta_csv)
    df = pd.read_csv(ruta_csv)

    filas_antes = len(df)
    df = df.dropna(subset=[columna_texto]).copy()
    df[columna_texto] = df[columna_texto].astype(str).str.strip()
    df = df[df[columna_texto] != ""].reset_index(drop=True)
    filas_descartadas = filas_antes - len(df)

    if filas_descartadas:
        logger.warning(
            "Se descartaron %s filas sin texto valido en %s",
            filas_descartadas,
            ruta_csv.name,
        )

    logger.info(
        "Corpus cargado: %s | documentos=%s | columnas=%s",
        ruta_csv.name,
        len(df),
        list(df.columns),
    )
    return df[[columna_id, columna_texto]].copy()


def generar_embeddings(
    modelo: SentenceTransformer,
    textos: list[str],
    batch_size: int,
    descripcion: str,
    logger: logging.Logger,
) -> np.ndarray:
    """Genera embeddings para una lista de textos con barra de progreso."""
    logger.info(
        "Generando embeddings para %s | documentos=%s | batch_size=%s",
        descripcion,
        len(textos),
        batch_size,
    )
    embeddings = modelo.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def main() -> None:
    """Orquesta la generacion y el guardado de embeddings de ambas agendas."""
    logger = configurar_logger()

    base_dir = Path(__file__).resolve().parents[2]
    carpeta_datos = base_dir / "data" / "processed"
    carpeta_reportes = base_dir / "outputs"
    carpeta_datos.mkdir(parents=True, exist_ok=True)
    carpeta_reportes.mkdir(parents=True, exist_ok=True)

    ruta_institucional = carpeta_datos / "corpus_institucional.csv"
    ruta_social = carpeta_datos / "corpus_social_salud.csv"

    df_institucional = cargar_corpus(
        ruta_institucional, COLUMNA_ID_INSTITUCIONAL, COLUMNA_TEXTO_INSTITUCIONAL, logger
    )
    df_social = cargar_corpus(
        ruta_social, COLUMNA_ID_SOCIAL, COLUMNA_TEXTO_SOCIAL, logger
    )

    logger.info("Cargando modelo SentenceTransformer: %s", NOMBRE_MODELO)
    modelo = SentenceTransformer(NOMBRE_MODELO)

    emb_social = generar_embeddings(
        modelo,
        df_social[COLUMNA_TEXTO_SOCIAL].tolist(),
        BATCH_SIZE_SOCIAL,
        "corpus social",
        logger,
    )
    emb_institucional = generar_embeddings(
        modelo,
        df_institucional[COLUMNA_TEXTO_INSTITUCIONAL].tolist(),
        BATCH_SIZE_INSTITUCIONAL,
        "corpus institucional",
        logger,
    )

    ruta_emb_social = carpeta_datos / "emb_social.npy"
    ruta_emb_institucional = carpeta_datos / "emb_institucional.npy"
    np.save(ruta_emb_social, emb_social)
    np.save(ruta_emb_institucional, emb_institucional)

    ruta_orden_social = carpeta_datos / "orden_social.parquet"
    ruta_orden_institucional = carpeta_datos / "orden_institucional.parquet"
    df_social.to_parquet(ruta_orden_social, index=False)
    df_institucional.to_parquet(ruta_orden_institucional, index=False)

    logger.info("Embeddings y archivos de orden guardados en %s", carpeta_datos)

    lineas_reporte = [
        "=" * 80,
        "REPORTE - FASE 3 / PASO 1: GENERACION DE EMBEDDINGS",
        "=" * 80,
        f"Modelo utilizado: {NOMBRE_MODELO}",
        "",
        "Corpus social:",
        f"  - Documentos            : {len(df_social)}",
        f"  - Dimension del embedding: {emb_social.shape[1]}",
        f"  - Batch size            : {BATCH_SIZE_SOCIAL}",
        f"  - Embeddings guardados en: {ruta_emb_social}",
        f"  - Orden guardado en      : {ruta_orden_social}",
        "",
        "Corpus institucional:",
        f"  - Documentos            : {len(df_institucional)}",
        f"  - Dimension del embedding: {emb_institucional.shape[1]}",
        f"  - Batch size            : {BATCH_SIZE_INSTITUCIONAL}",
        f"  - Embeddings guardados en: {ruta_emb_institucional}",
        f"  - Orden guardado en      : {ruta_orden_institucional}",
        "=" * 80,
    ]
    texto_reporte = "\n".join(lineas_reporte)

    ruta_reporte = carpeta_reportes / "reporte_embeddings.txt"
    ruta_reporte.write_text(texto_reporte, encoding="utf-8")

    print(texto_reporte)


if __name__ == "__main__":
    main()
