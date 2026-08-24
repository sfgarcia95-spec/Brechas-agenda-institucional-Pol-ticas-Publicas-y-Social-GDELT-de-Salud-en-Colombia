"""Fase 3 - Paso 2: modelado de topicos del corpus social con BERTopic.

Este script identifica los temas dominantes en la conversacion mediatica sobre
salud (agenda social/GDELT) a partir de los embeddings semanticos generados en
el paso 1 (01_embeddings.py). Los topicos resultantes se compararan mas
adelante con los de la agenda institucional para estimar la brecha de agenda.

Dependencias necesarias:
- pandas, pyarrow, numpy
- bertopic
- umap-learn
- hdbscan
- scikit-learn (CountVectorizer)
- spacy + modelo es_core_news_sm
- sentence-transformers
- safetensors

Comando de instalacion:
pip install pandas pyarrow numpy bertopic umap-learn hdbscan scikit-learn spacy safetensors
python -m spacy download es_core_news_sm

Entradas:
- data/processed/corpus_social_salud.csv (columna "titulo", id "id_art")
- data/processed/emb_social.npy (embeddings precomputados, mismo orden que
  data/processed/orden_social.parquet generado en el paso 1)

Salidas:
- models/bertopic_social (modelo BERTopic entrenado, serializado en
  safetensors, incluye el embedding_model)
- data/processed/topics_social.csv (get_topic_info() del modelo)
- data/processed/asignacion_social.parquet (id_art, topic, prob)
- outputs/reporte_bertopic_social.txt (resumen del paso ejecutado)

Nota sobre alineacion: se recarga el corpus social directamente desde el CSV
original (misma logica de limpieza que en el paso 1: se eliminan filas sin
texto valido) para que el orden de los titulos coincida exactamente con el
orden de las filas de emb_social.npy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# Constante configurable para las pruebas de estabilidad del paso siguiente.
MIN_CLUSTER_SIZE = 50

SEED = 42
NOMBRE_MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
COLUMNA_TEXTO = "titulo"
COLUMNA_ID = "id_art"
N_TOPICOS_REPORTE = 10


def configurar_logger() -> logging.Logger:
    """Configura el logger principal del script."""
    logger = logging.getLogger("fase3_bertopic_social")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    manejador = logging.StreamHandler()
    manejador.setLevel(logging.INFO)
    manejador.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(manejador)
    return logger


def cargar_corpus_social(ruta_csv: Path, logger: logging.Logger) -> pd.DataFrame:
    """Carga el corpus social aplicando la misma limpieza que en el paso 1.

    Se replica exactamente el filtrado de filas sin texto valido usado al
    generar emb_social.npy, para asegurar que el orden de los titulos
    coincida con el orden de los embeddings precomputados.
    """
    logger.info("Cargando corpus social desde %s", ruta_csv)
    df = pd.read_csv(ruta_csv)

    filas_antes = len(df)
    df = df.dropna(subset=[COLUMNA_TEXTO]).copy()
    df[COLUMNA_TEXTO] = df[COLUMNA_TEXTO].astype(str).str.strip()
    df = df[df[COLUMNA_TEXTO] != ""].reset_index(drop=True)
    filas_descartadas = filas_antes - len(df)

    if filas_descartadas:
        logger.warning(
            "Se descartaron %s filas sin texto valido (deben coincidir con el paso 1)",
            filas_descartadas,
        )

    logger.info("Corpus social cargado: %s documentos", len(df))
    return df


def obtener_stopwords_es() -> list[str]:
    """Obtiene la lista de stopwords en espanol desde spacy es_core_news_sm."""
    nlp = spacy.load("es_core_news_sm")
    return sorted(nlp.Defaults.stop_words)


def construir_modelo_bertopic(stopwords_es: list[str]) -> tuple[BERTopic, SentenceTransformer]:
    """Configura el pipeline completo de BERTopic con los componentes solicitados."""
    modelo_embedding = SentenceTransformer(NOMBRE_MODELO_EMBEDDING)

    modelo_umap = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED,
    )

    modelo_hdbscan = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizador = CountVectorizer(stop_words=stopwords_es, ngram_range=(1, 2))

    topic_model = BERTopic(
        embedding_model=modelo_embedding,
        umap_model=modelo_umap,
        hdbscan_model=modelo_hdbscan,
        vectorizer_model=vectorizador,
        calculate_probabilities=True,
        language="multilingual",
        verbose=True,
    )

    return topic_model, modelo_embedding


def generar_reporte(
    topic_model: BERTopic,
    n_documentos: int,
    n_documentos_ruido: int,
    info_topicos: pd.DataFrame,
    logger: logging.Logger,
) -> str:
    """Construye el texto del reporte final con las metricas solicitadas."""
    n_topicos = int((info_topicos["Topic"] != -1).sum())
    pct_ruido = 100 * n_documentos_ruido / n_documentos if n_documentos else 0.0

    top10 = (
        info_topicos[info_topicos["Topic"] != -1]
        .sort_values("Count", ascending=False)
        .head(N_TOPICOS_REPORTE)
    )

    lineas = [
        "=" * 80,
        "REPORTE - FASE 3 / PASO 2: BERTOPIC SOBRE EL CORPUS SOCIAL",
        "=" * 80,
        f"min_cluster_size configurado: {MIN_CLUSTER_SIZE}",
        f"Documentos totales           : {n_documentos}",
        f"Numero de topicos (sin -1)   : {n_topicos}",
        f"Documentos en topico -1 (ruido): {n_documentos_ruido} ({pct_ruido:.2f}%)",
        "",
        f"Top {N_TOPICOS_REPORTE} topicos por tamano:",
        top10[["Topic", "Count", "Name"]].to_string(index=False),
        "=" * 80,
    ]
    texto_reporte = "\n".join(lineas)
    logger.info("Numero de topicos (sin -1): %s", n_topicos)
    logger.info("Documentos en ruido: %s (%.2f%%)", n_documentos_ruido, pct_ruido)
    return texto_reporte


def main() -> None:
    """Orquesta el entrenamiento de BERTopic sobre el corpus social."""
    logger = configurar_logger()

    base_dir = Path(__file__).resolve().parents[2]
    carpeta_datos = base_dir / "data" / "processed"
    carpeta_reportes = base_dir / "outputs"
    carpeta_modelos = base_dir / "models"
    carpeta_reportes.mkdir(parents=True, exist_ok=True)
    carpeta_modelos.mkdir(parents=True, exist_ok=True)

    ruta_corpus_social = carpeta_datos / "corpus_social_salud.csv"
    ruta_emb_social = carpeta_datos / "emb_social.npy"

    df_social = cargar_corpus_social(ruta_corpus_social, logger)
    emb_social = np.load(ruta_emb_social)

    if len(df_social) != emb_social.shape[0]:
        raise ValueError(
            f"Desalineacion detectada: {len(df_social)} documentos vs "
            f"{emb_social.shape[0]} embeddings. Revisar el paso 1 (01_embeddings.py)."
        )

    logger.info("Cargando stopwords en espanol desde spacy (es_core_news_sm)")
    stopwords_es = obtener_stopwords_es()
    logger.info("Stopwords cargadas: %s", len(stopwords_es))

    topic_model, _ = construir_modelo_bertopic(stopwords_es)

    titulos = df_social[COLUMNA_TEXTO].tolist()
    logger.info("Entrenando BERTopic sobre %s documentos", len(titulos))
    topics, probs = topic_model.fit_transform(titulos, embeddings=emb_social)

    # Con serializacion "safetensors", save_embedding_model debe ser el nombre
    # del modelo (string) para que BERTopic incluya realmente el
    # embedding_model (como referencia a un modelo HuggingFace publico) en
    # config.json; pasar solo `True` no persiste el modelo de embeddings.
    ruta_modelo = carpeta_modelos / "bertopic_social"
    topic_model.save(
        ruta_modelo,
        serialization="safetensors",
        save_embedding_model=NOMBRE_MODELO_EMBEDDING,
    )
    logger.info("Modelo BERTopic guardado en %s", ruta_modelo)

    info_topicos = topic_model.get_topic_info()
    ruta_topics_csv = carpeta_datos / "topics_social.csv"
    info_topicos.to_csv(ruta_topics_csv, index=False, encoding="utf-8")
    logger.info("Informacion de topicos guardada en %s", ruta_topics_csv)

    # `probs` puede ser una matriz (documentos x topicos) o un vector segun la
    # configuracion; con calculate_probabilities=True y HDBSCAN se obtiene la
    # probabilidad del topico asignado por documento.
    probs_array = np.asarray(probs)
    if probs_array.ndim == 2:
        prob_asignada = probs_array.max(axis=1)
    else:
        prob_asignada = probs_array

    df_asignacion = pd.DataFrame(
        {
            COLUMNA_ID: df_social[COLUMNA_ID].to_numpy(),
            "topic": np.asarray(topics),
            "prob": prob_asignada,
        }
    )
    ruta_asignacion = carpeta_datos / "asignacion_social.parquet"
    df_asignacion.to_parquet(ruta_asignacion, index=False)
    logger.info("Asignacion documento->topico guardada en %s", ruta_asignacion)

    n_documentos_ruido = int((df_asignacion["topic"] == -1).sum())
    texto_reporte = generar_reporte(
        topic_model, len(df_social), n_documentos_ruido, info_topicos, logger
    )

    ruta_reporte = carpeta_reportes / "reporte_bertopic_social.txt"
    ruta_reporte.write_text(texto_reporte, encoding="utf-8")

    print(texto_reporte)


if __name__ == "__main__":
    main()
