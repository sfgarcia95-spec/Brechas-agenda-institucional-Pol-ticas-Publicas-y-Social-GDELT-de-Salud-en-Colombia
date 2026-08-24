"""Fase 3 - Paso 3: modelado de topicos del corpus institucional con BERTopic.

Este script identifica los temas dominantes en la agenda institucional de
salud publica (Plan Decenal de Salud Publica y politicas asociadas) a partir
de los embeddings semanticos generados en el paso 1 (01_embeddings.py). Los
topicos resultantes se compararan mas adelante con los del corpus social para
estimar la brecha de agenda.

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
- data/processed/corpus_institucional.csv (columna "texto", id "id_doc")
- data/processed/emb_institucional.npy (embeddings precomputados, mismo orden
  que data/processed/orden_institucional.parquet generado en el paso 1)

Salidas:
- models/bertopic_institucional (modelo BERTopic entrenado, serializado en
  safetensors, incluye el embedding_model)
- data/processed/topics_institucional.csv (get_topic_info() del modelo)
- data/processed/asignacion_institucional.parquet (id_doc, topic, prob)
- outputs/reporte_bertopic_institucional.txt (resumen del paso ejecutado)

Nota clave: el corpus institucional es mucho mas pequeno que el social (miles
de parrafos frente a decenas de miles de titulares), por lo que se usa un
min_cluster_size mucho menor (MIN_CLUSTER_SIZE = 10) que en el script del
corpus social. Aun asi, si el resultado es degenerado (demasiado ruido o muy
pocos topicos), el script NO aborta: imprime una advertencia clara y guarda
igualmente todos los artefactos disponibles, para que el analisis de
estabilidad del paso siguiente pueda decidir como ajustar los hiperparametros.

Nota sobre alineacion: se recarga el corpus institucional directamente desde
el CSV original (misma logica de limpieza que en el paso 1: se eliminan filas
sin texto valido) para que el orden de los textos coincida exactamente con el
orden de las filas de emb_institucional.npy.
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

# Constante configurable: corpus institucional pequeno -> cluster minimo bajo.
MIN_CLUSTER_SIZE = 10

SEED = 42
NOMBRE_MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
COLUMNA_TEXTO = "texto"
COLUMNA_ID = "id_doc"
N_TOPICOS_REPORTE = 10

# Umbrales para detectar un resultado degenerado del clustering.
UMBRAL_PCT_RUIDO_ADVERTENCIA = 60.0
UMBRAL_MIN_TOPICOS_ADVERTENCIA = 3


def configurar_logger() -> logging.Logger:
    """Configura el logger principal del script."""
    logger = logging.getLogger("fase3_bertopic_institucional")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    manejador = logging.StreamHandler()
    manejador.setLevel(logging.INFO)
    manejador.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(manejador)
    return logger


def cargar_corpus_institucional(ruta_csv: Path, logger: logging.Logger) -> pd.DataFrame:
    """Carga el corpus institucional aplicando la misma limpieza que en el paso 1.

    Se replica exactamente el filtrado de filas sin texto valido usado al
    generar emb_institucional.npy, para asegurar que el orden de los textos
    coincida con el orden de los embeddings precomputados.
    """
    logger.info("Cargando corpus institucional desde %s", ruta_csv)
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

    logger.info("Corpus institucional cargado: %s documentos", len(df))
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


def verificar_resultado_degenerado(
    n_topicos: int,
    pct_ruido: float,
    logger: logging.Logger,
) -> str | None:
    """Detecta un clustering degenerado y arma un mensaje de advertencia.

    No aborta el script: solo genera el texto de advertencia (o None si el
    resultado es razonable) para incluirlo en el reporte y en la consola.
    """
    problemas: list[str] = []
    if pct_ruido > UMBRAL_PCT_RUIDO_ADVERTENCIA:
        problemas.append(
            f"el {pct_ruido:.2f}% de los documentos cayo en el topico -1 (ruido), "
            f"por encima del umbral de {UMBRAL_PCT_RUIDO_ADVERTENCIA:.0f}%"
        )
    if n_topicos < UMBRAL_MIN_TOPICOS_ADVERTENCIA:
        problemas.append(
            f"solo se generaron {n_topicos} topico(s), por debajo del minimo "
            f"esperado de {UMBRAL_MIN_TOPICOS_ADVERTENCIA}"
        )

    if not problemas:
        return None

    mensaje = (
        "ADVERTENCIA: resultado de clustering posiblemente degenerado en el "
        "corpus institucional -> " + "; ".join(problemas) + ". "
        f"Sugerencia: reducir MIN_CLUSTER_SIZE (actual={MIN_CLUSTER_SIZE}) y/o "
        "fijar `nr_topics` en BERTopic (p. ej. nr_topics='auto' o un entero "
        "concreto) para forzar un numero minimo de topicos utilizables. "
        "El modelo y los artefactos disponibles se guardan de todas formas."
    )
    logger.warning(mensaje)
    return mensaje


def generar_reporte(
    n_documentos: int,
    n_documentos_ruido: int,
    info_topicos: pd.DataFrame,
    advertencia: str | None,
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
        "REPORTE - FASE 3 / PASO 3: BERTOPIC SOBRE EL CORPUS INSTITUCIONAL",
        "=" * 80,
        f"min_cluster_size configurado: {MIN_CLUSTER_SIZE}",
        f"Documentos totales           : {n_documentos}",
        f"Numero de topicos (sin -1)   : {n_topicos}",
        f"Documentos en topico -1 (ruido): {n_documentos_ruido} ({pct_ruido:.2f}%)",
        "",
    ]

    if advertencia:
        lineas.extend([advertencia, ""])

    lineas.extend(
        [
            f"Top {min(N_TOPICOS_REPORTE, len(top10))} topicos por tamano:",
            top10[["Topic", "Count", "Name"]].to_string(index=False)
            if not top10.empty
            else "No se generaron topicos distintos de -1.",
            "=" * 80,
        ]
    )

    texto_reporte = "\n".join(lineas)
    logger.info("Numero de topicos (sin -1): %s", n_topicos)
    logger.info("Documentos en ruido: %s (%.2f%%)", n_documentos_ruido, pct_ruido)
    return texto_reporte


def main() -> None:
    """Orquesta el entrenamiento de BERTopic sobre el corpus institucional."""
    logger = configurar_logger()

    base_dir = Path(__file__).resolve().parents[2]
    carpeta_datos = base_dir / "data" / "processed"
    carpeta_reportes = base_dir / "outputs"
    carpeta_modelos = base_dir / "models"
    carpeta_reportes.mkdir(parents=True, exist_ok=True)
    carpeta_modelos.mkdir(parents=True, exist_ok=True)

    ruta_corpus_institucional = carpeta_datos / "corpus_institucional.csv"
    ruta_emb_institucional = carpeta_datos / "emb_institucional.npy"

    df_institucional = cargar_corpus_institucional(ruta_corpus_institucional, logger)
    emb_institucional = np.load(ruta_emb_institucional)

    if len(df_institucional) != emb_institucional.shape[0]:
        raise ValueError(
            f"Desalineacion detectada: {len(df_institucional)} documentos vs "
            f"{emb_institucional.shape[0]} embeddings. Revisar el paso 1 (01_embeddings.py)."
        )

    logger.info("Cargando stopwords en espanol desde spacy (es_core_news_sm)")
    stopwords_es = obtener_stopwords_es()
    logger.info("Stopwords cargadas: %s", len(stopwords_es))

    topic_model, _ = construir_modelo_bertopic(stopwords_es)

    textos = df_institucional[COLUMNA_TEXTO].tolist()
    logger.info("Entrenando BERTopic sobre %s documentos", len(textos))
    topics, probs = topic_model.fit_transform(textos, embeddings=emb_institucional)

    # Con serializacion "safetensors", save_embedding_model debe ser el nombre
    # del modelo (string) para que BERTopic incluya realmente el
    # embedding_model (como referencia a un modelo HuggingFace publico) en
    # config.json; pasar solo `True` no persiste el modelo de embeddings.
    ruta_modelo = carpeta_modelos / "bertopic_institucional"
    topic_model.save(
        ruta_modelo,
        serialization="safetensors",
        save_embedding_model=NOMBRE_MODELO_EMBEDDING,
    )
    logger.info("Modelo BERTopic guardado en %s", ruta_modelo)

    info_topicos = topic_model.get_topic_info()
    ruta_topics_csv = carpeta_datos / "topics_institucional.csv"
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
            COLUMNA_ID: df_institucional[COLUMNA_ID].to_numpy(),
            "topic": np.asarray(topics),
            "prob": prob_asignada,
        }
    )
    ruta_asignacion = carpeta_datos / "asignacion_institucional.parquet"
    df_asignacion.to_parquet(ruta_asignacion, index=False)
    logger.info("Asignacion documento->topico guardada en %s", ruta_asignacion)

    n_documentos_ruido = int((df_asignacion["topic"] == -1).sum())
    n_topicos = int((info_topicos["Topic"] != -1).sum())
    pct_ruido = 100 * n_documentos_ruido / len(df_institucional) if len(df_institucional) else 0.0

    advertencia = verificar_resultado_degenerado(n_topicos, pct_ruido, logger)

    texto_reporte = generar_reporte(
        len(df_institucional), n_documentos_ruido, info_topicos, advertencia, logger
    )

    ruta_reporte = carpeta_reportes / "reporte_bertopic_institucional.txt"
    ruta_reporte.write_text(texto_reporte, encoding="utf-8")

    print(texto_reporte)


if __name__ == "__main__":
    main()
