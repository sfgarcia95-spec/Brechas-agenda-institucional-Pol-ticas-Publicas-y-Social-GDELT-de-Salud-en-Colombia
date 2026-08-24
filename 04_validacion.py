"""Fase 3 - Pasos 4 y 5: validacion de los modelos BERTopic y etiquetado interpretativo.

Este script no entrena nuevos modelos "oficiales": consume los modelos ya
entrenados en 02_bertopic_social.py y 03_bertopic_institucional.py para:

PASO 4 - Coherencia y estabilidad
  a) Calcula la coherencia de topicos (c_v, con gensim) de AMBOS modelos.
  b) Evalua la estabilidad del modelo SOCIAL reentrenando (solo en memoria,
     sin sobreescribir el modelo principal) con dos valores alternativos de
     min_cluster_size y comparando si los 10 topicos mayores del modelo
     principal se mantienen, usando solapamiento de documentos y de palabras
     clave.

PASO 5 - Etiquetado interpretativo (semiautomatico)
  Para cada topico de cada agenda extrae: top-10 palabras (c-TF-IDF), tamano y
  5 documentos representativos, y genera una tabla con una columna "etiqueta"
  vacia para revision y etiquetado MANUAL (la IA generativa se reserva para la
  fase 6 de este TFM).

Dependencias necesarias:
- pandas, pyarrow, numpy
- bertopic, umap-learn, hdbscan, scikit-learn
- spacy + modelo es_core_news_sm
- sentence-transformers
- gensim (coherencia c_v)

Comando de instalacion:
pip install pandas pyarrow numpy bertopic umap-learn hdbscan scikit-learn spacy sentence-transformers gensim
python -m spacy download es_core_news_sm

Entradas:
- data/processed/corpus_social_salud.csv y corpus_institucional.csv
- data/processed/emb_social.npy (para reentrenar las variantes de estabilidad)
- data/processed/asignacion_social.parquet y asignacion_institucional.parquet
  (columnas id, topic, prob; generadas en los pasos 2 y 3)
- models/bertopic_social y models/bertopic_institucional (modelos entrenados)

Salidas:
- data/processed/estabilidad_social.csv (tabla de estabilidad del modelo social)
- data/processed/topicos_para_etiquetar_social.csv
- data/processed/topicos_para_etiquetar_institucional.csv
- outputs/reporte_validacion_fase3.txt

Nota tecnica importante sobre la serializacion "safetensors" de BERTopic:
al recargar un modelo guardado con `topic_model.save(..., serialization=
"safetensors")`, BERTopic NO conserva el vectorizador ajustado, el c-TF-IDF
ni los documentos representativos internos (`representative_docs_` queda
vacio y `vectorizer_model` vuelve a ser un CountVectorizer() por defecto sin
stopwords ni n-gramas). Por eso este script:
  - reconstruye un analizador de texto identico al usado en el entrenamiento
    (mismas stopwords en espanol y ngram_range=(1,2)) para calcular la
    coherencia, en lugar de usar el vectorizador del modelo cargado; y
  - selecciona los documentos representativos del paso 5 a partir de la
    probabilidad de asignacion (columna "prob" de asignacion_*.parquet)
    generada durante el entrenamiento, en lugar de usar el metodo interno
    `get_representative_docs`, que tras la recarga queda vacio.
Lo que SI se conserva correctamente tras la recarga (y por tanto se reutiliza
tal cual) son `topics_` (asignacion documento->topico) y las palabras por
topico obtenidas con `get_topic()` / `get_topic_info()`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

SEED = 42
NOMBRE_MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

COLUMNA_TEXTO_SOCIAL = "titulo"
COLUMNA_ID_SOCIAL = "id_art"
COLUMNA_TEXTO_INSTITUCIONAL = "texto"
COLUMNA_ID_INSTITUCIONAL = "id_doc"

N_TOPICOS_REPORTE = 10
N_DOCS_REPRESENTATIVOS = 5

# Valores alternativos de min_cluster_size para la prueba de estabilidad del
# modelo social (el modelo principal usa MIN_CLUSTER_SIZE=50, definido en
# 02_bertopic_social.py). Estos modelos alternativos se entrenan solo en
# memoria para comparar y se descartan: el modelo principal NUNCA se
# sobreescribe.
MIN_CLUSTER_SIZE_ALTERNATIVOS_SOCIAL = [30, 80]

# Un topico del modelo principal se considera "estable" frente a una variante
# si existe un topico equivalente con solapamiento de documentos (Jaccard)
# mayor o igual a este umbral.
UMBRAL_SOLAPAMIENTO_ESTABLE = 0.5


def configurar_logger() -> logging.Logger:
    """Configura el logger principal del script."""
    logger = logging.getLogger("fase3_validacion")
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
    """Carga un corpus aplicando la misma limpieza usada en los pasos 1-3.

    Se replica exactamente el filtrado de filas sin texto valido para que el
    orden de los documentos coincida con el orden usado durante el
    entrenamiento (embeddings y asignaciones topic_model.topics_).
    """
    logger.info("Cargando corpus desde %s", ruta_csv)
    df = pd.read_csv(ruta_csv)

    filas_antes = len(df)
    df = df.dropna(subset=[columna_texto]).copy()
    df[columna_texto] = df[columna_texto].astype(str).str.strip()
    df = df[df[columna_texto] != ""].reset_index(drop=True)
    filas_descartadas = filas_antes - len(df)

    if filas_descartadas:
        logger.warning(
            "Se descartaron %s filas sin texto valido en %s (deben coincidir con pasos previos)",
            filas_descartadas,
            ruta_csv.name,
        )

    logger.info("Corpus cargado: %s | documentos=%s", ruta_csv.name, len(df))
    return df[[columna_id, columna_texto]].copy()


def obtener_stopwords_es() -> list[str]:
    """Obtiene la lista de stopwords en espanol desde spacy es_core_news_sm."""
    nlp = spacy.load("es_core_news_sm")
    return sorted(nlp.Defaults.stop_words)


# ---------------------------------------------------------------------------
# PASO 4a: coherencia de topicos (c_v)
# ---------------------------------------------------------------------------
def calcular_coherencia(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    stopwords_es: list[str],
    nombre_modelo: str,
    logger: logging.Logger,
) -> float:
    """Calcula la coherencia c_v del modelo siguiendo la receta oficial de
    BERTopic + gensim (ver FAQ de BERTopic: "How do I calculate the coherence
    score?"), adaptada para reconstruir el analizador de texto porque el
    modelo cargado desde safetensors no conserva el vectorizador ajustado.
    """
    documentos_df = pd.DataFrame({"Document": docs, "Topic": topics})
    documentos_por_topico = documentos_df.groupby(["Topic"], as_index=False).agg(
        {"Document": " ".join}
    )

    ids_topicos = sorted(t for t in documentos_por_topico["Topic"].unique() if t != -1)
    if not ids_topicos:
        logger.warning(
            "No hay topicos validos (distintos de -1) para calcular coherencia en %s",
            nombre_modelo,
        )
        return float("nan")

    textos_limpios = topic_model._preprocess_text(
        documentos_por_topico["Document"].values
    )

    # Se reconstruye el analizador con la MISMA configuracion usada en el
    # entrenamiento (stopwords en espanol + ngram_range=(1,2)); el
    # vectorizador del modelo cargado no sirve porque tras la recarga vuelve
    # a ser un CountVectorizer() por defecto, sin ajustar.
    analizador = CountVectorizer(stop_words=stopwords_es, ngram_range=(1, 2)).build_analyzer()
    tokens = [analizador(doc) for doc in textos_limpios]

    diccionario = Dictionary(tokens)
    corpus_bow = [diccionario.doc2bow(t) for t in tokens]

    palabras_por_topico = [
        [palabra for palabra, _ in topic_model.get_topic(topico)] for topico in ids_topicos
    ]
    palabras_por_topico = [p for p in palabras_por_topico if p]

    if not palabras_por_topico:
        logger.warning("Ningun topico tiene palabras clave validas en %s", nombre_modelo)
        return float("nan")

    modelo_coherencia = CoherenceModel(
        topics=palabras_por_topico,
        texts=tokens,
        corpus=corpus_bow,
        dictionary=diccionario,
        coherence="c_v",
    )
    score = float(modelo_coherencia.get_coherence())
    logger.info("Coherencia c_v (%s): %.4f", nombre_modelo, score)
    return score


# ---------------------------------------------------------------------------
# PASO 4b: estabilidad del modelo social
# ---------------------------------------------------------------------------
def entrenar_modelo_social_alterno(
    min_cluster_size: int,
    stopwords_es: list[str],
    modelo_embedding: SentenceTransformer,
    emb_social: np.ndarray,
    titulos_social: list[str],
    logger: logging.Logger,
) -> tuple[BERTopic, list[int]]:
    """Reentrena el modelo social solo en memoria con un min_cluster_size
    alternativo, reutilizando los embeddings precomputados. Este modelo NO se
    guarda en disco: se usa unicamente para la comparacion de estabilidad.
    """
    modelo_umap = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED,
    )
    modelo_hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizador = CountVectorizer(stop_words=stopwords_es, ngram_range=(1, 2))

    topic_model_alt = BERTopic(
        embedding_model=modelo_embedding,
        umap_model=modelo_umap,
        hdbscan_model=modelo_hdbscan,
        vectorizer_model=vectorizador,
        calculate_probabilities=False,
        language="multilingual",
        verbose=True,
    )

    logger.info(
        "Reentrenando modelo social alterno (min_cluster_size=%s) solo para estabilidad",
        min_cluster_size,
    )
    topics_alt, _ = topic_model_alt.fit_transform(titulos_social, embeddings=emb_social)
    return topic_model_alt, list(topics_alt)


def comparar_estabilidad(
    topic_model_base: BERTopic,
    topics_base: list[int],
    info_base: pd.DataFrame,
    modelos_alternos: dict[int, tuple[BERTopic, list[int]]],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compara los N_TOPICOS_REPORTE topicos mayores del modelo social
    principal contra cada variante alternativa (distinto min_cluster_size).

    Para cada topico grande del modelo principal se busca, en cada modelo
    alternativo, el topico con mayor solapamiento de documentos (indice de
    Jaccard sobre los indices de documentos asignados) y se reporta tambien
    el solapamiento de palabras clave (Jaccard sobre las top-10 palabras
    c-TF-IDF) para ese topico equivalente. Un topico se considera "estable"
    si el solapamiento de documentos alcanza UMBRAL_SOLAPAMIENTO_ESTABLE.
    """
    topics_base_arr = np.asarray(topics_base)
    top_n_base = (
        info_base[info_base["Topic"] != -1].sort_values("Count", ascending=False).head(N_TOPICOS_REPORTE)
    )

    filas: list[dict[str, object]] = []

    for min_cluster_size, (topic_model_alt, topics_alt) in modelos_alternos.items():
        topics_alt_arr = np.asarray(topics_alt)
        info_alt = topic_model_alt.get_topic_info()
        topicos_alt_validos = [t for t in info_alt["Topic"].tolist() if t != -1]

        docs_por_topico_alt = {
            t: set(np.where(topics_alt_arr == t)[0].tolist()) for t in topicos_alt_validos
        }

        for _, fila_base in top_n_base.iterrows():
            topic_base_id = int(fila_base["Topic"])
            docs_base = set(np.where(topics_base_arr == topic_base_id)[0].tolist())
            palabras_base = {w for w, _ in topic_model_base.get_topic(topic_base_id)}

            mejor_topico_alt: int | None = None
            mejor_jaccard_docs = 0.0
            for t_alt, docs_alt in docs_por_topico_alt.items():
                interseccion = len(docs_base & docs_alt)
                if interseccion == 0:
                    continue
                union = len(docs_base | docs_alt)
                jaccard = interseccion / union if union else 0.0
                if jaccard > mejor_jaccard_docs:
                    mejor_jaccard_docs = jaccard
                    mejor_topico_alt = t_alt

            if mejor_topico_alt is not None:
                palabras_alt = {w for w, _ in topic_model_alt.get_topic(mejor_topico_alt)}
                interseccion_palabras = len(palabras_base & palabras_alt)
                union_palabras = len(palabras_base | palabras_alt)
                jaccard_palabras = interseccion_palabras / union_palabras if union_palabras else 0.0
                tamano_alt = int(info_alt.loc[info_alt["Topic"] == mejor_topico_alt, "Count"].iloc[0])
            else:
                jaccard_palabras = 0.0
                tamano_alt = 0

            filas.append(
                {
                    "min_cluster_size_alterno": min_cluster_size,
                    "topic_base": topic_base_id,
                    "tamano_base": int(fila_base["Count"]),
                    "topic_alterno_equivalente": mejor_topico_alt,
                    "tamano_alterno": tamano_alt,
                    "solapamiento_documentos_jaccard": round(mejor_jaccard_docs, 4),
                    "solapamiento_palabras_jaccard": round(jaccard_palabras, 4),
                    "estable": mejor_jaccard_docs >= UMBRAL_SOLAPAMIENTO_ESTABLE,
                }
            )

    tabla = pd.DataFrame(filas)
    logger.info("Tabla de estabilidad construida con %s filas", len(tabla))
    return tabla


# ---------------------------------------------------------------------------
# PASO 5: etiquetado interpretativo (semiautomatico)
# ---------------------------------------------------------------------------
def construir_tabla_etiquetado(
    topic_model: BERTopic,
    df_corpus: pd.DataFrame,
    df_asignacion: pd.DataFrame,
    columna_id: str,
    columna_texto: str,
    n_docs: int = N_DOCS_REPRESENTATIVOS,
) -> pd.DataFrame:
    """Construye la tabla de apoyo al etiquetado manual de topicos.

    Los documentos representativos se seleccionan como los `n_docs`
    documentos con mayor probabilidad de pertenencia a cada topico (columna
    "prob" de la asignacion generada durante el entrenamiento), ya que el
    modelo BERTopic recargado desde safetensors no conserva el c-TF-IDF ni el
    vectorizador ajustado necesarios para usar `get_representative_docs`.
    """
    df_merge = df_asignacion.merge(
        df_corpus[[columna_id, columna_texto]], on=columna_id, how="left"
    )

    info = topic_model.get_topic_info()
    filas: list[dict[str, object]] = []

    for _, fila in info.iterrows():
        topic_id = int(fila["Topic"])
        tamano = int(fila["Count"])

        palabras = topic_model.get_topic(topic_id)
        palabras_clave = "|".join(palabra for palabra, _ in palabras) if palabras else ""

        subset = df_merge[df_merge["topic"] == topic_id].sort_values("prob", ascending=False)
        docs_repr = subset[columna_texto].head(n_docs).tolist()
        docs_repr_limpios = [
            str(doc).replace("\n", " ").replace("\r", " ").strip() for doc in docs_repr
        ]
        docs_representativos = " ||| ".join(docs_repr_limpios)

        filas.append(
            {
                "topic_id": topic_id,
                "palabras_clave": palabras_clave,
                "tamano": tamano,
                "docs_representativos": docs_representativos,
                "etiqueta": "",
            }
        )

    return pd.DataFrame(
        filas,
        columns=["topic_id", "palabras_clave", "tamano", "docs_representativos", "etiqueta"],
    )


def generar_reporte(
    coherencia_social: float,
    coherencia_institucional: float,
    tabla_estabilidad: pd.DataFrame,
) -> str:
    """Construye el texto del reporte final de los pasos 4 y 5."""
    lineas = [
        "=" * 80,
        "REPORTE - FASE 3 / PASOS 4 y 5: VALIDACION Y ETIQUETADO INTERPRETATIVO",
        "=" * 80,
        "",
        "PASO 4a - Coherencia de topicos (c_v, gensim):",
        f"  - Modelo social        : {coherencia_social:.4f}",
        f"  - Modelo institucional : {coherencia_institucional:.4f}",
        "",
        "PASO 4b - Estabilidad del modelo social (variando min_cluster_size):",
        f"  Modelo principal: min_cluster_size=50 (ver 02_bertopic_social.py, NO sobreescrito)",
        f"  Variantes evaluadas (solo en memoria): {MIN_CLUSTER_SIZE_ALTERNATIVOS_SOCIAL}",
        f"  Umbral de estabilidad (Jaccard documentos): {UMBRAL_SOLAPAMIENTO_ESTABLE}",
        "",
    ]

    if tabla_estabilidad.empty:
        lineas.append("  No se pudo construir la tabla de estabilidad.")
    else:
        for min_cluster_size, grupo in tabla_estabilidad.groupby("min_cluster_size_alterno"):
            n_estables = int(grupo["estable"].sum())
            n_total = len(grupo)
            media_doc = grupo["solapamiento_documentos_jaccard"].mean()
            media_palabras = grupo["solapamiento_palabras_jaccard"].mean()
            lineas.append(
                f"  - min_cluster_size={min_cluster_size}: {n_estables}/{n_total} de los "
                f"top-{N_TOPICOS_REPORTE} topicos se mantienen estables "
                f"(Jaccard docs medio={media_doc:.3f}, Jaccard palabras medio={media_palabras:.3f})"
            )
        lineas.extend(["", "Tabla de estabilidad completa:", tabla_estabilidad.to_string(index=False)])

    lineas.extend(
        [
            "",
            "PASO 5 - Tablas de etiquetado interpretativo generadas:",
            "  - data/processed/topicos_para_etiquetar_social.csv",
            "  - data/processed/topicos_para_etiquetar_institucional.csv",
            "  Columnas: topic_id, palabras_clave, tamano, docs_representativos, etiqueta",
            "  NOTA: la columna 'etiqueta' se deja vacia a proposito. El etiquetado final",
            "  de los topicos es MANUAL (revision humana); la IA generativa se reserva",
            "  para la fase 6 de este TFM.",
            "=" * 80,
        ]
    )

    return "\n".join(lineas)


def main() -> None:
    """Orquesta los pasos 4 (coherencia y estabilidad) y 5 (etiquetado)."""
    logger = configurar_logger()

    base_dir = Path(__file__).resolve().parents[2]
    carpeta_datos = base_dir / "data" / "processed"
    carpeta_reportes = base_dir / "outputs"
    carpeta_modelos = base_dir / "models"
    carpeta_reportes.mkdir(parents=True, exist_ok=True)

    ruta_modelo_social = carpeta_modelos / "bertopic_social"
    ruta_modelo_institucional = carpeta_modelos / "bertopic_institucional"
    if not ruta_modelo_social.exists() or not ruta_modelo_institucional.exists():
        raise FileNotFoundError(
            "No se encontraron los modelos entrenados. Ejecuta primero "
            "02_bertopic_social.py y 03_bertopic_institucional.py."
        )

    df_social = cargar_corpus(
        carpeta_datos / "corpus_social_salud.csv", COLUMNA_ID_SOCIAL, COLUMNA_TEXTO_SOCIAL, logger
    )
    df_institucional = cargar_corpus(
        carpeta_datos / "corpus_institucional.csv",
        COLUMNA_ID_INSTITUCIONAL,
        COLUMNA_TEXTO_INSTITUCIONAL,
        logger,
    )

    logger.info("Cargando stopwords en espanol desde spacy (es_core_news_sm)")
    stopwords_es = obtener_stopwords_es()
    logger.info("Stopwords cargadas: %s", len(stopwords_es))

    logger.info("Cargando modelos BERTopic entrenados (pasos 2 y 3)")
    topic_model_social = BERTopic.load(ruta_modelo_social)
    topic_model_institucional = BERTopic.load(ruta_modelo_institucional)

    if len(df_social) != len(topic_model_social.topics_):
        raise ValueError(
            "Desalineacion entre corpus_social_salud.csv y el modelo social entrenado."
        )
    if len(df_institucional) != len(topic_model_institucional.topics_):
        raise ValueError(
            "Desalineacion entre corpus_institucional.csv y el modelo institucional entrenado."
        )

    # ------------------------------------------------------------------
    # PASO 4a: coherencia c_v
    # ------------------------------------------------------------------
    coherencia_social = calcular_coherencia(
        topic_model_social,
        df_social[COLUMNA_TEXTO_SOCIAL].tolist(),
        list(topic_model_social.topics_),
        stopwords_es,
        "corpus social",
        logger,
    )
    coherencia_institucional = calcular_coherencia(
        topic_model_institucional,
        df_institucional[COLUMNA_TEXTO_INSTITUCIONAL].tolist(),
        list(topic_model_institucional.topics_),
        stopwords_es,
        "corpus institucional",
        logger,
    )

    # ------------------------------------------------------------------
    # PASO 4b: estabilidad del modelo social
    # ------------------------------------------------------------------
    emb_social = np.load(carpeta_datos / "emb_social.npy")
    if len(df_social) != emb_social.shape[0]:
        raise ValueError("Desalineacion entre corpus social y emb_social.npy (revisar paso 1).")

    logger.info("Cargando modelo de embeddings para entrenar variantes de estabilidad")
    modelo_embedding = SentenceTransformer(NOMBRE_MODELO_EMBEDDING)

    info_social_base = topic_model_social.get_topic_info()
    titulos_social = df_social[COLUMNA_TEXTO_SOCIAL].tolist()

    modelos_alternos: dict[int, tuple[BERTopic, list[int]]] = {}
    for min_cluster_size in MIN_CLUSTER_SIZE_ALTERNATIVOS_SOCIAL:
        modelos_alternos[min_cluster_size] = entrenar_modelo_social_alterno(
            min_cluster_size, stopwords_es, modelo_embedding, emb_social, titulos_social, logger
        )

    tabla_estabilidad = comparar_estabilidad(
        topic_model_social,
        list(topic_model_social.topics_),
        info_social_base,
        modelos_alternos,
        logger,
    )
    ruta_estabilidad = carpeta_datos / "estabilidad_social.csv"
    tabla_estabilidad.to_csv(ruta_estabilidad, index=False, encoding="utf-8")
    logger.info("Tabla de estabilidad guardada en %s", ruta_estabilidad)

    # ------------------------------------------------------------------
    # PASO 5: tablas de etiquetado interpretativo
    # ------------------------------------------------------------------
    df_asignacion_social = pd.read_parquet(carpeta_datos / "asignacion_social.parquet")
    df_asignacion_institucional = pd.read_parquet(carpeta_datos / "asignacion_institucional.parquet")

    tabla_social = construir_tabla_etiquetado(
        topic_model_social,
        df_social,
        df_asignacion_social,
        COLUMNA_ID_SOCIAL,
        COLUMNA_TEXTO_SOCIAL,
    )
    tabla_institucional = construir_tabla_etiquetado(
        topic_model_institucional,
        df_institucional,
        df_asignacion_institucional,
        COLUMNA_ID_INSTITUCIONAL,
        COLUMNA_TEXTO_INSTITUCIONAL,
    )

    ruta_tabla_social = carpeta_datos / "topicos_para_etiquetar_social.csv"
    ruta_tabla_institucional = carpeta_datos / "topicos_para_etiquetar_institucional.csv"
    tabla_social.to_csv(ruta_tabla_social, index=False, encoding="utf-8")
    tabla_institucional.to_csv(ruta_tabla_institucional, index=False, encoding="utf-8")
    logger.info(
        "Tablas de etiquetado guardadas en %s y %s", ruta_tabla_social, ruta_tabla_institucional
    )

    # ------------------------------------------------------------------
    # Reporte final
    # ------------------------------------------------------------------
    texto_reporte = generar_reporte(coherencia_social, coherencia_institucional, tabla_estabilidad)
    ruta_reporte = carpeta_reportes / "reporte_validacion_fase3.txt"
    ruta_reporte.write_text(texto_reporte, encoding="utf-8")

    print(texto_reporte)


if __name__ == "__main__":
    main()
