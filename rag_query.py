# rag_query.py

import os
import time
from typing import Any, Dict, List, Optional

from faiss_store import FaissStore

from models_utils import (
    EMBED_MODEL,
    BATCH_SIZE,  # même si non utilisé directement, laissé pour compatibilité
    SNOWFLAKE_API_KEY,
    SNOWFLAKE_API_BASE,
    make_logger,
    create_http_client,
    DirectOpenAIEmbeddings,
    embed_in_batches,
    call_dallem_chat,
)

# Import optionnel du FeedbackStore pour le re-ranking
try:
    from feedback_store import FeedbackStore
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    FeedbackStore = None

logger = make_logger(debug=False)


# =====================================================================
#  FAISS STORE
# =====================================================================

def build_store(db_path: str) -> FaissStore:
    """
    Construit un store FAISS sur le répertoire db_path.
    Pas de retry nécessaire: FAISS fonctionne sans problème sur réseau!
    """
    logger.info(f"[QUERY] Creating FAISS store at: {db_path}")
    store = FaissStore(path=db_path)
    logger.info(f"[QUERY] ✅ FAISS store ready (no network issues!)")
    return store


# =====================================================================
#  RAG SUR UNE SEULE COLLECTION
# =====================================================================

def _run_rag_query_single_collection(
    db_path: str,
    collection_name: str,
    question: str,
    top_k: int = 30,
    call_llm: bool = True,
    log=None,
    feedback_store: Optional["FeedbackStore"] = None,
    use_feedback_reranking: bool = False,
    feedback_alpha: float = 0.3,
) -> Dict[str, Any]:
    """
    RAG sur une seule collection.

    - Si call_llm=True : retrieval + appel LLM
    - Si call_llm=False : retrieval uniquement (retourne context_str & sources, answer vide)
    - Si use_feedback_reranking=True et feedback_store fourni : applique le re-ranking
      basé sur les feedbacks utilisateurs
    """
    _log = log or logger

    question = (question or "").strip()
    if not question:
        raise ValueError("Question vide")

    _log.info(
        f"[RAG] (single) db={db_path} | collection={collection_name} | top_k={top_k}"
    )

    # 1) FAISS store + collection
    store = build_store(db_path)
    collection = store.get_collection(name=collection_name)

    # 2) Client embeddings Snowflake
    http_client = create_http_client()
    emb_client = DirectOpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SNOWFLAKE_API_KEY,
        base_url=SNOWFLAKE_API_BASE,
        http_client=http_client,
        role_prefix=True,
        logger=_log,
    )

    # 3) Embedding de la question (role="query")
    q_emb = embed_in_batches(
        texts=[question],
        role="query",
        batch_size=1,
        emb_client=emb_client,
        log=_log,
        dry_run=False,
    )[0]

    # 4) Requête FAISS (simple et fiable sur réseau!)
    _log.info("[RAG] Querying FAISS index...")
    raw = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    _log.info(f"[RAG] ✅ Query successful (FAISS = no network issues!)")

    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    if not docs:
        _log.warning("[RAG] Aucun document retourné par FAISS")
        return {
            "answer": (
                "Aucun contexte trouvé dans la base pour répondre à la question."
                if call_llm
                else ""
            ),
            "context_str": "",
            "raw_results": raw,
            "sources": [],
        }

    # 5) Construction du contexte + liste des sources
    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []

    for doc, meta, dist in zip(docs, metas, dists):
        if not isinstance(meta, dict):
            meta = {}

        source_file = meta.get("source_file", "unknown")
        chunk_id = meta.get("chunk_id", "?")
        section_id = meta.get("section_id") or ""
        section_kind = meta.get("section_kind") or ""
        section_title = meta.get("section_title") or ""
        path = meta.get("path") or ""

        # ID de référence pour matcher avec un CSV global
        ref_id = f"{db_path}|{collection_name}|{path or chunk_id}"

        header = (
            f"[source={source_file}, chunk={chunk_id}, "
            f"dist={float(dist):.3f}]"
        )
        context_blocks.append(f"{header}\n{doc}")

        sources.append(
            {
                "collection": collection_name,
                "source_file": source_file,
                "path": path,
                "chunk_id": chunk_id,
                "distance": float(dist),
                "score": 1.0 - float(dist),
                "section_id": section_id,
                "section_kind": section_kind,
                "section_title": section_title,
                "ref_id": ref_id,
                "text": doc,
            }
        )

    # ========== RE-RANKING BASÉ SUR LES FEEDBACKS ==========
    if use_feedback_reranking and feedback_store and FEEDBACK_AVAILABLE:
        _log.info("[RAG] Applying feedback-based re-ranking...")
        try:
            # Extraire le nom de la base depuis db_path
            base_name = os.path.basename(db_path)

            # Appliquer le re-ranking
            sources = feedback_store.compute_reranking_factors(
                sources=sources,
                base_name=base_name,
                collection_name=collection_name,
                question=question,
                alpha=feedback_alpha
            )

            # Reconstruire le contexte avec l'ordre des sources re-rankées
            context_blocks = []
            for src in sources:
                header = (
                    f"[source={src['source_file']}, chunk={src['chunk_id']}, "
                    f"score={src['score']:.3f}, boost={src.get('feedback_boost', 0):.3f}]"
                )
                context_blocks.append(f"{header}\n{src['text']}")

            _log.info(f"[RAG] ✅ Re-ranking applied (alpha={feedback_alpha})")
        except Exception as e:
            _log.warning(f"[RAG] Re-ranking failed, using original order: {e}")

    full_context = "\n\n".join(context_blocks)

    if not call_llm:
        # Mode "retrieval only"
        return {
            "answer": "",
            "context_str": full_context,
            "raw_results": raw,
            "sources": sources,
        }

    # 6) Appel LLM DALLEM
    answer = call_dallem_chat(
        http_client=http_client,
        question=question,
        context=full_context,
        log=_log,
    )

    return {
        "answer": answer,
        "context_str": full_context,
        "raw_results": raw,
        "sources": sources,
    }


# =====================================================================
#  RAG : UNE COLLECTION OU TOUTES LES COLLECTIONS (ALL)
# =====================================================================

def run_rag_query(
    db_path: str,
    collection_name: str,
    question: str,
    top_k: int = 30,
    synthesize_all: bool = False,
    log=None,
    feedback_store: Optional["FeedbackStore"] = None,
    use_feedback_reranking: bool = False,
    feedback_alpha: float = 0.3,
) -> Dict[str, Any]:
    """
    RAG "haut niveau" :

    - Si collection_name != "ALL" :
        → requête sur UNE collection (cf. _run_rag_query_single_collection)

    - Si collection_name == "ALL" :
        - synthesize_all = True :
             → retrieval sur toutes les collections,
               concaténé dans un gros contexte global,
               puis un SEUL appel LLM DALLEM.
        - synthesize_all = False :
             → par sécurité, on se contente de lever une erreur ou
               de déléguer à l'appelant (dans ton streamlit, ce cas
               est géré côté interface, pas ici).

    Options de re-ranking basé sur les feedbacks :
    - feedback_store : instance de FeedbackStore pour accéder aux feedbacks
    - use_feedback_reranking : activer le re-ranking basé sur les feedbacks
    - feedback_alpha : facteur d'influence (0-1, défaut 0.3)
    """
    _log = log or logger

    # Cas "ALL" (utilisé par streamlit_RAG quand synthesize_all=True)
    if collection_name == "ALL":
        _log.info(
            f"[RAG] Mode ALL collections | db={db_path} | synthesize_all={synthesize_all}"
        )
        store = build_store(db_path)
        collections = store.list_collections()  # FAISS retourne directement une liste de noms

        if not collections:
            return {
                "answer": "Aucune collection disponible dans cette base FAISS.",
                "context_str": "",
                "raw_results": {},
                "sources": [],
            }

        if not synthesize_all:
            # Dans ton streamlit, ce cas est géré côté interface (boucle sur les collections),
            # donc ici on renvoie une erreur explicite pour éviter toute ambiguïté.
            raise ValueError(
                "run_rag_query(collection_name='ALL') appelé avec synthesize_all=False. "
                "Ce mode doit être géré côté interface (une requête par collection)."
            )

        # ---- Mode synthèse globale : un seul appel LLM avec le contexte concaténé ----
        all_sources: List[Dict[str, Any]] = []
        all_context_blocks: List[str] = []

        for col_name in collections:  # FAISS retourne directement les noms (strings)
            _log.info(f"[RAG-ALL-SYNTH] Retrieval sur collection '{col_name}'")

            try:
                res = _run_rag_query_single_collection(
                    db_path=db_path,
                    collection_name=col_name,
                    question=question,
                    top_k=top_k,
                    call_llm=False,  # pas d'appel LLM ici
                    log=_log,
                    feedback_store=feedback_store,
                    use_feedback_reranking=use_feedback_reranking,
                    feedback_alpha=feedback_alpha,
                )
            except Exception as e:
                _log.error(
                    f"[RAG-ALL-SYNTH] Erreur pendant la récupération sur '{col_name}' : {e}"
                )
                continue

            context_str = res.get("context_str", "")
            sources = res.get("sources", [])

            if context_str:
                all_context_blocks.append(
                    f"=== CONTEXTE {col_name} ===\n{context_str}".strip()
                )

            for s in sources:
                if "collection" not in s:
                    s["collection"] = col_name
                all_sources.append(s)

        global_context = "\n\n".join(all_context_blocks)

        if not global_context.strip():
            return {
                "answer": (
                    "Aucun contexte trouvé dans la base pour répondre à la question (mode ALL)."
                ),
                "context_str": "",
                "raw_results": {},
                "sources": [],
            }

        http_client = create_http_client()
        answer = call_dallem_chat(
            http_client=http_client,
            question=(question or "").strip(),
            context=global_context,
            log=_log,
        )

        return {
            "answer": answer,
            "context_str": global_context,
            "raw_results": {},
            "sources": all_sources,
        }

    # Cas normal : une seule collection
    return _run_rag_query_single_collection(
        db_path=db_path,
        collection_name=collection_name,
        question=question,
        top_k=top_k,
        call_llm=True,
        log=_log,
        feedback_store=feedback_store,
        use_feedback_reranking=use_feedback_reranking,
        feedback_alpha=feedback_alpha,
    )
