"""
Advanced Search Module - Query Expansion & Multi-Query
Améliore la recherche RAG sans dépendances supplémentaires
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def expand_query_with_llm(
    question: str,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    num_variations: int = 3,
    log=None
) -> List[str]:
    """
    Génère des variations de la question originale pour améliorer le recall.

    Args:
        question: Question originale
        http_client: Client HTTP
        api_key: Clé API du LLM
        api_base: URL de base de l'API
        model: Nom du modèle
        num_variations: Nombre de variations à générer
        log: Logger optionnel

    Returns:
        Liste de questions (originale + variations)
    """
    _log = log or logger

    # Toujours inclure la question originale
    queries = [question]

    system_prompt = """Tu es un expert en reformulation de questions pour la recherche documentaire.
Génère des variations de la question qui pourraient aider à trouver des documents pertinents.
Les variations doivent:
- Utiliser des synonymes
- Reformuler différemment
- Être plus spécifiques ou plus générales
- Garder le même sens

Réponds UNIQUEMENT avec les variations, une par ligne, sans numérotation ni explication."""

    user_prompt = f"""Question originale: {question}

Génère {num_variations} variations de cette question pour améliorer la recherche:"""

    try:
        url = api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        }

        resp = http_client.post(url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if content:
            # Parser les variations (une par ligne)
            variations = [v.strip() for v in content.split("\n") if v.strip()]
            # Filtrer les lignes qui ressemblent à des numéros ou sont trop courtes
            variations = [v for v in variations if len(v) > 10 and not v[0].isdigit()]
            queries.extend(variations[:num_variations])

        _log.info(f"[QUERY-EXPAND] {len(queries)} requêtes générées (original + {len(queries)-1} variations)")

    except Exception as e:
        _log.warning(f"[QUERY-EXPAND] Échec de l'expansion: {e}. Utilisation de la question originale uniquement.")

    return queries


def merge_search_results(
    results_list: List[Dict[str, Any]],
    max_results: int = 30,
    log=None
) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Fusionne les résultats de plusieurs requêtes en éliminant les doublons
    et en combinant les scores.

    Args:
        results_list: Liste des résultats de chaque requête
        max_results: Nombre max de résultats à retourner
        log: Logger optionnel

    Returns:
        Tuple (documents, metadatas, scores)
    """
    _log = log or logger

    # Dictionnaire pour agréger les scores par document
    doc_scores: Dict[str, Dict[str, Any]] = {}

    for query_idx, result in enumerate(results_list):
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Utiliser le chunk_id ou le hash du document comme clé
            doc_key = meta.get("chunk_id") or meta.get("path") or hash(doc[:100])
            doc_key = str(doc_key)

            # Score = 1 - distance (pour L2, plus petit = meilleur)
            score = 1.0 - float(dist) if dist < 1.0 else 0.0

            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    "document": doc,
                    "metadata": meta,
                    "scores": [],
                    "best_score": score,
                    "hit_count": 0,
                }

            doc_scores[doc_key]["scores"].append(score)
            doc_scores[doc_key]["hit_count"] += 1
            doc_scores[doc_key]["best_score"] = max(doc_scores[doc_key]["best_score"], score)

    # Calculer le score final combiné
    # Formule: score_final = best_score * (1 + 0.1 * (hit_count - 1))
    # Un document trouvé par plusieurs requêtes est boosté
    for doc_key, data in doc_scores.items():
        hit_bonus = 0.1 * (data["hit_count"] - 1)
        data["final_score"] = data["best_score"] * (1 + hit_bonus)

    # Trier par score final décroissant
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["final_score"], reverse=True)

    # Extraire les résultats
    documents = [d["document"] for d in sorted_docs[:max_results]]
    metadatas = [d["metadata"] for d in sorted_docs[:max_results]]
    scores = [d["final_score"] for d in sorted_docs[:max_results]]

    _log.info(f"[MERGE] {len(doc_scores)} documents uniques fusionnés, {len(documents)} retenus")

    return documents, metadatas, scores


def run_multi_query_search(
    collection,
    queries: List[str],
    embed_func,
    top_k: int = 20,
    log=None
) -> Dict[str, Any]:
    """
    Exécute plusieurs requêtes et fusionne les résultats.

    Args:
        collection: Collection FAISS
        queries: Liste des requêtes (question originale + variations)
        embed_func: Fonction pour générer les embeddings
        top_k: Nombre de résultats par requête
        log: Logger optionnel

    Returns:
        Résultats fusionnés au format FAISS
    """
    _log = log or logger

    all_results = []

    for i, query in enumerate(queries):
        _log.info(f"[MULTI-QUERY] Requête {i+1}/{len(queries)}: {query[:50]}...")

        try:
            # Générer l'embedding de la requête
            q_emb = embed_func(query)

            # Recherche FAISS
            result = collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            all_results.append(result)

        except Exception as e:
            _log.warning(f"[MULTI-QUERY] Erreur sur requête {i+1}: {e}")
            continue

    if not all_results:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Fusionner les résultats
    documents, metadatas, scores = merge_search_results(all_results, max_results=top_k * 2, log=_log)

    # Convertir scores en distances pour compatibilité
    distances = [1.0 - s for s in scores]

    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def generate_sub_questions(
    question: str,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    log=None
) -> List[str]:
    """
    Décompose une question complexe en sous-questions plus simples.
    Utile pour les questions multi-aspects.

    Args:
        question: Question complexe
        http_client: Client HTTP
        api_key: Clé API
        api_base: URL de base
        model: Nom du modèle
        log: Logger

    Returns:
        Liste de sous-questions
    """
    _log = log or logger

    system_prompt = """Tu es un expert en décomposition de questions complexes.
Si la question contient plusieurs aspects ou sous-questions implicites, décompose-la.
Si la question est simple, retourne-la telle quelle.

Réponds UNIQUEMENT avec les questions, une par ligne, sans numérotation ni explication."""

    user_prompt = f"""Question: {question}

Décompose cette question si elle contient plusieurs aspects:"""

    try:
        url = api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 200,
        }

        resp = http_client.post(url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if content:
            sub_questions = [q.strip() for q in content.split("\n") if q.strip() and len(q.strip()) > 10]
            if sub_questions:
                _log.info(f"[SUB-Q] Question décomposée en {len(sub_questions)} sous-questions")
                return sub_questions

    except Exception as e:
        _log.warning(f"[SUB-Q] Échec de la décomposition: {e}")

    return [question]
