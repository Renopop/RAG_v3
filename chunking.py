# chunking.py
"""
Chunking utilitaire pour le RAG.

- Découpe en paragraphes
- Re-colle en blocs d'environ `chunk_size` caractères
- Ajoute un overlap (chevauchement) pour garder le contexte
- Smart chunking pour sections EASA avec contexte préservé
"""

import re
from typing import List, Dict, Any, Optional


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Découpe le texte en paragraphes (séparés par au moins une ligne vide).
    """
    text = text.replace("\r\n", "\n")
    raw_paras = re.split(r"\n\s*\n", text)
    paras: List[str] = []
    for p in raw_paras:
        p = p.strip()
        if p:
            paras.append(p)
    return paras


def _chunk_block(text: str, chunk_size: int) -> List[str]:
    """
    Découpe un bloc de texte en sous-blocs ≤ chunk_size, en essayant de couper
    sur des fins de phrase / mots.
    """
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        # essayer de reculer jusqu'à un point / fin de phrase
        cut = end
        for sep in [". ", "; ", ", ", " "]:
            idx = text.rfind(sep, start, end)
            if idx != -1 and idx > start + int(chunk_size * 0.4):
                cut = idx + len(sep)
                break
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = cut
    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """
    Ajoute un overlap en réinjectant la fin du chunk précédent au début du suivant.
    Overlap exprimé en nombre de caractères (approx).
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    new_chunks: List[str] = []
    prev_tail = ""

    for i, ch in enumerate(chunks):
        if i == 0:
            new_chunks.append(ch)
        else:
            prefix = prev_tail
            combined = (prefix + "\n" + ch).strip()
            new_chunks.append(combined)

        # mettre à jour le tail pour le prochain
        if len(ch) > overlap:
            prev_tail = ch[-overlap:]
        else:
            prev_tail = ch

    return new_chunks


def simple_chunk(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[str]:
    """
    Chunking simple :

      - découpe en paragraphes
      - concatène les paragraphes jusqu'à chunk_size
      - si un paragraphe est très long, on le re-découpe avec _chunk_block
      - ajoute un overlap entre chunks

    Retourne une liste de chaînes (chunks).
    """
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    paras = _split_into_paragraphs(text)
    if not paras:
        return [text] if len(text) <= chunk_size else _add_overlap(
            _chunk_block(text, chunk_size), overlap
        )

    blocks: List[str] = []
    current: List[str] = []
    current_len = 0

    for p in paras:
        L = len(p)
        if L > int(chunk_size * 1.2):
            # paragraphe énorme : flush ce qu'on a, puis découpe lui-même
            if current:
                blocks.append("\n\n".join(current))
                current = []
                current_len = 0
            sub_chunks = _chunk_block(p, chunk_size)
            blocks.extend(sub_chunks)
            continue

        if current_len + L + 2 <= chunk_size:
            current.append(p)
            current_len += L + 2
        else:
            if current:
                blocks.append("\n\n".join(current))
            current = [p]
            current_len = L

    if current:
        blocks.append("\n\n".join(current))

    chunks = _add_overlap(blocks, overlap)
    return chunks


# =====================================================================
#  SMART CHUNKING POUR SECTIONS EASA
# =====================================================================

def _build_context_prefix(section: Dict[str, Any]) -> str:
    """
    Construit un préfixe de contexte pour un chunk.
    Ex: "[CS 25.1309 - Equipment, systems and installations]"
    """
    sec_id = section.get("id", "").strip()
    sec_title = section.get("title", "").strip()

    if sec_id and sec_title:
        return f"[{sec_id} - {sec_title}]"
    elif sec_id:
        return f"[{sec_id}]"
    return ""


def _detect_subsections(text: str) -> List[Dict[str, Any]]:
    """
    Détecte les sous-sections dans un texte EASA.
    Ex: (a), (b), (1), (2), (i), (ii)

    Returns:
        Liste de dicts avec 'marker', 'start', 'end', 'content'
    """
    # Pattern pour détecter les marqueurs de sous-section en début de ligne
    subsection_pattern = re.compile(
        r'^(\s*)(\([a-z]\)|\([0-9]+\)|\([ivx]+\))\s*',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(subsection_pattern.finditer(text))

    if not matches:
        return []

    subsections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        subsections.append({
            "marker": match.group(2),
            "indent": len(match.group(1)),
            "start": start,
            "end": end,
            "content": text[start:end].strip()
        })

    return subsections


def smart_chunk_section(
    section: Dict[str, Any],
    max_chunk_size: int = 1500,
    min_chunk_size: int = 200,
    overlap: int = 100,
    add_context_prefix: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chunking intelligent d'une section EASA.

    Règles:
    1. Si section < max_chunk_size : garder en un seul chunk
    2. Sinon, découper par sous-sections (a), (b), etc.
    3. Chaque chunk reçoit le préfixe de contexte [CS xx.xxx - Title]
    4. Les sous-sections trop petites sont fusionnées

    Args:
        section: Dict avec 'id', 'kind', 'number', 'title', 'full_text'
        max_chunk_size: Taille max avant découpage
        min_chunk_size: Taille min (fusion si en dessous)
        overlap: Chevauchement entre chunks
        add_context_prefix: Ajouter le préfixe [CS xx.xxx] à chaque chunk

    Returns:
        Liste de dicts avec 'text', 'section_id', 'section_title', 'chunk_index'
    """
    sec_id = section.get("id", "").strip()
    sec_title = section.get("title", "").strip()
    sec_kind = section.get("kind", "").strip()
    full_text = section.get("full_text", "").strip()

    if not full_text:
        return []

    context_prefix = _build_context_prefix(section) if add_context_prefix else ""
    prefix_len = len(context_prefix) + 2 if context_prefix else 0  # +2 pour \n\n

    effective_max = max_chunk_size - prefix_len

    chunks = []

    # Cas 1: Section assez petite → un seul chunk
    if len(full_text) <= effective_max:
        chunk_text = f"{context_prefix}\n\n{full_text}" if context_prefix else full_text
        chunks.append({
            "text": chunk_text.strip(),
            "section_id": sec_id,
            "section_kind": sec_kind,
            "section_title": sec_title,
            "chunk_index": 0,
            "is_complete_section": True,
        })
        return chunks

    # Cas 2: Section longue → essayer de découper par sous-sections
    subsections = _detect_subsections(full_text)

    if subsections:
        # Découpage par sous-sections avec fusion des petites
        current_text = ""
        current_markers = []
        chunk_index = 0

        # Texte avant la première sous-section (intro)
        intro_text = full_text[:subsections[0]["start"]].strip()
        if intro_text:
            current_text = intro_text

        for i, subsec in enumerate(subsections):
            subsec_content = subsec["content"]

            # Vérifier si on peut ajouter cette sous-section au chunk courant
            potential_len = len(current_text) + len(subsec_content) + 2

            if potential_len <= effective_max:
                # Ajouter au chunk courant
                if current_text:
                    current_text += "\n\n" + subsec_content
                else:
                    current_text = subsec_content
                current_markers.append(subsec["marker"])
            else:
                # Flush le chunk courant si non vide
                if current_text and len(current_text) >= min_chunk_size:
                    chunk_text = f"{context_prefix}\n\n{current_text}" if context_prefix else current_text
                    chunks.append({
                        "text": chunk_text.strip(),
                        "section_id": sec_id,
                        "section_kind": sec_kind,
                        "section_title": sec_title,
                        "chunk_index": chunk_index,
                        "subsections": current_markers.copy(),
                        "is_complete_section": False,
                    })
                    chunk_index += 1
                    current_text = ""
                    current_markers = []

                # Si la sous-section elle-même est trop grande, la découper
                if len(subsec_content) > effective_max:
                    sub_chunks = _chunk_block(subsec_content, effective_max)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk_text = f"{context_prefix}\n\n{sub_chunk}" if context_prefix else sub_chunk
                        chunks.append({
                            "text": chunk_text.strip(),
                            "section_id": sec_id,
                            "section_kind": sec_kind,
                            "section_title": sec_title,
                            "chunk_index": chunk_index,
                            "subsections": [subsec["marker"]],
                            "is_complete_section": False,
                        })
                        chunk_index += 1
                else:
                    current_text = subsec_content
                    current_markers = [subsec["marker"]]

        # Flush le dernier chunk
        if current_text:
            # Si trop petit, essayer de fusionner avec le précédent
            if len(current_text) < min_chunk_size and chunks:
                last_chunk = chunks[-1]
                combined = last_chunk["text"] + "\n\n" + current_text
                if len(combined) <= max_chunk_size + 200:  # Petite marge
                    last_chunk["text"] = combined
                    last_chunk["subsections"] = last_chunk.get("subsections", []) + current_markers
                else:
                    chunk_text = f"{context_prefix}\n\n{current_text}" if context_prefix else current_text
                    chunks.append({
                        "text": chunk_text.strip(),
                        "section_id": sec_id,
                        "section_kind": sec_kind,
                        "section_title": sec_title,
                        "chunk_index": chunk_index,
                        "subsections": current_markers,
                        "is_complete_section": False,
                    })
            else:
                chunk_text = f"{context_prefix}\n\n{current_text}" if context_prefix else current_text
                chunks.append({
                    "text": chunk_text.strip(),
                    "section_id": sec_id,
                    "section_kind": sec_kind,
                    "section_title": sec_title,
                    "chunk_index": chunk_index,
                    "subsections": current_markers,
                    "is_complete_section": False,
                })

    else:
        # Pas de sous-sections détectées → découpage classique avec contexte
        raw_chunks = simple_chunk(full_text, chunk_size=effective_max, overlap=overlap)

        for i, chunk in enumerate(raw_chunks):
            chunk_text = f"{context_prefix}\n\n{chunk}" if context_prefix else chunk
            chunks.append({
                "text": chunk_text.strip(),
                "section_id": sec_id,
                "section_kind": sec_kind,
                "section_title": sec_title,
                "chunk_index": i,
                "is_complete_section": False,
            })

    return chunks


def chunk_easa_sections(
    sections: List[Dict[str, Any]],
    max_chunk_size: int = 1500,
    min_chunk_size: int = 200,
    merge_small_sections: bool = True,
    add_context_prefix: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chunking intelligent de toutes les sections EASA d'un document.

    Features:
    - Préserve le contexte (préfixe [CS xx.xxx - Title])
    - Fusionne les petites sections adjacentes
    - Découpe intelligemment les grandes sections par sous-sections

    Args:
        sections: Liste de sections EASA (from split_easa_sections)
        max_chunk_size: Taille max d'un chunk
        min_chunk_size: Taille min (fusion si en dessous)
        merge_small_sections: Fusionner les sections trop petites
        add_context_prefix: Ajouter préfixe de contexte

    Returns:
        Liste de chunks avec métadonnées
    """
    if not sections:
        return []

    all_chunks = []
    pending_small_sections = []
    pending_text = ""
    pending_ids = []

    for section in sections:
        full_text = section.get("full_text", "").strip()
        sec_id = section.get("id", "").strip()

        if not full_text:
            continue

        # Si la section est petite et merge activé
        if merge_small_sections and len(full_text) < min_chunk_size:
            context = _build_context_prefix(section)
            section_with_context = f"{context}\n{full_text}" if context else full_text

            # Essayer de fusionner avec les pending
            if len(pending_text) + len(section_with_context) + 4 <= max_chunk_size:
                if pending_text:
                    pending_text += "\n\n---\n\n" + section_with_context
                else:
                    pending_text = section_with_context
                pending_ids.append(sec_id)
                pending_small_sections.append(section)
            else:
                # Flush pending et commencer nouveau groupe
                if pending_text:
                    all_chunks.append({
                        "text": pending_text.strip(),
                        "section_id": " | ".join(pending_ids),
                        "section_kind": pending_small_sections[0].get("kind", "") if pending_small_sections else "",
                        "section_title": "Sections fusionnées",
                        "chunk_index": 0,
                        "merged_sections": pending_ids.copy(),
                        "is_complete_section": True,
                    })
                pending_text = section_with_context
                pending_ids = [sec_id]
                pending_small_sections = [section]
        else:
            # Flush pending avant de traiter une grande section
            if pending_text:
                all_chunks.append({
                    "text": pending_text.strip(),
                    "section_id": " | ".join(pending_ids),
                    "section_kind": pending_small_sections[0].get("kind", "") if pending_small_sections else "",
                    "section_title": "Sections fusionnées",
                    "chunk_index": 0,
                    "merged_sections": pending_ids.copy(),
                    "is_complete_section": True,
                })
                pending_text = ""
                pending_ids = []
                pending_small_sections = []

            # Chunker la section normalement
            section_chunks = smart_chunk_section(
                section,
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
                add_context_prefix=add_context_prefix,
            )
            all_chunks.extend(section_chunks)

    # Flush final pending
    if pending_text:
        all_chunks.append({
            "text": pending_text.strip(),
            "section_id": " | ".join(pending_ids),
            "section_kind": pending_small_sections[0].get("kind", "") if pending_small_sections else "",
            "section_title": "Sections fusionnées",
            "chunk_index": 0,
            "merged_sections": pending_ids.copy(),
            "is_complete_section": True,
        })

    return all_chunks
