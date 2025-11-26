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
#  SMART CHUNKING POUR DOCUMENTS GÉNÉRIQUES
# =====================================================================

# Patterns pour détecter les titres/headers
HEADER_PATTERNS = [
    # Titres numérotés: "1. Introduction", "1.1 Overview", "Chapter 2: Methods"
    re.compile(r'^(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{0,100})$', re.MULTILINE),
    # Titres en majuscules seules
    re.compile(r'^([A-Z][A-Z\s]{5,60})$', re.MULTILINE),
    # Titres avec "Chapter", "Section", "Part"
    re.compile(r'^((?:Chapter|Section|Part|Appendix|Annex)\s+[\dIVXA-Z]+[:\s\-]*[^\n]*)$', re.MULTILINE | re.IGNORECASE),
]

# Patterns pour détecter les listes
LIST_PATTERNS = [
    # Listes à puces: -, *, •
    re.compile(r'^[\s]*[-*•]\s+.+', re.MULTILINE),
    # Listes numérotées: 1., 2., a), b), (1), (a)
    re.compile(r'^[\s]*(?:\d+[.)]|\([a-z0-9]+\)|[a-z][.)]\s).+', re.MULTILINE | re.IGNORECASE),
]


def _is_header(text: str) -> bool:
    """Vérifie si un paragraphe est un titre/header."""
    text = text.strip()
    if len(text) > 150 or len(text) < 3:
        return False
    for pattern in HEADER_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _is_list_item(text: str) -> bool:
    """Vérifie si un texte commence par un item de liste."""
    for pattern in LIST_PATTERNS:
        if pattern.match(text.strip()):
            return True
    return False


def _is_list_block(text: str) -> bool:
    """Vérifie si un bloc est une liste (plusieurs items)."""
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    list_items = sum(1 for line in lines if _is_list_item(line))
    return list_items >= len(lines) * 0.6  # Au moins 60% des lignes sont des items


def _split_into_sentences(text: str) -> List[str]:
    """Découpe un texte en phrases."""
    # Pattern pour fin de phrase (. ! ? suivis d'espace ou fin)
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$')
    sentences = sentence_pattern.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _detect_document_structure(text: str) -> List[Dict[str, Any]]:
    """
    Détecte la structure d'un document (titres, sections, listes).

    Returns:
        Liste de blocs avec type ('header', 'paragraph', 'list') et contenu
    """
    paragraphs = _split_into_paragraphs(text)

    blocks = []
    current_header = None

    for para in paragraphs:
        if _is_header(para):
            blocks.append({
                "type": "header",
                "content": para,
                "level": _detect_header_level(para)
            })
            current_header = para
        elif _is_list_block(para):
            blocks.append({
                "type": "list",
                "content": para,
                "parent_header": current_header
            })
        else:
            blocks.append({
                "type": "paragraph",
                "content": para,
                "parent_header": current_header
            })

    return blocks


def _detect_header_level(header: str) -> int:
    """Détecte le niveau d'un header (1, 2, 3...)."""
    header = header.strip()

    # Compter les points dans la numérotation: "1.2.3" -> niveau 3
    match = re.match(r'^(\d+(?:\.\d+)*)', header)
    if match:
        return match.group(1).count('.') + 1

    # Chapitre/Section/Part = niveau 1-2
    if re.match(r'^(?:Chapter|Part)', header, re.IGNORECASE):
        return 1
    if re.match(r'^(?:Section|Appendix|Annex)', header, re.IGNORECASE):
        return 2

    # Titres en majuscules = niveau 1
    if header.isupper():
        return 1

    return 2  # Par défaut niveau 2


def smart_chunk_generic(
    text: str,
    source_file: str = "",
    chunk_size: int = 1500,
    min_chunk_size: int = 200,
    overlap: int = 100,
    add_source_prefix: bool = True,
    preserve_lists: bool = True,
    preserve_headers: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chunking intelligent pour documents génériques.

    Features:
    1. Détection des titres/headers - garde le titre avec son contenu
    2. Préservation des listes - ne coupe pas une liste
    3. Coupure aux phrases - coupe en fin de phrase
    4. Contexte source - ajoute [Source: filename] en préfixe
    5. Détection de structure - respecte les sections

    Args:
        text: Texte à chunker
        source_file: Nom du fichier source (pour le préfixe)
        chunk_size: Taille max d'un chunk
        min_chunk_size: Taille min (fusion si en dessous)
        overlap: Chevauchement entre chunks
        add_source_prefix: Ajouter préfixe [Source: ...]
        preserve_lists: Ne pas couper les listes
        preserve_headers: Garder le header avec son contenu

    Returns:
        Liste de dicts avec text, header, type, etc.
    """
    if not text or not text.strip():
        return []

    # Préfixe source
    source_prefix = f"[Source: {source_file}]" if add_source_prefix and source_file else ""
    prefix_len = len(source_prefix) + 2 if source_prefix else 0
    effective_max = chunk_size - prefix_len

    # Détecter la structure
    blocks = _detect_document_structure(text)

    if not blocks:
        # Fallback sur simple_chunk
        raw_chunks = simple_chunk(text, chunk_size=effective_max, overlap=overlap)
        return [{"text": f"{source_prefix}\n\n{ch}".strip() if source_prefix else ch,
                 "chunk_index": i, "header": None, "type": "text"}
                for i, ch in enumerate(raw_chunks)]

    chunks = []
    current_text = ""
    current_header = None
    chunk_index = 0

    def flush_chunk():
        nonlocal current_text, current_header, chunk_index
        if current_text and len(current_text.strip()) >= min_chunk_size:
            # Ajouter le header au début si présent
            chunk_content = current_text.strip()
            if current_header and preserve_headers:
                chunk_content = f"[{current_header}]\n\n{chunk_content}"

            # Ajouter le préfixe source
            if source_prefix:
                chunk_content = f"{source_prefix}\n\n{chunk_content}"

            chunks.append({
                "text": chunk_content.strip(),
                "chunk_index": chunk_index,
                "header": current_header,
                "type": "structured",
            })
            chunk_index += 1
        current_text = ""

    for block in blocks:
        block_type = block["type"]
        content = block["content"]

        if block_type == "header":
            # Nouveau header = nouveau chunk potentiel
            if current_text:
                flush_chunk()
            current_header = content
            # Ne pas ajouter le header au contenu, il sera ajouté en préfixe
            continue

        # Vérifier si on peut ajouter ce bloc au chunk courant
        potential_len = len(current_text) + len(content) + 2

        if block_type == "list" and preserve_lists:
            # Liste: essayer de la garder entière
            if len(content) > effective_max:
                # Liste trop grande: flush et chunker la liste
                flush_chunk()
                list_chunks = _chunk_list(content, effective_max)
                for i, lc in enumerate(list_chunks):
                    chunk_content = lc
                    if current_header and preserve_headers:
                        chunk_content = f"[{current_header}]\n\n{chunk_content}"
                    if source_prefix:
                        chunk_content = f"{source_prefix}\n\n{chunk_content}"
                    chunks.append({
                        "text": chunk_content.strip(),
                        "chunk_index": chunk_index,
                        "header": current_header,
                        "type": "list",
                    })
                    chunk_index += 1
                continue
            elif potential_len > effective_max:
                flush_chunk()

        elif potential_len > effective_max:
            # Bloc trop grand pour le chunk courant
            if block_type == "paragraph" and len(content) > effective_max:
                # Paragraphe énorme: flush et découper par phrases
                flush_chunk()
                para_chunks = _chunk_by_sentences(content, effective_max)
                for pc in para_chunks:
                    current_text = pc
                    flush_chunk()
                continue
            else:
                flush_chunk()

        # Ajouter le bloc au chunk courant
        if current_text:
            current_text += "\n\n" + content
        else:
            current_text = content

    # Flush final
    flush_chunk()

    # Fusionner les petits chunks
    chunks = _merge_small_chunks(chunks, min_chunk_size, chunk_size)

    # Ajouter l'overlap
    chunks = _add_overlap_to_smart_chunks(chunks, overlap)

    return chunks


def _chunk_list(list_text: str, max_size: int) -> List[str]:
    """Découpe une liste en gardant les items ensemble."""
    lines = list_text.strip().split('\n')

    chunks = []
    current = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_size and current:
            chunks.append('\n'.join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append('\n'.join(current))

    return chunks


def _chunk_by_sentences(text: str, max_size: int) -> List[str]:
    """Découpe un texte en chunks en coupant aux fins de phrases."""
    sentences = _split_into_sentences(text)

    if not sentences:
        return _chunk_block(text, max_size)

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence) + 1

        if sent_len > max_size:
            # Phrase trop longue: flush et découper
            if current:
                chunks.append(' '.join(current))
                current = []
                current_len = 0
            sub_chunks = _chunk_block(sentence, max_size)
            chunks.extend(sub_chunks)
            continue

        if current_len + sent_len > max_size and current:
            chunks.append(' '.join(current))
            current = [sentence]
            current_len = sent_len
        else:
            current.append(sentence)
            current_len += sent_len

    if current:
        chunks.append(' '.join(current))

    return chunks


def _merge_small_chunks(chunks: List[Dict[str, Any]], min_size: int, max_size: int) -> List[Dict[str, Any]]:
    """Fusionne les chunks trop petits avec le précédent ou suivant."""
    if len(chunks) <= 1:
        return chunks

    merged = []
    i = 0

    while i < len(chunks):
        chunk = chunks[i]
        text = chunk["text"]

        if len(text) < min_size and merged:
            # Essayer de fusionner avec le précédent
            last = merged[-1]
            combined_len = len(last["text"]) + len(text) + 2

            if combined_len <= max_size + 200:
                last["text"] = last["text"] + "\n\n" + text
                i += 1
                continue

        if len(text) < min_size and i + 1 < len(chunks):
            # Essayer de fusionner avec le suivant
            next_chunk = chunks[i + 1]
            combined_len = len(text) + len(next_chunk["text"]) + 2

            if combined_len <= max_size + 200:
                chunk["text"] = text + "\n\n" + next_chunk["text"]
                chunk["header"] = chunk.get("header") or next_chunk.get("header")
                merged.append(chunk)
                i += 2
                continue

        merged.append(chunk)
        i += 1

    # Renuméroter les chunks
    for i, ch in enumerate(merged):
        ch["chunk_index"] = i

    return merged


def _add_overlap_to_smart_chunks(chunks: List[Dict[str, Any]], overlap: int) -> List[Dict[str, Any]]:
    """Ajoute un overlap entre les smart chunks."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1]["text"]

        # Extraire la fin du chunk précédent
        if len(prev_text) > overlap:
            # Essayer de couper à une fin de phrase
            tail = prev_text[-overlap * 2:]
            sentence_end = max(
                tail.rfind('. '),
                tail.rfind('? '),
                tail.rfind('! ')
            )
            if sentence_end > overlap // 2:
                overlap_text = tail[sentence_end + 2:]
            else:
                overlap_text = prev_text[-overlap:]
        else:
            overlap_text = prev_text

        # Ajouter au début du chunk courant (après le préfixe si présent)
        current_text = chunks[i]["text"]

        # Trouver où insérer l'overlap (après [Source:] et [Header])
        insert_pos = 0
        if current_text.startswith("[Source:"):
            end_bracket = current_text.find("]\n\n")
            if end_bracket > 0:
                insert_pos = end_bracket + 3

        if insert_pos < len(current_text) and current_text[insert_pos:].startswith("["):
            end_bracket = current_text.find("]\n\n", insert_pos)
            if end_bracket > 0:
                insert_pos = end_bracket + 3

        # Insérer l'overlap
        prefix = current_text[:insert_pos]
        suffix = current_text[insert_pos:]
        chunks[i]["text"] = f"{prefix}...{overlap_text}\n\n{suffix}".strip()

    return chunks

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
