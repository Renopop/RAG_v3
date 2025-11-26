# chunking.py
"""
Chunking utilitaire pour le RAG.

- Découpe en paragraphes
- Re-colle en blocs d'environ `chunk_size` caractères
- Ajoute un overlap (chevauchement) pour garder le contexte
"""

import re
from typing import List


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
