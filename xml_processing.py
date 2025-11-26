"""
XML Processing Module - Parser pour fichiers XML de normes EASA
Spécialisé pour extraire et découper par sections CS xx.xxx
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


# Pattern pour détecter les sections EASA (CS 25.101, CS-25.101, CS25.101, etc.)
EASA_SECTION_PATTERN = re.compile(
    r'(CS[-\s]?\d+[A-Z]?[-.]?\d+(?:\.\d+)?(?:\s*[a-z])?)',
    re.IGNORECASE
)


@dataclass
class XMLParseConfig:
    """Configuration pour le parsing XML"""
    split_by_sections: bool = True           # Découper par sections CS xx.xxx
    include_section_title: bool = True       # Inclure le titre de section dans le chunk
    min_section_length: int = 50             # Longueur minimum d'une section (caractères)
    excluded_tags: List[str] = field(default_factory=list)  # Tags XML à exclure


@dataclass
class EASASection:
    """Une section de norme EASA"""
    code: str           # Ex: "CS 25.101"
    title: str          # Titre de la section
    content: str        # Contenu textuel
    start_pos: int      # Position dans le texte original


def extract_text_from_xml(xml_path: str, config: Optional[XMLParseConfig] = None) -> str:
    """
    Extrait le texte d'un fichier XML.

    Args:
        xml_path: Chemin vers le fichier XML
        config: Configuration (optionnelle)

    Returns:
        Texte extrait du XML
    """
    if config is None:
        config = XMLParseConfig()

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return _extract_all_text(root, config)
    except ET.ParseError as e:
        logger.error(f"Erreur de parsing XML {xml_path}: {e}")
        # Fallback: lire comme texte brut
        try:
            with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction XML {xml_path}: {e}")
        return ""


def _strip_namespace(tag: str) -> str:
    """Retire le namespace d'un tag XML"""
    if "}" in tag:
        return tag.split("}")[1]
    return tag


def _extract_all_text(root: ET.Element, config: XMLParseConfig) -> str:
    """Extrait tout le texte d'un élément XML"""
    texts = []

    def recurse(elem):
        tag_name = _strip_namespace(elem.tag)

        # Ignorer les tags exclus
        if tag_name.lower() in [t.lower() for t in config.excluded_tags]:
            return

        if elem.text:
            text = elem.text.strip()
            if text:
                texts.append(text)

        for child in elem:
            recurse(child)

        if elem.tail:
            tail = elem.tail.strip()
            if tail:
                texts.append(tail)

    recurse(root)
    return "\n".join(texts)


def detect_easa_sections(text: str) -> List[EASASection]:
    """
    Détecte et extrait les sections EASA (CS xx.xxx) dans un texte.

    Args:
        text: Texte à analyser

    Returns:
        Liste des sections trouvées
    """
    sections = []

    # Trouver tous les marqueurs de section
    matches = list(EASA_SECTION_PATTERN.finditer(text))

    if not matches:
        # Pas de sections trouvées, retourner tout le texte comme une seule section
        return [EASASection(
            code="DOCUMENT",
            title="Contenu complet",
            content=text.strip(),
            start_pos=0
        )]

    for i, match in enumerate(matches):
        code = match.group(1).strip()
        start_pos = match.start()

        # Fin de la section = début de la suivante ou fin du texte
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        # Extraire le contenu de la section
        section_text = text[start_pos:end_pos].strip()

        # Extraire le titre (première ligne après le code)
        lines = section_text.split('\n')
        title = ""
        if len(lines) > 0:
            # Le titre est souvent sur la même ligne ou la ligne suivante
            first_line = lines[0].replace(code, "").strip()
            if first_line:
                title = first_line[:100]  # Limiter la longueur du titre
            elif len(lines) > 1:
                title = lines[1].strip()[:100]

        content = section_text

        sections.append(EASASection(
            code=code,
            title=title,
            content=content,
            start_pos=start_pos
        ))

    return sections


def analyze_xml_for_easa(xml_path: str) -> Dict[str, Any]:
    """
    Analyse un fichier XML pour détecter les sections EASA.

    Returns:
        Dict avec informations sur le document et les sections trouvées
    """
    result = {
        "file": os.path.basename(xml_path),
        "total_chars": 0,
        "sections_count": 0,
        "sections": [],
        "error": None
    }

    try:
        text = extract_text_from_xml(xml_path)
        result["total_chars"] = len(text)

        sections = detect_easa_sections(text)
        result["sections_count"] = len(sections)

        # Résumé des sections
        for sec in sections:
            result["sections"].append({
                "code": sec.code,
                "title": sec.title[:50] + "..." if len(sec.title) > 50 else sec.title,
                "length": len(sec.content),
                "preview": sec.content[:150].replace('\n', ' ') + "..." if len(sec.content) > 150 else sec.content.replace('\n', ' ')
            })

    except Exception as e:
        result["error"] = str(e)

    return result


def preview_xml_sections(xml_path: str, max_sections: int = 10) -> Tuple[str, Dict[str, Any]]:
    """
    Génère une prévisualisation des sections EASA trouvées dans un XML.

    Args:
        xml_path: Chemin vers le fichier XML
        max_sections: Nombre max de sections à afficher

    Returns:
        Tuple (texte_preview, stats)
    """
    analysis = analyze_xml_for_easa(xml_path)

    if analysis["error"]:
        return f"Erreur: {analysis['error']}", analysis

    lines = []
    lines.append(f"📄 Fichier: {analysis['file']}")
    lines.append(f"📊 {analysis['total_chars']:,} caractères")
    lines.append(f"📑 {analysis['sections_count']} section(s) détectée(s)")
    lines.append("")
    lines.append("=" * 50)
    lines.append("SECTIONS TROUVÉES:")
    lines.append("=" * 50)

    for i, sec in enumerate(analysis["sections"][:max_sections]):
        lines.append("")
        lines.append(f"[{i+1}] {sec['code']}")
        if sec['title']:
            lines.append(f"    Titre: {sec['title']}")
        lines.append(f"    Taille: {sec['length']:,} caractères")
        lines.append(f"    Aperçu: {sec['preview'][:100]}...")

    if analysis["sections_count"] > max_sections:
        lines.append("")
        lines.append(f"... et {analysis['sections_count'] - max_sections} autres sections")

    return "\n".join(lines), analysis


def get_sections_for_chunking(xml_path: str, config: Optional[XMLParseConfig] = None) -> List[Dict[str, str]]:
    """
    Retourne les sections prêtes pour le chunking.

    Chaque section devient un chunk avec:
    - text: le contenu
    - metadata: code de section, titre

    Args:
        xml_path: Chemin vers le fichier XML
        config: Configuration de parsing

    Returns:
        Liste de dicts {text, code, title}
    """
    if config is None:
        config = XMLParseConfig()

    text = extract_text_from_xml(xml_path, config)
    sections = detect_easa_sections(text)

    chunks = []
    for sec in sections:
        if len(sec.content) >= config.min_section_length:
            chunk_text = sec.content
            if config.include_section_title and sec.title:
                chunk_text = f"{sec.code} - {sec.title}\n\n{sec.content}"

            chunks.append({
                "text": chunk_text,
                "code": sec.code,
                "title": sec.title
            })

    return chunks


# Pour compatibilité avec l'ancien code
def detect_xml_structure(xml_path: str) -> Dict[str, Any]:
    """Analyse la structure d'un fichier XML (compatibilité)"""
    return analyze_xml_for_easa(xml_path)


def get_recommended_config(structure_info: Dict[str, Any]) -> XMLParseConfig:
    """Retourne une config par défaut (compatibilité)"""
    return XMLParseConfig()
