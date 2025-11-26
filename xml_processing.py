"""
XML Processing Module - Parser intelligent pour fichiers XML
Permet différentes stratégies d'extraction de texte avec prévisualisation
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class XMLParseStrategy(Enum):
    """Stratégies de parsing XML disponibles"""
    FULL_TEXT = "full_text"           # Extrait tout le texte (ignore les balises)
    STRUCTURED = "structured"          # Garde la structure avec indentation
    TAG_FILTERED = "tag_filtered"      # Filtre par tags spécifiques
    XPATH_QUERY = "xpath_query"        # Extraction via XPath
    ATTRIBUTES_INCLUDED = "attributes" # Inclut les attributs dans le texte


@dataclass
class XMLParseConfig:
    """Configuration pour le parsing XML"""
    strategy: XMLParseStrategy = XMLParseStrategy.FULL_TEXT
    selected_tags: Optional[List[str]] = None      # Tags à inclure (pour TAG_FILTERED)
    excluded_tags: Optional[List[str]] = None      # Tags à exclure
    xpath_queries: Optional[List[str]] = None      # Requêtes XPath
    include_attributes: bool = False               # Inclure les attributs
    preserve_whitespace: bool = False              # Préserver les espaces
    add_tag_markers: bool = False                  # Ajouter [TAG] devant le contenu
    separator: str = "\n"                          # Séparateur entre éléments


def detect_xml_structure(xml_path: str) -> Dict[str, Any]:
    """
    Analyse la structure d'un fichier XML et retourne des informations utiles.

    Returns:
        Dict avec:
        - root_tag: nom de la balise racine
        - all_tags: set de tous les tags trouvés
        - tag_counts: comptage de chaque tag
        - sample_content: aperçu du contenu par tag
        - has_namespaces: si le XML utilise des namespaces
        - encoding: encodage détecté
    """
    result = {
        "root_tag": None,
        "all_tags": set(),
        "tag_counts": {},
        "sample_content": {},
        "has_namespaces": False,
        "encoding": "utf-8",
        "total_elements": 0,
        "max_depth": 0,
        "attributes_found": set(),
    }

    try:
        # Détecter l'encodage depuis la déclaration XML
        with open(xml_path, "rb") as f:
            first_line = f.readline().decode("utf-8", errors="ignore")
            if "encoding=" in first_line:
                match = re.search(r'encoding=["\']([^"\']+)["\']', first_line)
                if match:
                    result["encoding"] = match.group(1)

        # Parser le XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Analyser la racine
        result["root_tag"] = _strip_namespace(root.tag)
        result["has_namespaces"] = "{" in root.tag

        # Parcourir tous les éléments
        def analyze_element(elem, depth=0):
            result["total_elements"] += 1
            result["max_depth"] = max(result["max_depth"], depth)

            tag_name = _strip_namespace(elem.tag)
            result["all_tags"].add(tag_name)
            result["tag_counts"][tag_name] = result["tag_counts"].get(tag_name, 0) + 1

            # Collecter les attributs
            for attr in elem.attrib.keys():
                result["attributes_found"].add(f"{tag_name}@{attr}")

            # Sample de contenu (premier non-vide pour chaque tag)
            if tag_name not in result["sample_content"]:
                text = (elem.text or "").strip()
                if text:
                    result["sample_content"][tag_name] = text[:200] + ("..." if len(text) > 200 else "")

            for child in elem:
                analyze_element(child, depth + 1)

        analyze_element(root)

        # Convertir le set en liste triée pour JSON
        result["all_tags"] = sorted(result["all_tags"])
        result["attributes_found"] = sorted(result["attributes_found"])

    except ET.ParseError as e:
        logger.error(f"Erreur de parsing XML {xml_path}: {e}")
        result["error"] = str(e)
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse XML {xml_path}: {e}")
        result["error"] = str(e)

    return result


def _strip_namespace(tag: str) -> str:
    """Retire le namespace d'un tag XML"""
    if "}" in tag:
        return tag.split("}")[1]
    return tag


def extract_text_from_xml(
    xml_path: str,
    config: Optional[XMLParseConfig] = None
) -> str:
    """
    Extrait le texte d'un fichier XML selon la configuration spécifiée.

    Args:
        xml_path: Chemin vers le fichier XML
        config: Configuration de parsing (utilise FULL_TEXT par défaut)

    Returns:
        Texte extrait du XML
    """
    if config is None:
        config = XMLParseConfig()

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if config.strategy == XMLParseStrategy.FULL_TEXT:
            return _extract_full_text(root, config)
        elif config.strategy == XMLParseStrategy.STRUCTURED:
            return _extract_structured(root, config)
        elif config.strategy == XMLParseStrategy.TAG_FILTERED:
            return _extract_tag_filtered(root, config)
        elif config.strategy == XMLParseStrategy.XPATH_QUERY:
            return _extract_xpath(root, config)
        elif config.strategy == XMLParseStrategy.ATTRIBUTES_INCLUDED:
            return _extract_with_attributes(root, config)
        else:
            return _extract_full_text(root, config)

    except ET.ParseError as e:
        logger.error(f"Erreur de parsing XML {xml_path}: {e}")
        # Fallback: essayer de lire comme texte brut
        try:
            with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction XML {xml_path}: {e}")
        return ""


def _extract_full_text(root: ET.Element, config: XMLParseConfig) -> str:
    """Extrait tout le texte, ignore les balises"""
    texts = []

    def recurse(elem):
        tag_name = _strip_namespace(elem.tag)

        # Vérifier exclusions
        if config.excluded_tags and tag_name in config.excluded_tags:
            return

        if elem.text:
            text = elem.text if config.preserve_whitespace else elem.text.strip()
            if text:
                if config.add_tag_markers:
                    texts.append(f"[{tag_name}] {text}")
                else:
                    texts.append(text)

        for child in elem:
            recurse(child)

        if elem.tail:
            tail = elem.tail if config.preserve_whitespace else elem.tail.strip()
            if tail:
                texts.append(tail)

    recurse(root)
    return config.separator.join(texts)


def _extract_structured(root: ET.Element, config: XMLParseConfig) -> str:
    """Extrait le texte en gardant une structure indentée"""
    lines = []

    def recurse(elem, indent=0):
        tag_name = _strip_namespace(elem.tag)

        if config.excluded_tags and tag_name in config.excluded_tags:
            return

        prefix = "  " * indent

        # Ajouter le tag
        text = (elem.text or "").strip() if not config.preserve_whitespace else (elem.text or "")
        if text:
            lines.append(f"{prefix}[{tag_name}] {text}")
        elif len(elem) == 0:
            # Tag vide sans enfants
            pass
        else:
            lines.append(f"{prefix}[{tag_name}]")

        for child in elem:
            recurse(child, indent + 1)

    recurse(root)
    return "\n".join(lines)


def _extract_tag_filtered(root: ET.Element, config: XMLParseConfig) -> str:
    """Extrait uniquement le contenu des tags sélectionnés"""
    if not config.selected_tags:
        return _extract_full_text(root, config)

    texts = []

    def recurse(elem):
        tag_name = _strip_namespace(elem.tag)

        if tag_name in config.selected_tags:
            # Extraire tout le texte de cet élément et ses enfants
            full_text = ET.tostring(elem, encoding="unicode", method="text")
            text = full_text.strip() if not config.preserve_whitespace else full_text
            if text:
                if config.add_tag_markers:
                    texts.append(f"[{tag_name}] {text}")
                else:
                    texts.append(text)
        else:
            # Continuer à chercher dans les enfants
            for child in elem:
                recurse(child)

    recurse(root)
    return config.separator.join(texts)


def _extract_xpath(root: ET.Element, config: XMLParseConfig) -> str:
    """Extrait le texte via requêtes XPath"""
    if not config.xpath_queries:
        return _extract_full_text(root, config)

    texts = []

    for xpath in config.xpath_queries:
        try:
            elements = root.findall(xpath)
            for elem in elements:
                if isinstance(elem, str):
                    texts.append(elem)
                else:
                    full_text = ET.tostring(elem, encoding="unicode", method="text")
                    text = full_text.strip() if not config.preserve_whitespace else full_text
                    if text:
                        texts.append(text)
        except Exception as e:
            logger.warning(f"XPath query failed '{xpath}': {e}")

    return config.separator.join(texts)


def _extract_with_attributes(root: ET.Element, config: XMLParseConfig) -> str:
    """Extrait le texte en incluant les attributs"""
    texts = []

    def recurse(elem):
        tag_name = _strip_namespace(elem.tag)

        if config.excluded_tags and tag_name in config.excluded_tags:
            return

        parts = []

        # Ajouter les attributs
        if elem.attrib:
            attr_str = ", ".join([f"{k}={v}" for k, v in elem.attrib.items()])
            parts.append(f"[{tag_name}: {attr_str}]")
        elif config.add_tag_markers:
            parts.append(f"[{tag_name}]")

        # Ajouter le texte
        if elem.text:
            text = elem.text if config.preserve_whitespace else elem.text.strip()
            if text:
                parts.append(text)

        if parts:
            texts.append(" ".join(parts))

        for child in elem:
            recurse(child)

    recurse(root)
    return config.separator.join(texts)


def preview_xml_extraction(
    xml_path: str,
    config: XMLParseConfig,
    max_chars: int = 2000
) -> Tuple[str, Dict[str, Any]]:
    """
    Génère une prévisualisation de l'extraction XML.

    Returns:
        Tuple (texte_preview, stats)
    """
    full_text = extract_text_from_xml(xml_path, config)

    stats = {
        "total_chars": len(full_text),
        "total_words": len(full_text.split()),
        "total_lines": full_text.count("\n") + 1,
        "strategy": config.strategy.value,
    }

    preview = full_text[:max_chars]
    if len(full_text) > max_chars:
        preview += f"\n\n... [Tronqué - {len(full_text) - max_chars} caractères restants]"

    return preview, stats


def get_recommended_config(structure_info: Dict[str, Any]) -> XMLParseConfig:
    """
    Suggère une configuration de parsing basée sur l'analyse de la structure.
    """
    config = XMLParseConfig()

    # Si beaucoup de tags différents, suggérer STRUCTURED
    if len(structure_info.get("all_tags", [])) > 20:
        config.strategy = XMLParseStrategy.STRUCTURED
        config.add_tag_markers = True

    # Si des attributs importants, suggérer ATTRIBUTES_INCLUDED
    if len(structure_info.get("attributes_found", [])) > 5:
        config.strategy = XMLParseStrategy.ATTRIBUTES_INCLUDED
        config.include_attributes = True

    # Tags courants à exclure par défaut
    common_metadata_tags = {"meta", "head", "style", "script", "link"}
    found_metadata = common_metadata_tags.intersection(set(structure_info.get("all_tags", [])))
    if found_metadata:
        config.excluded_tags = list(found_metadata)

    return config


# Constantes pour les stratégies disponibles (pour l'UI)
STRATEGY_DESCRIPTIONS = {
    XMLParseStrategy.FULL_TEXT: "Texte complet - Extrait tout le texte en ignorant les balises",
    XMLParseStrategy.STRUCTURED: "Structuré - Garde l'indentation et les marqueurs de tags",
    XMLParseStrategy.TAG_FILTERED: "Filtré par tags - Extrait uniquement les tags sélectionnés",
    XMLParseStrategy.XPATH_QUERY: "XPath - Extraction via requêtes XPath personnalisées",
    XMLParseStrategy.ATTRIBUTES_INCLUDED: "Avec attributs - Inclut les attributs XML dans le texte",
}
