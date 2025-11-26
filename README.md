# 🚀 RaGME_UP - PROP

Système RAG (Retrieval-Augmented Generation) pour l'indexation et l'interrogation de documents techniques avec FAISS, Snowflake Arctic Embeddings et DALLEM. Inclut un système de feedback utilisateur avec re-ranking intelligent.

---

## 📖 Documentation

- **[Guide Utilisateur](GUIDE_UTILISATEUR.md)** - Documentation complète pour utiliser l'application
- **[Installation Réseau](INSTALLATION_RESEAU.md)** - Guide de déploiement multi-utilisateurs
- **[Synthèse Développement](SYNTHESE_DEVELOPPEMENT.md)** - Documentation technique complète

---

## ⚡ Démarrage rapide

### Installation

```bash
# Windows: double-cliquez sur
install.bat
```

### Lancement

```bash
# Windows: double-cliquez sur
launch.bat

# Ou manuellement
streamlit run streamlit_RAG.py
```

L'application s'ouvre automatiquement dans votre navigateur sur `http://localhost:8501`

---

## ✨ Fonctionnalités principales

- 📝 **Gestion CSV** avec interface GUI moderne
- 📥 **Ingestion documents** (PDF, DOCX, TXT) avec tracking automatique
- 🔒 **Coordination multi-utilisateurs** avec système de verrous
- 🗑️ **Purge des bases** FAISS
- ❓ **Questions RAG** avec recherche sémantique et génération de réponses
- 📝 **Feedback utilisateur** : évaluation granulaire des réponses et sources
- 🔄 **Re-ranking intelligent** : amélioration des résultats basée sur les feedbacks
- 📊 **Tableau de bord analytique** : statistiques et tendances des retours
- 👥 **Authentification** utilisateurs pour l'accès aux paramètres

---

## 📋 Prérequis

- Python 3.8 ou supérieur
- Windows 10/11 (ou Linux/macOS avec adaptations)
- Accès réseau pour API Snowflake et DALLEM (ou mode test local)

---

## 🆘 Support

Consultez la documentation pour toute question :
- Questions d'utilisation → [Guide Utilisateur](GUIDE_UTILISATEUR.md)
- Installation réseau → [Installation Réseau](INSTALLATION_RESEAU.md)
- Développement/maintenance → [Synthèse Développement](SYNTHESE_DEVELOPPEMENT.md)

---

**Version:** 1.1
**Dernière mise à jour:** 2025-01-24
