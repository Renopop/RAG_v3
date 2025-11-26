# 📖 Guide Utilisateur RaGME_UP - PROP

**Bienvenue dans RaGME_UP - PROP !** Ce guide vous accompagne pas à pas pour utiliser l'application.

---

## 🚀 Démarrage rapide

### 1. Lancer l'application

```bash
streamlit run streamlit_RAG.py
```

L'application s'ouvre automatiquement dans votre navigateur sur `http://localhost:8501`

---

## 📋 Les 5 onglets de l'application

### 📝 **Onglet 1 : Gestion CSV**

Créez et gérez des fichiers CSV pour organiser vos documents avant l'ingestion.

#### Interface graphique moderne

Cet onglet utilise une **interface graphique de bureau (GUI)** au lieu d'une interface web, vous permettant d'accéder directement aux fichiers de votre système.

#### 🆕 Créer un nouveau CSV

1. Cliquez sur **"📝 Création d'un CSV"** dans l'onglet Gestion CSV
2. Une fenêtre GUI moderne s'ouvre automatiquement

**Méthode A : Scanner un répertoire**

1. Cliquez sur **"📂 Scanner un répertoire"**
2. Collez le chemin du dossier à scanner :
   - Ouvrez l'Explorateur Windows
   - Cliquez dans la barre d'adresse (ou `Ctrl+L`)
   - Copiez le chemin (`Ctrl+C`)
   - Collez dans le champ "Chemin du répertoire"
3. **Options de scan** :
   - ☑️ **Récursif** : inclut tous les sous-dossiers
   - **Extensions** : personnalisez les types de fichiers (PDF, DOCX, TXT par défaut)
4. Cliquez sur **"🔍 Lancer le scan"**
5. Résultat : tous les fichiers trouvés sont ajoutés à la liste

**Méthode B : Ajouter des fichiers manuellement**

1. Cliquez sur **"➕ Ajouter des fichiers"**
2. Sélectionnez un ou plusieurs fichiers dans la boîte de dialogue
3. Les fichiers sont automatiquement ajoutés à la liste

**Assigner les groupes (collections)**

- Chaque fichier peut être assigné à un groupe (= collection FAISS)
- Exemples de groupes : `CS`, `AMC`, `GM`, `ALL`
- Pour modifier un groupe : double-cliquez sur la cellule "Groupe" dans le tableau
- Pour appliquer le même groupe à plusieurs fichiers :
  1. Tapez le nom du groupe dans "Groupe par défaut"
  2. Sélectionnez les lignes dans le tableau
  3. Le groupe est appliqué automatiquement

**Sauvegarder le CSV**

1. Tapez le nom du CSV (sans extension) dans le champ "Nom du CSV" : `mes_documents`
2. Cliquez sur **"💾 Sauvegarder le CSV"**
3. Le CSV est **automatiquement sauvegardé** dans le répertoire configuré
4. Si le fichier existe déjà, une confirmation vous sera demandée
5. Le nom du CSV devient le nom de la base FAISS

**Note** : Plus besoin de choisir manuellement l'emplacement, tout est sauvegardé au bon endroit automatiquement !

#### ✏️ Modifier un CSV existant

**Option 1 : Depuis l'interface Streamlit**

1. Cliquez sur **"✏️ Gestion des CSV"** dans l'onglet Gestion CSV
2. Sélectionnez le CSV à modifier dans le menu déroulant
3. Cliquez sur **"Ouvrir pour édition"**
4. La GUI s'ouvre avec le contenu du CSV chargé

**Option 2 : Depuis l'Explorateur Windows**

1. Ouvrez l'Explorateur et naviguez vers votre répertoire CSV configuré
2. Double-cliquez sur le fichier CSV à modifier
3. La GUI s'ouvre automatiquement (si configuré)

**Édition du CSV**

1. Le contenu s'affiche dans le tableau avec :
   - ✅ Fichiers existants (chemins valides)
   - ❌ Fichiers manquants (chemins invalides) - affichés en rouge
2. Vous pouvez :
   - Modifier les groupes (double-clic sur la cellule)
   - Supprimer des lignes avec **"🗑️ Supprimer sélection"**
   - Ajouter de nouveaux fichiers avec **"➕ Ajouter des fichiers"** ou **"📂 Scanner un répertoire"**
   - Vider complètement la liste avec **"🧹 Tout effacer"**
3. Cliquez sur **"💾 Sauvegarder le CSV"** pour enregistrer les modifications

---

### 📥 **Onglet 2 : Ingestion documents**

Indexez vos documents dans FAISS pour pouvoir les interroger.

#### 🌐 Compatible partages réseau Windows

**Important** : Le système utilise maintenant FAISS au lieu de ChromaDB pour une **parfaite compatibilité avec les partages réseau Windows** :
- ✅ Pas de problèmes de verrous de fichiers SQLite
- ✅ Synchronisation automatique sur réseau
- ✅ Multi-utilisateurs sans conflit
- ✅ Plus rapide et plus fiable

#### 📄 Ingestion via CSV (méthode recommandée)

**Pourquoi cette méthode ?**
- Gère de gros volumes facilement
- Organisée et traçable
- Évite automatiquement les doublons
- Fonctionne en multi-utilisateurs
- **Compatible réseau Windows grâce à FAISS**

**Étapes :**

1. **Préparez votre CSV** dans l'onglet "Gestion CSV" (ou utilisez un existant)

2. **Uploadez le CSV** :
   - Cliquez sur "Upload CSV contenant des chemins de fichiers"
   - Sélectionnez votre fichier `mes_documents.csv`
   - ⚠️ **Important** : Le nom du CSV = nom de la base FAISS
     - `normes_easa.csv` → base `normes_easa`
     - `manuels.csv` → base `manuels`

3. **Lancez l'ingestion** :
   - Cliquez sur **"🚀 Lancer l'ingestion"**
   - Une barre de progression s'affiche
   - Les logs détaillent chaque étape

4. **Résultats** :
   - **Nouveaux fichiers** : documents ingérés avec succès
   - **Fichiers manquants** : fichiers introuvables (chemins invalides)
   - **Déjà présents (skipped)** : fichiers déjà ingérés (évite les doublons)
   - **Pièces jointes** : pièces jointes PDF extraites et ingérées automatiquement

5. **Fichier de tracking créé** :
   - `documents_ingeres_[nom_base].csv` dans le dossier CSV
   - Permet d'éviter les doublons automatiquement lors des prochaines ingestions
   - Contient tous les fichiers déjà ingérés dans cette base

#### 🤖 Que fait l'ingestion automatiquement ?

✅ **Extraction intelligente avec fallback robuste**
- Extrait le texte des PDF avec pdfminer.six
- Si échec, utilise PyMuPDF comme fallback (plus robuste)
- **Extrait les pièces jointes PDF et les ingère automatiquement**
- Extrait le contenu des DOCX, DOC, TXT, MD, CSV
- Détecte automatiquement l'encodage des fichiers
- **Gère les caractères Unicode complexes (surrogates)**

✅ **Traitement parallèle optimisé**
- Extraction multi-threads pour meilleure performance
- Compatible Windows avec PyMuPDF (pas de crashes mémoire)
- Barre de progression en temps réel

✅ **Détection EASA**
- Détecte les sections CS, AMC, GM automatiquement
- Exemple : `CS 25.613 Fatigue evaluation`
- Stocke les métadonnées pour recherche précise

✅ **Chunking intelligent**
- Découpe en morceaux de ~1000 caractères
- Overlap de 150 caractères pour garder le contexte
- Respecte les frontières de sentences

✅ **Déduplication**
- Vérifie le CSV de tracking de la base
- Skip automatiquement les fichiers déjà ingérés
- Évite les doublons même sur plusieurs sessions

✅ **Stockage FAISS réseau**
- Sauvegarde automatique après chaque ajout
- Compatible partages réseau Windows
- Pas de problèmes de verrous (pas de SQLite)

---

### 🗑️ **Onglet 3 : Purge des bases**

Supprimez tout le contenu d'une base (les collections sont vidées mais pas supprimées).

#### ⚠️ Attention : Action irréversible !

**Étapes :**

1. **Sélectionnez la base** à purger dans le menu déroulant

2. **Consultez les statistiques** :
   - Nombre de collections
   - Total de chunks indexés
   - Détail par collection
   - CSV de tracking associé

3. **Confirmez la purge** :
   - Tapez **exactement** le nom de la base : `normes_easa`
   - Le bouton **"🗑️ PURGER LA BASE"** devient actif

4. **Cliquez sur PURGER LA BASE** :
   - Toutes les collections sont vidées
   - Le CSV de tracking est supprimé
   - Un résumé détaillé s'affiche

5. **Rechargez la page** pour voir les changements

**Quand utiliser la purge ?**
- Vous voulez réinitialiser complètement une base
- Vous avez ingéré de mauvaises données
- Vous voulez repartir de zéro avec une nouvelle organisation

---

### ❓ **Onglet 4 : Questions RAG**

Posez des questions sur vos documents indexés et obtenez des réponses contextuelles.

#### 🎯 Sélection de la base et collection

**En haut de l'onglet** :

1. **Sélectionnez une base** dans le menu déroulant :
   - Liste toutes les bases FAISS disponibles
   - Exemple : `normes_easa`, `manuels`, etc.

2. **Sélectionnez une collection** dans le menu déroulant :
   - `CS` : seulement les Certification Specifications
   - `AMC` : seulement les Acceptable Means of Compliance
   - `GM` : seulement les Guidance Material
   - `ALL` : toutes les collections (recherche globale)

#### 💬 Poser une question

**Étapes :**

1. **Tapez votre question** dans la zone de texte :
   - Exemple : *"What are the fatigue evaluation requirements for CS 25?"*
   - Soyez précis et clair
   - Utilisez des termes techniques présents dans vos documents

2. **Cliquez sur "🤖 Poser la question"**

3. **Résultat** :
   - 🧠 **Réponse** du LLM basée sur vos documents
   - 📚 **Sources** citées avec :
     - 🟢 Score élevé (≥ 0.8) = très pertinent
     - 🟠 Score moyen (0.6-0.8) = pertinent
     - 🔴 Score faible (< 0.6) = peu pertinent
   - 📄 **Bouton "Ouvrir"** pour ouvrir le document source dans son application par défaut
   - 🧩 Contexte brut utilisé (pour debug)

#### 📂 Ouvrir les documents sources

Le bouton **"Ouvrir"** à côté de chaque source permet d'ouvrir directement le fichier dans son application par défaut (Adobe Reader pour PDF, Word pour DOCX, etc.).

**Avantages** :
- ✅ Vérifiez la source dans son contexte complet
- ✅ Les résultats de recherche restent affichés (pas d'effacement)
- ✅ Ouverture automatique dans l'application appropriée

#### 🔍 Comprendre les sources

Chaque source affiche :
- **Nom du fichier** : `CS_25.pdf`
- **Chunk ID** : identifiant du morceau de texte
- **Score** : pertinence (0 = pas pertinent, 1 = très pertinent)
- **Distance** : distance L2 FAISS (plus petit = meilleur)
- **Section EASA** : si détectée (ex: `CS 25.613`)
- **Passage utilisé** : le texte exact récupéré de vos documents

#### 🔄 Amélioration par retours utilisateurs (Re-ranking)

Une option **"🔄 Utiliser les retours utilisateurs pour améliorer les résultats"** permet d'activer le re-ranking intelligent :

- **Sources bien notées** : les sources ayant reçu de bons feedbacks sont favorisées
- **Sources mal notées** : les sources ayant reçu de mauvais feedbacks sont pénalisées
- **Questions similaires** : si une question similaire a déjà été posée et évaluée, le système utilise cette information pour améliorer les résultats

> 💡 Plus vous donnez de feedbacks, plus le système s'améliore !

#### 📝 Donner votre avis (Feedback simplifié)

Après chaque réponse, deux boutons apparaissent :

- **👍 Oui** : La réponse vous a aidé
- **👎 Non** : La réponse n'est pas satisfaisante

**Si vous cliquez 👎 :**
Un champ texte s'affiche pour décrire la **réponse que vous attendiez**. Cette information est précieuse pour améliorer les futures recherches !

> 💡 Plus vous donnez de feedbacks, plus le système s'améliore pour tous les utilisateurs !

---

### 📊 **Onglet 5 : Tableau de bord analytique**

Visualisez les statistiques et tendances des retours utilisateurs.

#### 📊 Filtres

- **Base à analyser** : sélectionnez une base spécifique ou "Toutes les bases"
- **Période d'analyse** : 7, 14, 30, 60 ou 90 derniers jours

#### 📈 Métriques globales

- **Total feedbacks** : nombre total de feedbacks enregistrés
- **👍 Positifs** : nombre de réponses jugées utiles
- **Taux de satisfaction** : pourcentage de feedbacks positifs

#### 📉 Graphiques de tendance

- **Évolution des feedbacks** : graphique en barres montrant les feedbacks positifs et négatifs par jour

#### 📋 Statistiques détaillées

- **Satisfaction par collection** : tableau avec les feedbacks 👍/👎 par collection
- **Questions avec feedback négatif** : liste des questions où les utilisateurs ont cliqué 👎, avec la réponse attendue
- **Activité par utilisateur** : répartition des feedbacks par utilisateur

#### 📥 Export des données

- **Exporter en CSV** : téléchargez tous les feedbacks au format CSV
- **Rafraîchir les statistiques** : mettez à jour les données affichées

---

## ❓ FAQ - Questions fréquentes

### Installation et Réseau

**Q : Puis-je utiliser l'application sur un partage réseau Windows ?**
- ✅ **Oui !** FAISS est conçu pour fonctionner parfaitement sur réseau
- ✅ Pas de problèmes de verrous (contrairement à ChromaDB/SQLite)
- ✅ Plusieurs utilisateurs peuvent travailler simultanément
- ✅ Synchronisation automatique des fichiers

**Q : Où sont stockées mes données ?**
- Configuré dans `streamlit_RAG.py` (lignes 48-51)
- Par défaut sur partage réseau : `N:\...\FAISS_DATABASE\`
- Bases FAISS : `BaseDB\[nom_base]`
- CSV tracking : `Fichiers_Tracking_CSV\documents_ingeres_[nom_base].csv`
- CSV ingestion : `CSV_Ingestion\[nom].csv`

### Ingestion

**Q : Les pièces jointes PDF sont-elles gérées ?**
- ✅ **Oui, automatiquement !**
- Extraction et ingestion des fichiers joints (PDF, images, etc.)
- Gestion des noms de fichiers avec caractères spéciaux
- Extensions préservées automatiquement

**Q : Que se passe-t-il avec des PDFs contenant des caractères spéciaux ?**
- ✅ Gestion automatique des caractères Unicode surrogates
- ✅ Nettoyage des noms de fichiers invalides
- ✅ Préservation des extensions (.pdf, .docx, etc.)

**Q : Pourquoi l'ingestion utilise des threads au lieu de processus ?**
- Meilleure compatibilité Windows avec PyMuPDF
- Pas de MemoryError ou crashes de workers
- Chargement parallèle toujours actif et performant

### Performance

**Q : FAISS est-il plus rapide que ChromaDB ?**
- ✅ **Oui, nettement plus rapide !**
- Recherche vectorielle optimisée
- Pas de couche SQLite (overhead réduit)
- Meilleure scalabilité

**Q : Combien de documents puis-je indexer ?**
- Pas de limite théorique
- Testé avec plusieurs milliers de documents
- Performance stable même sur partage réseau

### Requêtes

**Q : Quelle est la différence entre distance ChromaDB et FAISS ?**
- ChromaDB utilisait la distance cosinus
- FAISS utilise la distance L2 (euclidienne)
- Les résultats sont comparables
- Plus petit score = plus pertinent

---

## 🆘 Besoin d'aide ?

### Logs

Les logs détaillés sont dans : `rag_da_debug.log`

Consultez-les en cas d'erreur pour voir ce qui s'est passé.

### Contact

Pour toute question ou problème, contactez l'équipe de développement RaGME_UP - PROP.

---

## 🎯 Workflow recommandé

### Pour démarrer un nouveau projet

1. **Organisez vos documents** dans un ou plusieurs dossiers
2. **Créez un CSV** via l'onglet "Gestion CSV"
3. **Lancez l'ingestion** via l'onglet "Ingestion documents"
4. **Posez vos questions** via l'onglet "Questions RAG"

### Pour ajouter des documents à une base existante

1. **Créez un CSV** avec uniquement les nouveaux fichiers
2. **Nommez-le comme la base existante** : `ma_base.csv`
3. **Lancez l'ingestion** : les doublons seront skippés automatiquement

### Travail en équipe sur réseau

1. **Configurez les chemins réseau** dans `streamlit_RAG.py`
2. **Partagez le répertoire FAISS** avec droits lecture/écriture
3. **Chaque utilisateur** peut ingérer et requêter simultanément
4. **Les requêtes RAG** peuvent être faites en parallèle sans problème

---

## 🆕 Nouveautés de cette version

### 📝 Système de feedback utilisateur simplifié (NOUVEAU)
- 👍👎 **Feedback rapide** : un simple clic pouce haut ou pouce bas
- 💡 **Réponse attendue** : champ pour indiquer la réponse souhaitée si 👎
- 📊 **Tableau de bord** : taux de satisfaction et questions problématiques
- 🔟 **10 sources affichées** : plus de contexte pour chaque réponse

### 🔄 Re-ranking intelligent (NOUVEAU)
- 🎯 **Amélioration par feedbacks** : apprentissage à partir des 👍 et 👎
- 🔍 **Questions similaires** : utilisation des feedbacks de questions passées
- ⚙️ **Option activable** : checkbox "Utiliser les retours utilisateurs"

### Migration FAISS
- ✨ **FAISS remplace ChromaDB** pour une meilleure compatibilité réseau Windows
- 🚀 **Plus rapide** : recherche vectorielle optimisée
- 🌐 **Compatible réseau** : pas de problèmes de verrous SQLite
- 💾 **Auto-save** : sauvegarde après chaque ajout

### Extraction PDF robuste
- 🔧 **Gestion Unicode** : caractères surrogates nettoyés automatiquement
- 📎 **Pièces jointes** : extraction et ingestion automatiques
- 🧵 **Threads** : traitement parallèle sans crashes Windows
- ✅ **Extensions préservées** : .pdf, .docx, etc. correctement conservés

### Corrections critiques
- 🐛 Fix validation longueurs dans FAISS (chunks perdus)
- 🐛 Fix clean_filename() pour préserver extensions
- 🐛 Fix multiprocessing Windows + PyMuPDF (MemoryError)
- 🐛 Fix caractères surrogates dans noms de fichiers

### Amélioration technique
- 🔄 PyMuPDF fallback pour extraction PDF robuste
- 💾 Persistance des résultats RAG
- 🎯 Interface optimisée pour partages réseau
- 📝 Code nettoyé et optimisé

---

**Bon RAG avec RaGME_UP - PROP ! 🚀**
