# 🧠 Machine Learning App with Streamlit

Une application **Auto ML** interactive développée avec **Streamlit** permettant de :
- charger vos jeux de données,  
- entraîner automatiquement plusieurs modèles de **Machine Learning** (avec réglage automatique des hyperparamètres),  
- effectuer des **prédictions**,  
- et réaliser du **clustering non supervisé** avec détection automatique du nombre optimal de clusters.

---

## 🖼️ Aperçu

![Streamlit App Screenshot](screenshot.png)  
*(Ajoutez ici une capture d’écran de votre interface une fois lancée)*

---

## 🚀 Fonctionnalités principales

### 🔹 1. Importation et Prétraitement
- Chargement d’un dataset CSV.
- Encodage automatique des variables catégorielles.
- Remplissage des valeurs manquantes.
- Normalisation des variables numériques (StandardScaler).
- Séparation automatique en ensembles **train/test**.

### 🔹 2. Entraînement de modèles supervisés
- **Modèles disponibles :**
  - Régression Logistique  
  - Forêt Aléatoire  
  - SVM (Support Vector Machine)  
  - K-Nearest Neighbors  
  - Arbre de Décision  
  - Gradient Boosting

- **Options avancées :**
  - Sélection automatique du meilleur modèle selon la précision.  
  - Réglage automatique des hyperparamètres via **GridSearchCV**.  
  - Sauvegarde du modèle entraîné au format `.pkl`.

### 🔹 3. Prédiction
- Chargement du modèle sauvegardé ou utilisation du modèle entraîné dans la session.  
- Entrée manuelle de nouvelles données pour prédire une classe.  
- Décodage automatique des labels si la cible était catégorielle.

### 🔹 4. Clustering (non supervisé)
- Algorithmes pris en charge :
  - K-Means (avec **détection automatique du coude** via *KneeLocator*)  
  - Agglomerative Clustering  
  - DBSCAN
- Évaluation automatique via le **Silhouette Score**.  
- Visualisation du coude pour K-Means.  
- Sauvegarde du modèle de clustering.

---

## ⚙️ Installation

### 🔹 1. Cloner le dépôt

- git clone https://github.com/BENMEZIAN/ml-app-streamlit.git
- cd ml-app-streamlit
- pip install -r requirements.txt

### 🔹 2. Lancer l’application
streamlit run ml.py

### 🔹 3. Utiliser l’application

- Chargez un dataset .csv
- Choisissez la colonne cible à prédire.
- Sélectionnez un ou plusieurs modèles.
- Activez la recherche d’hyperparamètres. (Optionnel) 
- Lancez l’entraînement 🚀
- Consultez les performances et sauvegardez le meilleur modèle.
- Passez à la section Prediction pour tester votre modèle sur de nouvelles données.
- Explorez la section Clustering pour découvrir les structures cachées dans vos données.
