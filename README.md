# 🎭 Analyseur de Visages IA - Protection des Données

## 📋 Description du Projet

Ce projet a été développé dans le cadre d'un cours sur la **Protection des Données** pour illustrer les défis et enjeux liés au traitement de données biométriques par l'Intelligence Artificielle. Il démontre concrètement les implications du RGPD dans le développement d'applications d'IA.

## ⚠️ Avertissement RGPD

**Cette application traite des données biométriques sensibles selon l'Article 9 du RGPD.**

Avant d'utiliser cette application :
- ✅ Assurez-vous d'avoir le **consentement explicite** de toutes les personnes filmées
- ✅ Informez clairement sur la **finalité du traitement**
- ✅ Respectez les principes de **minimisation des données**
- ✅ Garantissez la **sécurité** des données collectées

## 🚀 Fonctionnalités

### Détection et Analyse
- 👤 **Détection de visages** - Identification automatique des visages
- 🎂 **Estimation d'âge** - Prédiction de la tranche d'âge
- ⚧️ **Classification du genre** - Identification du genre apparent
- 🌍 **Origine ethnique** - Estimation de l'origine ethnique (données sensibles)
- 😊 **Reconnaissance d'émotions** - Analyse de l'état émotionnel
- 📊 **Analyse statistique** - Génération de rapports et visualisations

### Modes d'Utilisation

#### 📤 Mode 1: Upload de Vidéo
- Upload de fichiers MP4, AVI, MOV, MKV
- Traitement complet de la vidéo
- Génération de rapport détaillé
- Téléchargement des résultats (vidéo annotée + données CSV)
- Barre de progression en temps réel

#### 📹 Mode 2: Temps Réel
- Flux vidéo en temps réel via webcam
- Détection instantanée avec annotations
- Statistiques live et historique
- Compatible avec caméra de téléphone (DroidCam, IP Webcam)
- Contrôles en temps réel

## 🛠️ Installation

### Prérequis
- Python 3.8 ou supérieur
- Webcam (pour le mode temps réel)
- 4GB RAM minimum (8GB recommandé)

### Installation des Dépendances

```bash
# Cloner le projet
git clone [URL_DU_PROJET]
cd protection-donnees-ia

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances Principales
- `opencv-python` - Traitement d'images et vidéos
- `mediapipe` - Détection de visages Google
- `deepface` - Analyse des attributs faciaux
- `streamlit` - Interface web
- `streamlit-webrtc` - Streaming vidéo temps réel
- `tensorflow` - Modèles d'apprentissage automatique
- `pandas` - Manipulation de données

## 🚀 Utilisation

### Lancement de l'Application Principale

```bash
streamlit run main_app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

### Lancement des Modes Séparément

#### Mode 1 (Upload Vidéo)
```bash
streamlit run app_mode1.py --server.port 8502
```

#### Mode 2 (Temps Réel)
```bash
streamlit run app_mode2.py --server.port 8503
```

## 📱 Utilisation avec Téléphone

### Option 1: DroidCam
1. Téléchargez **DroidCam** sur votre téléphone (Android/iOS)
2. Installez **DroidCam Client** sur votre PC
3. Connectez via WiFi ou USB
4. Sélectionnez DroidCam comme source vidéo

### Option 2: IP Webcam (Android)
1. Installez **IP Webcam** depuis Google Play Store
2. Lancez l'application et démarrez le serveur
3. Notez l'adresse IP affichée (ex: 192.168.1.100:8080)
4. Utilisez cette URL dans votre navigateur

## 🔒 Aspects de Protection des Données

### Données Sensibles Traitées (Article 9 RGPD)
- 🧬 **Données biométriques** (géométrie faciale)
- 🌍 **Origine raciale/ethnique** (estimation)
- ⚧️ **Données relatives au genre**
- 🧠 **État psychologique** (émotions)

### Mesures de Protection Implémentées
- ✅ **Traitement local** - Aucune donnée envoyée vers des serveurs externes
- ✅ **Transparence** - Information claire sur les données collectées
- ✅ **Minimisation** - Collecte uniquement des données nécessaires
- ✅ **Limitation de conservation** - Historique limité dans le temps
- ✅ **Droit à l'effacement** - Possibilité de supprimer les données
- ✅ **Privacy by Design** - Protection intégrée dès la conception

### Articles RGPD Concernés
- **Article 6** : Licéité du traitement
- **Article 9** : Traitement portant sur des catégories particulières de données à caractère personnel
- **Article 13-14** : Informations à fournir lorsque des données à caractère personnel sont collectées auprès de la personne concernée
- **Article 25** : Protection des données dès la conception et protection des données par défaut

## 📊 Structure du Projet

```
protection-donnees-ia/
├── main_app.py              # Application principale avec navigation
├── app_mode1.py             # Interface Mode 1 (Upload vidéo)
├── app_mode2.py             # Interface Mode 2 (Temps réel)
├── face_analyzer.py         # Module d'analyse de visages
├── requirements.txt         # Dépendances Python
├── README.md               # Documentation
└── examples/               # Exemples et captures d'écran
```

## 🎯 Objectifs Pédagogiques

Ce projet permet de :
- 🎓 **Comprendre** les implications du RGPD dans l'IA
- ⚖️ **Identifier** les données sensibles collectées
- 🛡️ **Appliquer** les principes de privacy by design
- 📋 **Documenter** les traitements de données
- 🔍 **Analyser** les risques liés aux données biométriques
- 💡 **Sensibiliser** aux enjeux éthiques de l'IA

## 🐛 Dépannage

### Problèmes Courants

#### Erreur d'installation de dépendances
```bash
# Mettre à jour pip
pip install --upgrade pip

# Installation avec cache désactivé
pip install --no-cache-dir -r requirements.txt
```

#### Problème de caméra (Mode 2)
- Vérifiez que votre navigateur a l'autorisation d'accéder à la caméra
- Fermez les autres applications utilisant la caméra
- Redémarrez le navigateur si nécessaire

#### Performance lente
- Réduisez la résolution de la caméra
- Fermez les autres applications gourmandes
- Utilisez un GPU si disponible

## 📝 Licence et Utilisation

Ce projet est destiné à des fins **éducatives uniquement**. 

### Restrictions d'Utilisation
- ❌ **Pas d'utilisation commerciale** sans autorisation
- ❌ **Pas de surveillance** sans consentement
- ❌ **Pas de stockage** de données biométriques sans base légale
- ✅ **Utilisation pédagogique** et recherche autorisée

---

**⚠️ Rappel Important :** Ce projet traite des données biométriques sensibles. Assurez-vous de respecter la réglementation en vigueur (RGPD, lois locales) avant toute utilisation. 