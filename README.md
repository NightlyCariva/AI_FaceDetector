# Analyseur de Visages IA - Protection des Données

##  Description du Projet

Ce projet a été développé dans le cadre d'un cours sur la **Protection des Données** à  L'UPJV pour illustrer les défis et enjeux liés au traitement de données biométriques par l'Intelligence Artificielle. Il démontre concrètement le potentiel de l'IA et les raison de l'implications du RGPD dans le développement d'applications d'IA.

## ⚠️ Avertissement RGPD

**Cette application traite des données biométriques sensibles selon l'Article 9 du RGPD.**

Avant d'utiliser cette application :
- ✅ Assurez-vous d'avoir le **consentement explicite** de toutes les personnes filmées
- ✅ Informez clairement sur la **finalité du traitement**
- ✅ Respectez les principes de **minimisation des données**
- ✅ Garantissez la **sécurité** des données collectées

## Fonctionnalités

- **Détection de Visages**: Identification automatique des visages dans les images et vidéos
- **Estimation d'Age**: Prédiction de la tranche d'âge des personnes détectées
- **Classification Genre**: Identification du genre apparent des individus
- **Analyse Emotions**: Reconnaissance des expressions faciales et émotions
- **Estimation Ethnie**: Classification de l'origine ethnique apparente

## Modes Disponibles

### Mode 1: Upload Vidéo
- Upload de vidéos (MP4, AVI, MOV, MKV)
- Analyse complète frame par frame
- Export des résultats en CSV
- Téléchargement de la vidéo annotée

### Mode 2: Temps Réel
- Analyse en temps réel via webcam
- Détection instantanée des visages
- Statistiques live
- Export des sessions en CSV

## Installation

### Prérequis
- Python 3.10^
- Webcam (pour le mode temps réel)

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Lancement de l'application
```bash
streamlit run main.py
```

L'application sera accessible à l'adresse: `http://localhost:8501`

## Structure du Projet

```
├── main.py              # Point d'entrée principal
├── mode1_upload.py      # Mode upload vidéo
├── mode2_realtime.py    # Mode temps réel
├── face_detector.py     # Module de détection faciale
├── .env                 # Configuration
├── requirements.txt     # Dépendances
└── README.md           # Documentation
```

## Utilisation

1. Lancez l'application avec `streamlit run main.py`
2. Lisez et acceptez les conditions d'utilisation
3. Choisissez entre Mode 1 (Upload) ou Mode 2 (Temps Réel)
4. Configurez les paramètres d'analyse
5. Lancez l'analyse et consultez les résultats

## Format d'Export CSV

Les résultats sont exportés avec les colonnes suivantes:

- `face_id`: Identifiant unique du visage détecté
- `timestamp`: Horodatage de la détection (HH:MM:SS)
- `frame_number`: Numéro de l'image dans la séquence
- `age_estimation`: Tranche d'âge prédite (ex: 20-30)
- `gender_classification`: Genre apparent (Male/Female)
- `ethnicity_estimation`: Origine ethnique estimée
- `emotion`: Émotion dominante (Happy, Sad, etc.)

## Avertissement

Ce projet traite des données biométriques sensibles. Assurez-vous de respecter la réglementation en vigueur et d'obtenir le consentement approprié avant toute utilisation.

### Restrictions d'Utilisation
- ❌ **Pas d'utilisation commerciale** sans autorisation
- ❌ **Pas de surveillance** sans consentement
- ❌ **Pas de stockage** de données biométriques sans base légale
- ✅ **Utilisation pédagogique** et recherche autorisée

---

**⚠️ Rappel Important :** Ce projet est un projet universitaire à but éducatives uniquement, traite des données biométriques sensibles. Assurez-vous de 
respecter la réglementation en vigueur (RGPD, lois locales) avant toute utilisation.