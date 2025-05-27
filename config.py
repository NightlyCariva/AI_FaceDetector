"""
Configuration pour l'Analyseur de Visages IA
Projet de Protection des Données
"""

import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# Créer les dossiers s'ils n'existent pas
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Configuration de l'analyse de visages
FACE_DETECTION_CONFIG = {
    "model_selection": 0,  # 0 pour courte distance, 1 pour longue distance
    "min_detection_confidence": 0.5,
    "max_faces": 10,  # Nombre maximum de visages à détecter par frame
}

# Configuration de l'analyse des attributs
DEEPFACE_CONFIG = {
    "actions": ['age', 'gender', 'race', 'emotion'],
    "enforce_detection": False,
    "silent": True,
    "detector_backend": 'opencv',  # opencv, ssd, dlib, mtcnn, retinaface
    "model_name": 'VGG-Face',  # VGG-Face, Facenet, OpenFace, DeepFace
}

# Configuration vidéo
VIDEO_CONFIG = {
    "supported_formats": ['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
    "max_file_size_mb": 500,  # Taille maximum en MB
    "output_codec": 'mp4v',
    "output_fps": None,  # None pour garder le FPS original
    "max_resolution": (1920, 1080),  # Résolution maximum
}

# Configuration temps réel
REALTIME_CONFIG = {
    "max_history_frames": 100,  # Nombre de frames à garder en historique
    "fps_calculation_window": 30,  # Fenêtre pour calculer les FPS
    "processing_interval": 1,  # Traiter 1 frame sur N
    "max_processing_time": 0.1,  # Temps maximum par frame (secondes)
}

# Configuration de l'interface
UI_CONFIG = {
    "page_title": "Analyseur de Visages IA",
    "page_icon": "🎭",
    "layout": "wide",
    "theme": {
        "primary_color": "#1f77b4",
        "background_color": "#ffffff",
        "secondary_background_color": "#f0f2f6",
        "text_color": "#262730",
    }
}

# Configuration RGPD et sécurité
PRIVACY_CONFIG = {
    "data_retention_hours": 24,  # Durée de conservation des données temporaires
    "auto_cleanup": True,  # Nettoyage automatique des fichiers temporaires
    "anonymize_logs": True,  # Anonymiser les logs
    "consent_required": True,  # Exiger le consentement
    "audit_trail": True,  # Enregistrer les actions
}

# Messages d'avertissement RGPD
GDPR_WARNINGS = {
    "main_warning": """
    ⚠️ AVERTISSEMENT RGPD
    Cette application traite des données biométriques sensibles selon l'Article 9 du RGPD.
    Assurez-vous d'avoir le consentement explicite de toutes les personnes concernées.
    """,
    
    "data_types": [
        "🧬 Données biométriques (géométrie faciale)",
        "🌍 Origine raciale/ethnique (estimation)",
        "⚧️ Données relatives au genre",
        "🧠 État psychologique (émotions)",
        "🎂 Estimation d'âge"
    ],
    
    "recommendations": [
        "✅ Obtenir un consentement explicite",
        "✅ Informer sur la finalité du traitement",
        "✅ Limiter la conservation des données",
        "✅ Assurer la sécurité des données",
        "✅ Permettre l'exercice des droits"
    ]
}

# Configuration des modèles ML
ML_CONFIG = {
    "tensorflow_log_level": "ERROR",  # Réduire les logs TensorFlow
    "gpu_memory_growth": True,  # Croissance progressive de la mémoire GPU
    "model_cache": True,  # Cache des modèles chargés
    "parallel_processing": False,  # Traitement parallèle (expérimental)
}

# Configuration de performance
PERFORMANCE_CONFIG = {
    "low_memory_mode": False,  # Mode économie de mémoire
    "adaptive_quality": True,  # Qualité adaptative selon les performances
    "max_concurrent_analyses": 1,  # Nombre d'analyses simultanées
    "frame_skip_threshold": 0.2,  # Seuil pour ignorer des frames (secondes)
}

# Configuration de logging
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": True,
    "log_file": "face_analyzer.log",
    "max_log_size_mb": 10,
    "backup_count": 3,
}

# Configuration de développement
DEV_CONFIG = {
    "debug_mode": False,
    "show_processing_time": True,
    "save_debug_images": False,
    "verbose_errors": True,
}

# Validation de la configuration
def validate_config():
    """Valide la configuration et affiche des avertissements si nécessaire"""
    warnings = []
    
    # Vérifier les limites de mémoire
    if FACE_DETECTION_CONFIG["max_faces"] > 20:
        warnings.append("⚠️ Nombre maximum de visages élevé, peut impacter les performances")
    
    # Vérifier la taille des fichiers
    if VIDEO_CONFIG["max_file_size_mb"] > 1000:
        warnings.append("⚠️ Taille maximum de fichier élevée, peut causer des problèmes de mémoire")
    
    # Vérifier la configuration de confidentialité
    if not PRIVACY_CONFIG["consent_required"]:
        warnings.append("⚠️ Le consentement n'est pas requis - Attention RGPD!")
    
    if warnings:
        print("Configuration - Avertissements:")
        for warning in warnings:
            print(f"  {warning}")
    
    return len(warnings) == 0

# Fonction pour obtenir la configuration d'un module
def get_config(module_name):
    """Retourne la configuration pour un module spécifique"""
    configs = {
        "face_detection": FACE_DETECTION_CONFIG,
        "deepface": DEEPFACE_CONFIG,
        "video": VIDEO_CONFIG,
        "realtime": REALTIME_CONFIG,
        "ui": UI_CONFIG,
        "privacy": PRIVACY_CONFIG,
        "ml": ML_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "dev": DEV_CONFIG,
    }
    
    return configs.get(module_name, {})

# Fonction pour mettre à jour la configuration
def update_config(module_name, updates):
    """Met à jour la configuration d'un module"""
    global FACE_DETECTION_CONFIG, DEEPFACE_CONFIG, VIDEO_CONFIG
    global REALTIME_CONFIG, UI_CONFIG, PRIVACY_CONFIG
    global ML_CONFIG, PERFORMANCE_CONFIG, LOGGING_CONFIG, DEV_CONFIG
    
    configs = {
        "face_detection": FACE_DETECTION_CONFIG,
        "deepface": DEEPFACE_CONFIG,
        "video": VIDEO_CONFIG,
        "realtime": REALTIME_CONFIG,
        "ui": UI_CONFIG,
        "privacy": PRIVACY_CONFIG,
        "ml": ML_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "dev": DEV_CONFIG,
    }
    
    if module_name in configs:
        configs[module_name].update(updates)
        return True
    
    return False

# Initialisation
if __name__ == "__main__":
    print("Configuration de l'Analyseur de Visages IA")
    print("=" * 50)
    validate_config()
    print("Configuration validée ✅") 