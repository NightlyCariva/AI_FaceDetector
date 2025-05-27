#!/usr/bin/env python3
"""
Script de lancement pour l'Analyseur de Visages IA
Projet de Protection des Données
"""

import subprocess
import sys
import os
import argparse
import webbrowser
import time

def check_dependencies():
    """Vérifie que les dépendances sont installées"""
    try:
        import streamlit
        import cv2
        import mediapipe
        import deepface
        import tensorflow
        print("✅ Toutes les dépendances sont installées")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("📦 Installez les dépendances avec: pip install -r requirements.txt")
        return False

def install_dependencies():
    """Installe les dépendances automatiquement"""
    print("📦 Installation des dépendances...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dépendances installées avec succès")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'installation des dépendances")
        return False

def run_app(mode="main", port=8501, auto_open=True):
    """Lance l'application Streamlit"""
    
    # Mapping des modes vers les fichiers
    mode_files = {
        "main": "main_app.py",
        "mode1": "app_mode1.py", 
        "mode2": "app_mode2.py"
    }
    
    if mode not in mode_files:
        print(f"❌ Mode invalide: {mode}")
        print(f"Modes disponibles: {list(mode_files.keys())}")
        return False
    
    file_to_run = mode_files[mode]
    
    if not os.path.exists(file_to_run):
        print(f"❌ Fichier non trouvé: {file_to_run}")
        return False
    
    print(f"🚀 Lancement de l'application: {file_to_run}")
    print(f"🌐 URL: http://localhost:{port}")
    print("⏹️  Appuyez sur Ctrl+C pour arrêter")
    
    # Ouvrir automatiquement le navigateur
    if auto_open:
        def open_browser():
            time.sleep(2)  # Attendre que le serveur démarre
            webbrowser.open(f"http://localhost:{port}")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        # Lancer Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            file_to_run, 
            "--server.port", str(port),
            "--server.headless", "true" if not auto_open else "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée")
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        return False
    
    return True

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Analyseur de Visages IA - Protection des Données",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python run.py                    # Lance l'application principale
  python run.py --mode mode1       # Lance le mode upload vidéo
  python run.py --mode mode2       # Lance le mode temps réel
  python run.py --install          # Installe les dépendances
  python run.py --check            # Vérifie les dépendances
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["main", "mode1", "mode2"],
        default="main",
        help="Mode à lancer (default: main)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port pour le serveur Streamlit (default: 8501)"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Ne pas ouvrir automatiquement le navigateur"
    )
    
    parser.add_argument(
        "--install",
        action="store_true",
        help="Installer les dépendances"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Vérifier les dépendances"
    )
    
    args = parser.parse_args()
    
    # Affichage du header
    print("🎭" + "="*60)
    print("    ANALYSEUR DE VISAGES IA - PROTECTION DES DONNÉES")
    print("="*63)
    print()
    
    # Avertissement RGPD
    print("⚠️  AVERTISSEMENT RGPD:")
    print("   Cette application traite des données biométriques sensibles.")
    print("   Assurez-vous d'avoir le consentement approprié avant utilisation.")
    print()
    
    # Actions spéciales
    if args.install:
        return install_dependencies()
    
    if args.check:
        return check_dependencies()
    
    # Vérification des dépendances
    if not check_dependencies():
        response = input("Voulez-vous installer les dépendances maintenant? (y/N): ")
        if response.lower() in ['y', 'yes', 'oui']:
            if not install_dependencies():
                return False
        else:
            print("❌ Impossible de continuer sans les dépendances")
            return False
    
    # Lancement de l'application
    return run_app(
        mode=args.mode,
        port=args.port,
        auto_open=not args.no_browser
    )

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 