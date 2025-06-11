#!/usr/bin/env python3
"""
Entry Point
"""

import subprocess
import sys
import os

def main():
    """Lance l'application Streamlit"""
    
    print("=" * 50)
    print("   Face Detector")
    print("   Projet UPJV - Protection des Données")
    print("=" * 50)
    
    # Vérifier que Streamlit est installé
    try:
        import streamlit
    except ImportError:
        print("Streamlit non trouvé. Installez les dépendances avec:")
        print("utiliser la commande: pip install -r requirements.txt")
        sys.exit(1)

    if not os.path.exists("main.py"):
        print("main.py non trouvé !")
        sys.exit(1)
    
    print("\nLancement de l'application...")
    print("L'application sera accessible à: http://localhost:8501")
    print("\nAppuyez sur Ctrl+C pour arrêter\n")
    
    # Lancer Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n\nApplication arrêtée par l'utilisateur")
    except subprocess.CalledProcessError as e:
        print(f"\nErreur lors du lancement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 