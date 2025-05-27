#!/usr/bin/env python3
"""
Test simple pour vérifier que l'analyseur fonctionne sans FER/moviepy
"""

import sys
import numpy as np

def test_imports():
    """Teste les imports essentiels"""
    print("🧪 Test des imports...")
    
    try:
        import cv2
        print("   ✅ OpenCV - OK")
    except ImportError as e:
        print(f"   ❌ OpenCV - {e}")
        return False
    
    try:
        import mediapipe
        print("   ✅ MediaPipe - OK")
    except ImportError as e:
        print(f"   ❌ MediaPipe - {e}")
        return False
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        print("   ✅ TensorFlow - OK")
    except ImportError as e:
        print(f"   ❌ TensorFlow - {e}")
        return False
    
    try:
        from deepface import DeepFace
        print("   ✅ DeepFace - OK")
    except ImportError as e:
        print(f"   ❌ DeepFace - {e}")
        return False
    
    try:
        import streamlit
        print("   ✅ Streamlit - OK")
    except ImportError as e:
        print(f"   ❌ Streamlit - {e}")
        return False
    
    return True

def test_face_analyzer():
    """Teste l'analyseur de visages simplifié"""
    print("\n🎭 Test de l'analyseur de visages...")
    
    try:
        from face_analyzer_simple import FaceAnalyzer
        
        # Créer l'analyseur
        analyzer = FaceAnalyzer()
        print("   ✅ Initialisation - OK")
        
        # Créer une image de test
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Tester la détection de visages
        faces = analyzer.detect_faces(test_image)
        print(f"   ✅ Détection de visages - OK (trouvé: {len(faces)} visages)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def main():
    """Fonction principale"""
    print("🎭" + "="*50)
    print("    TEST SIMPLE - ANALYSEUR DE VISAGES")
    print("="*53)
    
    # Test des imports
    if not test_imports():
        print("\n❌ Échec des imports. Installez les dépendances:")
        print("   py -3.10 -m pip install -r requirements_simple.txt")
        return False
    
    # Test de l'analyseur
    if not test_face_analyzer():
        print("\n❌ Échec du test de l'analyseur")
        return False
    
    print("\n🎉 Tous les tests sont réussis!")
    print("\n🚀 Vous pouvez maintenant lancer:")
    print("   py -3.10 -m streamlit run app_mode1.py")
    print("   (en utilisant face_analyzer_simple.py)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 