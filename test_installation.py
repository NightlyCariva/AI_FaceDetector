#!/usr/bin/env python3
"""
Script de test pour vérifier l'installation de l'Analyseur de Visages IA
Projet de Protection des Données
"""

import sys
import importlib
import subprocess
import platform
import cv2
import numpy as np

def test_python_version():
    """Teste la version de Python"""
    print("🐍 Test de la version Python...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Version trop ancienne")
        print("   📋 Python 3.8+ requis")
        return False

def test_dependencies():
    """Teste l'installation des dépendances"""
    print("\n📦 Test des dépendances...")
    
    dependencies = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("tensorflow", "tensorflow"),
        ("streamlit", "streamlit"),
        ("deepface", "deepface"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("streamlit-webrtc", "streamlit_webrtc"),
    ]
    
    results = []
    
    for package_name, import_name in dependencies:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'Version inconnue')
            print(f"   ✅ {package_name} ({version}) - OK")
            results.append(True)
        except ImportError:
            print(f"   ❌ {package_name} - Non installé")
            results.append(False)
        except Exception as e:
            print(f"   ⚠️  {package_name} - Erreur: {e}")
            results.append(False)
    
    return all(results)

def test_opencv_functionality():
    """Teste les fonctionnalités OpenCV"""
    print("\n📷 Test des fonctionnalités OpenCV...")
    
    try:
        # Test de création d'image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test de détection de contours
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Test d'écriture de texte
        cv2.putText(img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print("   ✅ Fonctionnalités OpenCV - OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur OpenCV: {e}")
        return False

def test_mediapipe():
    """Teste MediaPipe"""
    print("\n🎭 Test de MediaPipe...")
    
    try:
        import mediapipe as mp
        
        # Test d'initialisation de la détection de visages
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.5
        )
        
        # Test avec une image factice
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_img)
        
        print("   ✅ MediaPipe - OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur MediaPipe: {e}")
        return False

def test_tensorflow():
    """Teste TensorFlow"""
    print("\n🧠 Test de TensorFlow...")
    
    try:
        import tensorflow as tf
        
        # Supprimer les logs verbeux
        tf.get_logger().setLevel('ERROR')
        
        # Test simple
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.constant([[1.0], [1.0]])
        result = tf.matmul(x, y)
        
        print(f"   ✅ TensorFlow {tf.__version__} - OK")
        
        # Vérifier GPU
        if tf.config.list_physical_devices('GPU'):
            print("   🚀 GPU détecté - Accélération disponible")
        else:
            print("   💻 CPU uniquement - Pas de GPU détecté")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur TensorFlow: {e}")
        return False

def test_deepface():
    """Teste DeepFace"""
    print("\n🔍 Test de DeepFace...")
    
    try:
        from deepface import DeepFace
        
        # Créer une image de test simple
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test d'analyse (peut échouer sur l'image aléatoire, c'est normal)
        try:
            result = DeepFace.analyze(
                test_img, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            print("   ✅ DeepFace - OK (analyse réussie)")
        except:
            print("   ✅ DeepFace - OK (module chargé)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur DeepFace: {e}")
        return False

def test_streamlit():
    """Teste Streamlit"""
    print("\n🌐 Test de Streamlit...")
    
    try:
        import streamlit as st
        print(f"   ✅ Streamlit {st.__version__} - OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur Streamlit: {e}")
        return False

def test_system_info():
    """Affiche les informations système"""
    print("\n💻 Informations système:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Processeur: {platform.processor()}")
    
    # Mémoire disponible (si psutil est disponible)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   RAM: {memory.total // (1024**3)} GB (disponible: {memory.available // (1024**3)} GB)")
    except ImportError:
        print("   RAM: Information non disponible (psutil non installé)")

def test_file_structure():
    """Vérifie la structure des fichiers"""
    print("\n📁 Test de la structure des fichiers...")
    
    required_files = [
        "main_app.py",
        "app_mode1.py", 
        "app_mode2.py",
        "face_analyzer.py",
        "requirements.txt",
        "config.py",
        "README.md"
    ]
    
    missing_files = []
    
    for file in required_files:
        try:
            with open(file, 'r'):
                print(f"   ✅ {file} - Présent")
        except FileNotFoundError:
            print(f"   ❌ {file} - Manquant")
            missing_files.append(file)
    
    return len(missing_files) == 0

def run_all_tests():
    """Exécute tous les tests"""
    print("🎭" + "="*60)
    print("    TEST D'INSTALLATION - ANALYSEUR DE VISAGES IA")
    print("="*63)
    
    tests = [
        ("Version Python", test_python_version),
        ("Dépendances", test_dependencies),
        ("OpenCV", test_opencv_functionality),
        ("MediaPipe", test_mediapipe),
        ("TensorFlow", test_tensorflow),
        ("DeepFace", test_deepface),
        ("Streamlit", test_streamlit),
        ("Structure des fichiers", test_file_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Erreur inattendue dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Afficher les informations système
    test_system_info()
    
    # Résumé
    print("\n" + "="*63)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*63)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        print(f"   {test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Installation complète et fonctionnelle!")
        print("\n🚀 Vous pouvez maintenant lancer l'application avec:")
        print("   python run.py")
        print("   ou")
        print("   streamlit run main_app.py")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez l'installation.")
        print("\n📋 Pour installer les dépendances manquantes:")
        print("   pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 