import streamlit as st
import cv2
import numpy as np
import threading
import time
from datetime import datetime
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from face_detector import FaceDetector
from PIL import Image
import requests
import urllib.request
import socket

class DroidCamCapture:
    """Capture DroidCam non-bloquante (bugged)"""
    
    def __init__(self, url):
        self.url = url.strip()
        self.is_opened = False
        self.cached_frame = None
        
    def open(self):
        """Test rapide de connexion"""
        try:
            response = requests.get(self.url, timeout=3)
            if response.status_code == 200 and len(response.content) > 1000:
                self.is_opened = True
                nparr = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.cached_frame = frame
                return True
            return False
        except:
            return False
    
    def isOpened(self):
        return self.is_opened
    
    def read(self):
        """Retourne la frame cachée sans faire de requête bloquante"""
        if self.cached_frame is not None:
            return True, self.cached_frame.copy()
        return False, None
    
    def refresh_frame(self):
        """Met à jour la frame cachée (à appeler manuellement)"""
        if not self.is_opened:
            return False
        try:
            response = requests.get(self.url, timeout=2)
            if response.status_code == 200:
                nparr = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.cached_frame = frame
                    return True
        except:
            pass
        return False
    
    def set(self, prop, value):
        pass
    
    def release(self):
        self.is_opened = False




def run_mode2():
    """Interface du Mode 2: Temps Réel"""
    
    st.title("Mode 2: Analyse en Temps Réel")
    st.markdown("---")
    
    st.header("Fonctionnement")
    st.info("""
    Ce mode analyse en temps réel via votre webcam ou DroidCam. Configurez les paramètres d'analyse, 
    puis cliquez sur "Démarrer Caméra" pour commencer la détection. Les résultats s'affichent 
    en direct et peuvent être exportés en CSV.
    
    **Support caméras**: Webcam intégrée, USB, DroidCam via URL
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Paramètres")
        
        temperature = st.slider("Température", 0.0, 1.0, 0.5, 0.1, key="rt_temp")
        
        st.subheader("Options d'analyse")
        analyze_age = st.checkbox("Age", value=True, key="rt_age")
        analyze_gender = st.checkbox("Genre", value=True, key="rt_gender")
        analyze_emotion = st.checkbox("Emotion", value=True, key="rt_emotion")
        analyze_ethnicity = st.checkbox("Ethnie", value=True, key="rt_ethnicity")
        
        use_gpu = st.checkbox("Accélération matériel (GPU)", value=False, key="rt_gpu")
        
        st.subheader("Paramètres de tracking")
        detection_interval = st.slider(
            "Intervalle de détection (frames)", 
            min_value=10, 
            max_value=60, 
            value=30, 
            step=5,
            key="rt_interval",
            help="Fréquence de recherche de nouveaux visages"
        )
        
        st.header("Configuration Caméra")
        
        camera_source = st.radio(
            "Source vidéo:",
            ["Webcam locale", "DroidCam (URL)", "Webcam USB"],
            key="camera_source"
        )
        
        droidcam_url = ""
        camera_id = 0
        
        if camera_source == "DroidCam (URL)":
            droidcam_url = st.text_input(
                "URL DroidCam",
                placeholder="http://192.168.1.168:4747/video",
                help="URL du flux vidéo DroidCam (format: http://IP:PORT/video)"
            )
            if droidcam_url:
                st.info(f"URL configurée: {droidcam_url}")
        
        elif camera_source == "Webcam USB":
            camera_id = st.number_input(
                "ID Caméra USB", 
                min_value=0, 
                max_value=10, 
                value=1,
                help="0=webcam par défaut, 1+=caméras USB"
            )
        
        st.header("Contrôles")
        
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
        
        if 'realtime_detections' not in st.session_state:
            st.session_state.realtime_detections = []
        
        if 'video_capture' not in st.session_state:
            st.session_state.video_capture = None
        
        col_start, col_stop, col_clear = st.columns(3)
        
        with col_start:
            if st.button("🎥 Démarrer Caméra", type="primary"):
                start_camera(camera_source, camera_id, droidcam_url, use_gpu, detection_interval)
        
        with col_stop:
            if st.button("⏹️ Arrêter Caméra"):
                stop_camera()
        
        with col_clear:
            if st.button("🗑️ Effacer Données"):
                st.session_state.realtime_detections = []
                st.success("Données effacées")
                st.rerun()
    
    with col2:
        st.header("Flux Vidéo en Temps Réel")
        
        video_placeholder = st.empty()
        
        if camera_source == "DroidCam (URL)":
            st.info("""
            **Instructions DroidCam:**
            1. Installez DroidCam sur votre téléphone
            2. Démarrez DroidCam et notez l'URL affichée
            3. Saisissez l'URL ci-dessus
            4. Cliquez sur "Démarrer Caméra"
            
            **URLs à essayer:**
            - `http://IP:4747/video` (flux MJPEG)
            - `http://IP:4747/mjpegfeed` (flux alternatif)
            
            **Important:**
            - Vérifiez que le téléphone et PC sont sur le même réseau WiFi
            - Testez l'URL dans votre navigateur d'abord
            """)
            

        
        # Traitement de la caméra si active
        if st.session_state.camera_running and st.session_state.video_capture:
            col_refresh, col_auto, col_detect = st.columns(3)
            
            with col_refresh:
                if st.button("Actualiser Image"):
                    # Actualiser la frame DroidCam
                    if hasattr(st.session_state.video_capture, 'refresh_frame'):
                        if st.session_state.video_capture.refresh_frame():
                            st.success("Image actualisée")
                        else:
                            st.error("Échec actualisation")
            
            with col_auto:
                auto_refresh = st.checkbox("Auto-actualisation", value=True)
            
            with col_detect:
                if st.button("🔍 Forcer Détection"):
                    # Forcer une nouvelle détection en réinitialisant le compteur
                    st.session_state.frame_count = 0
                    st.success("Détection forcée")
            
            # Traiter la frame avec détection des visages
            process_camera_frame(
                video_placeholder, 
                analyze_age, 
                analyze_gender, 
                analyze_emotion, 
                analyze_ethnicity, 
                detection_interval
            )
            
            # Auto-actualisation non-bloquante
            if auto_refresh:
                time.sleep(0.5)  # Réduire le délai pour plus de fluidité
                st.rerun()
        else:
            video_placeholder.info("Cliquez sur 'Démarrer Caméra' pour commencer")
        
        # Statistiques en temps réel
        display_realtime_stats()
        
        # Export des données
        if st.session_state.realtime_detections:
            export_realtime_data()
    
    # Zone Console (pleine largeur)
    st.markdown("---")
    st.header("Console")
    
    if 'console_output_rt' in st.session_state:
        st.text_area(
            "Logs d'exécution",
            value=st.session_state.console_output_rt,
            height=150,
            disabled=True,
            key="console_rt"
        )
    else:
        st.text_area(
            "Logs d'exécution",
            value="En attente de démarrage de la caméra...",
            height=150,
            disabled=True,
            key="console_rt_empty"
        )
    


def start_camera(camera_source, camera_id, droidcam_url, use_gpu, detection_interval):
    """Démarre la capture caméra"""
    
    console_output = f"[{datetime.now().strftime('%H:%M:%S')}] Démarrage de la caméra\n"
    console_output += f"Source: {camera_source}\n"
    
    try:
        if camera_source == "DroidCam (URL)" and droidcam_url:
            video_source = droidcam_url
            console_output += f"URL DroidCam: {droidcam_url}\n"
        else:
            video_source = int(camera_id)
            console_output += f"ID Caméra: {camera_id}\n"
        
        if camera_source == "DroidCam (URL)":
            if not isinstance(video_source, str) or not video_source.strip():
                raise Exception("URL DroidCam invalide ou vide")
            
            clean_url = video_source.replace('htpp://', 'http://').strip()
            console_output += f"Connexion DroidCam: {clean_url}\n"
            
            console_output += "Test préliminaire de l'URL...\n"
            try:
                test_response = requests.get(clean_url, timeout=10)
                console_output += f"Test direct: Code {test_response.status_code}\n"
                console_output += f"Content-Type: {test_response.headers.get('content-type', 'unknown')}\n"
                console_output += f"Taille réponse: {len(test_response.content)} bytes\n"
                
                if test_response.status_code != 200:
                    raise Exception(f"URL ne répond pas correctement (Code: {test_response.status_code})")
                    
            except requests.exceptions.ConnectTimeout:
                raise Exception(f"Timeout de connexion - Vérifiez l'IP et le port")
            except requests.exceptions.ConnectionError:
                raise Exception(f"Impossible de se connecter - Vérifiez que DroidCam est démarré")
            except Exception as e:
                raise Exception(f"Erreur de test: {str(e)}")
            
            cap = DroidCamCapture(clean_url)
            if not cap.open():
                raise Exception(f"Test OK mais capture échoue pour {clean_url}")
                
            console_output += "DroidCam connecté avec succès\n"
        else:
            cap = cv2.VideoCapture(video_source)
            
            if not cap.isOpened():
                raise Exception(f"Impossible d'ouvrir la source vidéo: {video_source}")
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        detector = FaceDetector(use_gpu=use_gpu)
        detector.detection_interval = detection_interval
        
        console_output += "Test du détecteur de visages...\n"
        if detector.face_cascade.empty():
            console_output += "ATTENTION: Classificateur de visages non chargé!\n"
        else:
            console_output += "Classificateur de visages chargé avec succès\n"
        
        st.session_state.video_capture = cap
        st.session_state.face_detector = detector
        st.session_state.camera_running = True
        st.session_state.frame_count = 0
        
        console_output += "Caméra initialisée avec succès\n"
        console_output += f"Résolution: 640x480\n"
        console_output += f"Détecteur initialisé (GPU: {use_gpu})\n"
        console_output += f"Intervalle détection: {detection_interval} frames\n"
        console_output += f"Mode temps réel: détection forcée toutes les 5 frames max\n"
        
        st.success("Caméra démarrée avec succès!")
        
    except Exception as e:
        console_output += f"ERREUR: {str(e)}\n"
        st.error(f"Erreur lors du démarrage: {str(e)}")
        st.session_state.camera_running = False
    
    st.session_state.console_output_rt = console_output

def stop_camera():
    """Arrête la capture caméra"""
    
    console_output = st.session_state.get('console_output_rt', '')
    console_output += f"\n[{datetime.now().strftime('%H:%M:%S')}] Arrêt de la caméra\n"
    
    try:
        if st.session_state.video_capture:
            st.session_state.video_capture.release()
            st.session_state.video_capture = None
        
        st.session_state.camera_running = False
        
        total_detections = len(st.session_state.realtime_detections)
        console_output += f"Total détections: {total_detections}\n"
        console_output += "Caméra arrêtée\n"
        
        st.success("⏹️ Caméra arrêtée")
        
    except Exception as e:
        console_output += f"ERREUR lors de l'arrêt: {str(e)}\n"
        st.error(f"❌ Erreur lors de l'arrêt: {str(e)}")
    
    st.session_state.console_output_rt = console_output

def process_camera_frame(placeholder, analyze_age, analyze_gender, analyze_emotion, analyze_ethnicity, detection_interval):
    """Traite une frame de la caméra"""
    
    if not st.session_state.video_capture or not st.session_state.camera_running:
        return
    
    try:
        if hasattr(st.session_state.video_capture, 'refresh_frame'):
            st.session_state.video_capture.refresh_frame()
        
        ret, frame = st.session_state.video_capture.read()
        
        if not ret or frame is None:
            placeholder.error("Pas de frame disponible")
            return
        
        if hasattr(st.session_state, 'face_detector'):
            detector = st.session_state.face_detector
            timestamp = datetime.now().strftime("%H:%M:%S")
            frame_count = st.session_state.get('frame_count', 0)
            
            original_interval = detector.detection_interval
            detector.detection_interval = min(5, detection_interval)
            
            detections = detector.process_frame_with_tracking(
                frame, frame_count, timestamp,
                analyze_age, analyze_gender, analyze_emotion, analyze_ethnicity
            )
            
            detector.detection_interval = original_interval
            
            if detections:
                st.session_state.realtime_detections.extend(detections)
                
                if len(st.session_state.realtime_detections) > 100:
                    st.session_state.realtime_detections = st.session_state.realtime_detections[-100:]
            
            annotated_frame = detector.draw_annotations(
                frame, detections,
                analyze_age, analyze_gender, analyze_emotion, analyze_ethnicity
            )
            
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Visages: {len(detections)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Total: {len(st.session_state.realtime_detections)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            detection_status = "DETECTION ACTIVE" if frame_count % detector.detection_interval == 0 else "TRACKING"
            cv2.putText(annotated_frame, detection_status, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            st.session_state.frame_count = frame_count + 1
            
            if detections and 'console_output_rt' in st.session_state:
                st.session_state.console_output_rt += f"[{timestamp}] {len(detections)} visage(s) détecté(s)\n"
            
        else:
            annotated_frame = frame
        
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        placeholder.image(image, caption="Flux caméra en temps réel", width=640)
        
    except Exception as e:
        placeholder.error(f"Erreur traitement frame: {str(e)}")
        if 'console_output_rt' in st.session_state:
            st.session_state.console_output_rt += f"\nErreur frame: {str(e)}\n"

def display_realtime_stats():
    """Affiche les statistiques temps réel"""
    
    st.subheader("Statistiques")
    
    total_detections = len(st.session_state.realtime_detections)
    frame_count = st.session_state.get('frame_count', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Frames", frame_count)
    
    with col2:
        st.metric("Détections", total_detections)
    
    with col3:
        if st.session_state.camera_running:
            st.metric("Statut", "🟢 Actif")
        else:
            st.metric("Statut", "🔴 Arrêté")
    
    with col4:
        if total_detections > 0:
            unique_faces = len(set(d['face_id'] for d in st.session_state.realtime_detections))
            st.metric("Visages uniques", unique_faces)
        else:
            st.metric("Visages uniques", "0")
    
    if st.session_state.realtime_detections:
        st.subheader("Dernières Détections")
        
        recent_detections = st.session_state.realtime_detections[-10:]
        
        if recent_detections:
            import pandas as pd
            df = pd.DataFrame(recent_detections)
            display_df = df.drop('bbox', axis=1, errors='ignore')
            st.dataframe(display_df, use_container_width=True)
            
            if len(st.session_state.realtime_detections) >= 5:
                all_df = pd.DataFrame(st.session_state.realtime_detections)
                
                col1, col2 = st.columns(2)
                
                if 'gender_classification' in all_df.columns:
                    with col1:
                        gender_counts = all_df['gender_classification'].value_counts()
                        st.write("**Répartition Genre:**")
                        for gender, count in gender_counts.items():
                            st.write(f"- {gender}: {count}")
                
                if 'emotion' in all_df.columns:
                    with col2:
                        emotion_counts = all_df['emotion'].value_counts()
                        st.write("**Émotions Dominantes:**")
                        for emotion, count in emotion_counts.head(3).items():
                            st.write(f"- {emotion}: {count}")

def export_realtime_data():
    """Export des données temps réel"""
    
    st.subheader("Export")
    
    if st.session_state.realtime_detections:
        import pandas as pd
        df = pd.DataFrame(st.session_state.realtime_detections)
        
        export_df = df.drop('bbox', axis=1, errors='ignore')
        csv_data = export_df.to_csv(index=False, sep=';')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv_data,
                file_name=f"detections_realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.metric("Entrées à exporter", len(df))
        
        st.write("**Aperçu des données:**")
        st.dataframe(export_df.head(), use_container_width=True)
    else:
        st.info("Aucune donnée à exporter") 