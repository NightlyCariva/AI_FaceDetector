import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
from face_analyzer import FaceAnalyzer
import threading
import time
from collections import deque

# Initialisation de l'historique global dans la session Streamlit
if 'detection_history' not in st.session_state:
    st.session_state['detection_history'] = deque(maxlen=100)

import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Visages - Mode Temps Réel",
    page_icon="📹",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stats-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration RTC pour WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialisation défensive
        if 'detection_history' not in st.session_state:
            st.session_state['detection_history'] = deque(maxlen=100)
        self.analyzer = None
        self.frame_count = 0
        # Historique local au transformer
        self.detection_history = deque(maxlen=100)
        self.fps_counter = deque(maxlen=30)  # Pour calculer les FPS
        self.last_time = time.time()
        # Options d'affichage
        self.show_age = True
        self.show_gender = True
        self.show_emotion = True
        self.show_race = True
        self.confidence_threshold = 0.5
        self.initialize_analyzer()
    
    def initialize_analyzer(self):
        """Initialise l'analyseur de visages"""
        try:
            self.analyzer = FaceAnalyzer()
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation: {e}")
            self.analyzer = None
    
    def recv(self, frame):
        """Traite chaque frame de la webcam"""
        img = frame.to_ndarray(format="bgr24")
        
        if self.analyzer is None:
            cv2.putText(img, "Initialisation...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        try:
            # Calculer les FPS
            current_time = time.time()
            frame_time = current_time - self.last_time
            self.fps_counter.append(frame_time)
            self.last_time = current_time
            
            # Traiter la frame avec les options d'affichage
            annotated_frame, faces_data = self.analyzer.process_frame(
                img, 
                show_age=self.show_age,
                show_gender=self.show_gender, 
                show_emotion=self.show_emotion,
                show_race=self.show_race,
                confidence_threshold=self.confidence_threshold
            )
            
            # Sauvegarder TOUTES les détections (même 0 visage pour voir que ça fonctionne)
            faces_count = len(faces_data) if faces_data else 0
            detection_data = {
                'timestamp': current_time,
                'frame': self.frame_count,
                'faces_count': faces_count,
                'faces': faces_data if faces_data else []
            }
            # Ajoute la détection dans l'historique local
            self.detection_history.append(detection_data)
            
            # Ajouter les informations FPS sur l'image
            if len(self.fps_counter) > 1:
                avg_frame_time = sum(self.fps_counter) / len(self.fps_counter)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Ajouter le nombre de visages détectés
            cv2.putText(annotated_frame, f"Visages: {faces_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            self.frame_count += 1
            
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            
        except Exception as e:
            # En cas d'erreur, retourner la frame originale avec l'erreur
            error_msg = f"Erreur: {str(e)[:30]}"
            cv2.putText(img, error_msg, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

from streamlit_autorefresh import st_autorefresh

def main():
    # Rafraîchit la page automatiquement toutes les 1000ms (1s)
    st_autorefresh(interval=1000, key="auto-refresh")

    # Titre principal
    st.markdown('<h1 class="main-header">📹 Analyseur de Visages - Temps Réel</h1>', unsafe_allow_html=True)
    
    # Initialiser les variables de session pour l'auto-refresh
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()

    
    # Sidebar avec contrôles
    with st.sidebar:
        st.header("🎛️ Contrôles")
        
        # Paramètres de détection
        st.subheader("Paramètres de Détection")
        confidence_threshold = st.slider(
            "Seuil de confiance", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.1
        )
        
        # Options d'affichage
        st.subheader("Options d'Affichage")
        show_age = st.checkbox("Afficher l'âge", value=True)
        show_gender = st.checkbox("Afficher le genre", value=True)
        show_emotion = st.checkbox("Afficher l'émotion", value=True)
        show_race = st.checkbox("Afficher l'ethnie", value=True)
        

    
    # Interface principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📷 Flux Vidéo en Temps Réel")
        
        # Message d'aide important
        st.info("🎯 **IMPORTANT:** Cliquez sur 'START' pour activer la webcam et commencer l'analyse !")
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="face-analysis",
            video_transformer_factory=VideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Mettre à jour les options du transformer si il existe
        if webrtc_ctx.video_transformer:
            webrtc_ctx.video_transformer.show_age = show_age
            webrtc_ctx.video_transformer.show_gender = show_gender
            webrtc_ctx.video_transformer.show_emotion = show_emotion
            webrtc_ctx.video_transformer.show_race = show_race
            webrtc_ctx.video_transformer.confidence_threshold = confidence_threshold
    
    with col2:
        st.header("📈 Statistiques en Temps Réel")
        
        # Conteneur pour les statistiques qui se mettent à jour
        stats_container = st.empty()
        
        # Conteneur pour l'historique des détections
        history_container = st.empty()
        
        # Bouton pour effacer l'historique
        if st.button("🗑️ Effacer l'Historique"):
            if webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.detection_history.clear()
                webrtc_ctx.video_transformer.fps_counter.clear()
                webrtc_ctx.video_transformer.frame_count = 0
                st.success("Historique effacé!")
                st.rerun()
        
        # Supprimer le checkbox auto-refresh inutile
        st.info("🔄 Actualisation automatique activée")
    
    # FORCER la mise à jour TOUJOURS, peu importe l'état
    if webrtc_ctx.video_transformer:
        transformer = webrtc_ctx.video_transformer
        
        # Debug état WebRTC
        state_info = f"État: {webrtc_ctx.state.playing if hasattr(webrtc_ctx.state, 'playing') else 'inconnu'}"
        detection_history = transformer.detection_history
        history_len = len(detection_history)
        frame_count = transformer.frame_count if hasattr(transformer, 'frame_count') else 0
        
        # Statistiques dans le conteneur principal
        with stats_container.container():
            current_time = time.strftime("%H:%M:%S")
            st.markdown(f"**📊 Stats (MAJ: {current_time})**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Frames traitées", frame_count)
            with col2:
                st.metric("Historique", f"{history_len}/100")
        
        # Historique dans le conteneur historique  
        with history_container.container():
            st.markdown(f"### 📊 Historique des Détections ({history_len}/100 entrées)")
            # Affiche le nombre total de visages détectés dans l'historique
            total_faces = sum(det['faces_count'] for det in detection_history)
            zero_face = sum(1 for det in detection_history if det['faces_count'] == 0)
            one_face = sum(1 for det in detection_history if det['faces_count'] == 1)
            two_plus = sum(1 for det in detection_history if det['faces_count'] >= 2)
            st.markdown(f"**Total visages détectés sur {history_len} entrées : {total_faces}**")
            st.markdown(f"0 visage(s) : {zero_face} | 1 visage : {one_face} | 2+ visages : {two_plus}")
            if history_len > 0:
                # Affiche toujours les 5 dernières entrées, même si faces_count == 0
                recent = list(detection_history)[-5:]
                for i, det in enumerate(recent):
                    timestamp = time.strftime("%H:%M:%S", time.localtime(det['timestamp']))
                    count = det['faces_count']
                    label = "visage" if count == 1 else "visages"
                    st.text(f"{timestamp}: Frame {det['frame']} - {count} {label}")
            else:
                st.info("🟡 En attente de détections... Assurez-vous que la webcam est activée!")
        
    # REFRESH FORCÉ PERMANENT - PAS DE CONDITIONS
    current_time = time.time()
    if current_time - st.session_state.last_refresh > 0.5:  # Toutes les 500ms
        st.session_state.last_refresh = current_time
        st.rerun()
    
    # Section d'aide
    st.markdown("---")
    st.header("📱 Utilisation avec Téléphone")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Option 1: DroidCam
        1. Installez DroidCam sur votre téléphone
        2. Installez le client DroidCam sur votre PC
        3. Connectez via WiFi ou USB
        4. Sélectionnez DroidCam comme source vidéo
        """)
    
    with col2:
        st.markdown("""
        ### Option 2: IP Webcam
        1. Installez IP Webcam sur Android
        2. Démarrez le serveur dans l'app
        3. Notez l'adresse IP affichée
        4. Utilisez l'URL dans votre navigateur
        """)
    


def update_real_time_stats(transformer, stats_container, history_container):
    """Met à jour les statistiques en temps réel"""
    
    with stats_container.container():
        if hasattr(transformer, 'detection_history') and len(transformer.detection_history) > 0:
            # Statistiques générales
            total_detections = len(transformer.detection_history)
            
            # Calculer le nombre total de visages détectés (vraiment détectés)
            total_faces = 0
            for detection in transformer.detection_history:
                if detection['faces_count'] > 0:
                    total_faces += detection['faces_count']
            
            # Calculer les FPS moyens
            if hasattr(transformer, 'fps_counter') and len(transformer.fps_counter) > 1:
                avg_frame_time = sum(transformer.fps_counter) / len(transformer.fps_counter)
                avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            else:
                avg_fps = 0
            
            # Afficher les métriques avec un timestamp pour montrer l'actualisation
            current_time = time.strftime("%H:%M:%S")
            st.markdown(f"**📊 Stats (MAJ: {current_time})**")
            
            # Debug info sur l'historique
            st.markdown(f"*Historique: {total_detections}/100 entrées*")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Frames traitées", total_detections)
            
            with col2:
                st.metric("Visages détectés", total_faces)
            
            with col3:
                st.metric("FPS moyen", f"{avg_fps:.1f}")
            
            # Dernières détections (seulement celles avec des visages)
            detections_with_faces = [d for d in transformer.detection_history if d['faces_count'] > 0]
            
            if detections_with_faces:
                latest_detection = detections_with_faces[-1]
                
                st.markdown("### 🔍 Dernière Détection")
                
                for i, face in enumerate(latest_detection['faces']):
                    if face.get('attributes'):
                        details = []
                        attrs = face['attributes']
                        
                        # Vérifier les options d'affichage du transformer
                        if hasattr(transformer, 'show_age') and transformer.show_age:
                            details.append(f"🎂 Age: {attrs['age']}")
                        if hasattr(transformer, 'show_gender') and transformer.show_gender:
                            details.append(f"⚧️ Genre: {attrs['gender']}")
                        if hasattr(transformer, 'show_race') and transformer.show_race:
                            details.append(f"🌍 Ethnie: {attrs['race']}")
                        if hasattr(transformer, 'show_emotion') and transformer.show_emotion:
                            details.append(f"😊 Emotion: {attrs['emotion']}")
                        
                        if details:
                            st.markdown(f"**Visage {i+1}:**\n" + "\n".join([f"- {detail}" for detail in details]))
                        else:
                            st.markdown(f"**Visage {i+1}:** Détecté (options d'affichage désactivées)")
            else:
                st.info("Aucun visage détecté récemment")
        
        else:
            st.info("En attente de détections...")
    
    # Historique des détections - FORCÉ À SE METTRE À JOUR
    with history_container.container():
        if hasattr(transformer, 'detection_history'):
            history_length = len(transformer.detection_history)
            st.markdown(f"### 📊 Historique des Détections ({history_length}/100 entrées)")
            
            if history_length > 0:
                # Créer un DataFrame avec l'historique (prendre les 20 dernières)
                recent_detections = list(transformer.detection_history)[-20:]
                history_data = []
                
                for detection in recent_detections:
                    timestamp = time.strftime("%H:%M:%S", time.localtime(detection['timestamp']))
                    history_data.append({
                        'Heure': timestamp,
                        'Frame': detection['frame'],
                        'Visages': detection['faces_count']
                    })
                
                if history_data:
                    df = pd.DataFrame(history_data)
                    st.dataframe(df, use_container_width=True, height=200)
                    
                    # Graphique des détections dans le temps
                    if len(history_data) > 1:
                        st.line_chart(df.set_index('Heure')['Visages'])
                else:
                    st.info("Aucune donnée d'historique valide")
            else:
                st.info("Historique vide")
        else:
            st.info("Historique non initialisé")

if __name__ == "__main__":
    main() 