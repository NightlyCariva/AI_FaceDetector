import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
from face_analyzer import FaceAnalyzer
import threading
import time
from collections import deque
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
        self.analyzer = None
        self.frame_count = 0
        self.detection_history = deque(maxlen=100)  # Garder les 100 dernières détections
        self.fps_counter = deque(maxlen=30)  # Pour calculer les FPS
        self.last_time = time.time()
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
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        try:
            # Calculer les FPS
            current_time = time.time()
            self.fps_counter.append(current_time - self.last_time)
            self.last_time = current_time
            
            # Traiter la frame
            annotated_frame, faces_data = self.analyzer.process_frame(img)
            
            # Sauvegarder les détections
            if faces_data:
                detection_data = {
                    'timestamp': current_time,
                    'frame': self.frame_count,
                    'faces_count': len(faces_data),
                    'faces': faces_data
                }
                self.detection_history.append(detection_data)
            
            # Ajouter les informations FPS sur l'image
            if len(self.fps_counter) > 1:
                avg_frame_time = sum(self.fps_counter) / len(self.fps_counter)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Ajouter le nombre de visages détectés
            cv2.putText(annotated_frame, f"Visages: {len(faces_data)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            self.frame_count += 1
            
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            
        except Exception as e:
            # En cas d'erreur, retourner la frame originale
            cv2.putText(img, f"Erreur: {str(e)[:50]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Titre principal
    st.markdown('<h1 class="main-header">📹 Analyseur de Visages - Temps Réel</h1>', unsafe_allow_html=True)
    
    # Avertissement sur la protection des données
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Avertissement - Protection des Données</h3>
        <p>Cette application analyse en temps réel les caractéristiques biométriques des visages. 
        Les données traitées sont sensibles selon le RGPD. Assurez-vous d'avoir les autorisations 
        nécessaires avant d'utiliser cette fonctionnalité.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        # Informations sur la performance
        st.subheader("📊 Performance")
        st.info("""
        **Optimisations appliquées :**
        - Traitement adaptatif selon les performances
        - Cache des modèles ML
        - Réduction de résolution si nécessaire
        """)
        
        # Considérations légales
        st.subheader("⚖️ Aspects Légaux")
        st.warning("""
        **Points d'attention RGPD :**
        - Consentement explicite requis
        - Données biométriques = catégorie spéciale
        - Droit à l'effacement applicable
        - Minimisation des données
        """)
    
    # Interface principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📷 Flux Vidéo en Temps Réel")
        
        # Instructions
        st.markdown("""
        <div class="info-box">
            <p>🎥 Cliquez sur "START" pour commencer l'analyse en temps réel de votre webcam.</p>
            <p>📱 Pour utiliser votre téléphone comme caméra, vous pouvez utiliser des applications comme DroidCam ou IP Webcam.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="face-analysis",
            video_transformer_factory=VideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
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
                st.success("Historique effacé!")
    
    # Mise à jour des statistiques en temps réel
    if webrtc_ctx.video_transformer:
        update_real_time_stats(webrtc_ctx.video_transformer, stats_container, history_container)
    
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
    
    # Section de protection des données
    st.markdown("---")
    st.header("🔒 Protection des Données - Mode Temps Réel")
    
    st.markdown("""
    <div class="warning-box">
        <h4>Spécificités du traitement en temps réel :</h4>
        <ul>
            <li>🔄 <strong>Traitement continu</strong> - Les données sont analysées en permanence</li>
            <li>💾 <strong>Stockage temporaire</strong> - Historique limité aux 100 dernières détections</li>
            <li>🚫 <strong>Pas d'enregistrement</strong> - Aucune vidéo n'est sauvegardée par défaut</li>
            <li>⚡ <strong>Traitement local</strong> - Toutes les analyses se font sur votre machine</li>
        </ul>
        
        <h4>Mesures de protection :</h4>
        <ul>
            <li>✅ Effacement automatique de l'historique</li>
            <li>✅ Aucune transmission de données vers des serveurs externes</li>
            <li>✅ Contrôle total sur l'activation/désactivation</li>
            <li>✅ Transparence sur les données collectées</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def update_real_time_stats(transformer, stats_container, history_container):
    """Met à jour les statistiques en temps réel"""
    
    with stats_container.container():
        if hasattr(transformer, 'detection_history') and transformer.detection_history:
            # Statistiques générales
            total_detections = len(transformer.detection_history)
            total_faces = sum(d['faces_count'] for d in transformer.detection_history)
            
            # Calculer les FPS moyens
            if hasattr(transformer, 'fps_counter') and len(transformer.fps_counter) > 1:
                avg_frame_time = sum(transformer.fps_counter) / len(transformer.fps_counter)
                avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            else:
                avg_fps = 0
            
            # Afficher les métriques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Frames traitées", total_detections)
            
            with col2:
                st.metric("Visages détectés", total_faces)
            
            with col3:
                st.metric("FPS moyen", f"{avg_fps:.1f}")
            
            # Dernières détections
            if transformer.detection_history:
                latest_detection = transformer.detection_history[-1]
                
                st.markdown("### 🔍 Dernière Détection")
                
                if latest_detection['faces']:
                    for i, face in enumerate(latest_detection['faces']):
                        if face['attributes']:
                            st.markdown(f"""
                            **Visage {i+1}:**
                            - 🎂 Âge: {face['attributes']['age']}
                            - ⚧️ Genre: {face['attributes']['gender']}
                            - 🌍 Ethnie: {face['attributes']['race']}
                            - 😊 Émotion: {face['attributes']['emotion']}
                            """)
                else:
                    st.info("Aucun visage détecté dans la dernière frame")
        
        else:
            st.info("En attente de détections...")
    
    # Historique des détections
    with history_container.container():
        if hasattr(transformer, 'detection_history') and transformer.detection_history:
            st.markdown("### 📊 Historique des Détections")
            
            # Créer un DataFrame avec l'historique
            history_data = []
            for detection in list(transformer.detection_history)[-20:]:  # 20 dernières
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

if __name__ == "__main__":
    main() 