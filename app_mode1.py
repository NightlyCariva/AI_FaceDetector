import streamlit as st
import cv2
import tempfile
import os
import json
from datetime import datetime
import pandas as pd
from face_analyzer import FaceAnalyzer
import time

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Visages - Mode Upload",
    page_icon="🎭",
    layout="wide"
)

# CSS personnalisé pour améliorer l'apparence
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Titre principal
    st.markdown('<h1 class="main-header">🎭 Analyseur de Visages</h1>', unsafe_allow_html=True)
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("📋 Informations")
        st.markdown("""
        **Formats supportés :**
        - MP4, AVI, MOV, MKV
        - Résolution max : 1920x1080
        """)
        
        st.header("🔧 Paramètres")
        confidence_threshold = st.slider(
            "Seuil de confiance", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.1
        )
    
    # Interface principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload de Vidéo")
        
        uploaded_file = st.file_uploader(
            "Choisissez une vidéo",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Formats supportés: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None:
            # Afficher les informations du fichier
            file_details = {
                "Nom": uploaded_file.name,
                "Taille": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "Type": uploaded_file.type
            }
            
            st.markdown("**Détails du fichier :**")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            
            # Bouton de traitement
            if st.button("🚀 Analyser la Vidéo", type="primary"):
                process_video(uploaded_file, confidence_threshold)
    
    with col2:
        st.header("📊 Résultats")
        
        # Zone pour afficher les résultats
        if 'analysis_results' not in st.session_state:
            st.markdown("""
            <div class="info-box">
                <p>📹 Uploadez une vidéo et cliquez sur "Analyser" pour voir les résultats ici.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            display_results()

def process_video(uploaded_file, confidence_threshold):
    """Traite la vidéo uploadée"""
    
    # Créer un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name
    
    # Créer le fichier de sortie
    output_path = tempfile.mktemp(suffix='_analyzed.mp4')
    
    try:
        # Initialiser l'analyseur
        with st.spinner("🔄 Initialisation de l'analyseur..."):
            analyzer = FaceAnalyzer()
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress / 100)
            status_text.text(f"Traitement en cours... {progress:.1f}%")
        
        # Traiter la vidéo
        start_time = time.time()
        
        with st.spinner("🎬 Analyse de la vidéo en cours..."):
            detections = analyzer.process_video(
                input_path, 
                output_path, 
                progress_callback=update_progress
            )
        
        processing_time = time.time() - start_time
        
        # Sauvegarder les résultats dans la session
        st.session_state.analysis_results = {
            'detections': detections,
            'output_path': output_path,
            'processing_time': processing_time,
            'total_frames': len(detections),
            'filename': uploaded_file.name
        }
        
        # Afficher le succès
        st.success(f"✅ Analyse terminée en {processing_time:.2f} secondes!")
        
        # Nettoyer le fichier temporaire d'entrée
        os.unlink(input_path)
        
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement: {str(e)}")
        # Nettoyer les fichiers temporaires en cas d'erreur
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

def display_results():
    """Affiche les résultats de l'analyse"""
    results = st.session_state.analysis_results
    
    # Statistiques générales
    st.markdown("### 📈 Statistiques Générales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Frames analysées", results['total_frames'])
    
    with col2:
        total_faces = sum(len(frame['faces']) for frame in results['detections'])
        st.metric("Visages détectés", total_faces)
    
    with col3:
        st.metric("Temps de traitement", f"{results['processing_time']:.2f}s")
    
    with col4:
        fps = results['total_frames'] / results['processing_time'] if results['processing_time'] > 0 else 0
        st.metric("FPS de traitement", f"{fps:.1f}")
    
    # Téléchargement de la vidéo analysée
    st.markdown("### 📥 Téléchargement")
    
    if os.path.exists(results['output_path']):
        with open(results['output_path'], 'rb') as file:
            st.download_button(
                label="📹 Télécharger la vidéo analysée",
                data=file.read(),
                file_name=f"analyzed_{results['filename']}",
                mime="video/mp4"
            )
    
    # Analyse détaillée
    st.markdown("### 🔍 Analyse Détaillée")
    
    # Créer un DataFrame avec toutes les détections
    all_faces_data = []
    for frame_data in results['detections']:
        for face in frame_data['faces']:
            if face['attributes']:
                all_faces_data.append({
                    'Frame': frame_data['frame'],
                    'Timestamp': f"{frame_data['timestamp']:.2f}s",
                    'Âge': face['attributes']['age'],
                    'Genre': face['attributes']['gender'],
                    'Ethnie': face['attributes']['race'],
                    'Émotion': face['attributes']['emotion'],
                    'Confiance': f"{face['confidence']:.2f}"
                })
    
    if all_faces_data:
        df = pd.DataFrame(all_faces_data)
        
        # Afficher le tableau
        st.dataframe(df, use_container_width=True)
        
        # Graphiques de distribution
        st.markdown("### 📊 Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Genre' in df.columns:
                gender_counts = df['Genre'].value_counts()
                st.bar_chart(gender_counts)
                st.caption("Distribution par Genre")
        
        with col2:
            if 'Émotion' in df.columns:
                emotion_counts = df['Émotion'].value_counts()
                st.bar_chart(emotion_counts)
                st.caption("Distribution par Émotion")
        
        # Téléchargement des données
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Télécharger les données CSV",
            data=csv,
            file_name=f"analysis_{results['filename']}.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("⚠️ Aucun visage détecté dans la vidéo.")
    


if __name__ == "__main__":
    main() 