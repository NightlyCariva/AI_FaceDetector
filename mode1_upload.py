import streamlit as st
import cv2
import tempfile
import os
from datetime import datetime
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from face_detector import FaceDetector

def run_mode1():
    """Interface du Mode 1: Upload Vidéo"""
    
    st.title("Mode 1: Analyse de Vidéo Uploadée")
    st.markdown("---")
    
    # Zone d'explication
    st.header("Fonctionnement")
    st.info("""
    Ce mode permet d'analyser des vidéos pré-enregistrées. Uploadez votre fichier vidéo, 
    configurez les paramètres d'analyse, puis lancez le traitement. Vous pourrez ensuite 
    télécharger les résultats sous forme de CSV et la vidéo annotée.
    """)
    
    # Colonnes principales
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Zone de paramètrage
        st.header("Paramètres")
        
        temperature = st.slider("Température", 0.0, 1.0, 0.5, 0.1)
        
        st.subheader("Options d'analyse")
        analyze_age = st.checkbox("Age", value=True)
        analyze_gender = st.checkbox("Genre", value=True)
        analyze_emotion = st.checkbox("Emotion", value=True)
        analyze_ethnicity = st.checkbox("Ethnie", value=True)
        
        use_gpu = st.checkbox("Accélération matériel (GPU)", value=False)
        
        st.subheader("Paramètres de tracking")
        detection_interval = st.slider(
            "Intervalle de détection (frames)", 
            min_value=10, 
            max_value=60, 
            value=30, 
            step=5,
            help="Fréquence de recherche de nouveaux visages (plus élevé = plus rapide)"
        )
        
        # Zone upload vidéo
        st.header("Upload Vidéo")
        uploaded_file = st.file_uploader(
            "Choisissez un fichier vidéo",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Formats supportés: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None:
            file_details = {
                "Nom": uploaded_file.name,
                "Taille": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "Type": uploaded_file.type
            }
            
            st.write("**Détails du fichier:**")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            
            if st.button("Analyser la Vidéo", type="primary"):
                process_video(
                    uploaded_file, temperature, analyze_age, analyze_gender, 
                    analyze_emotion, analyze_ethnicity, use_gpu, detection_interval
                )
    
    with col2:
        # Zone Résultats
        st.header("Résultats")
        
        if 'video_results' in st.session_state:
            # Bouton pour effacer les résultats
            if st.button("🗑️ Effacer les résultats", key="clear_results"):
                del st.session_state.video_results
                if 'console_output' in st.session_state:
                    del st.session_state.console_output
                st.rerun()
            
            display_results()
        else:
            st.info("Uploadez une vidéo et lancez l'analyse pour voir les résultats ici.")
    
    # Zone Console (pleine largeur)
    st.markdown("---")
    st.header("Console")
    
    if 'console_output' in st.session_state:
        st.text_area(
            "Logs d'exécution",
            value=st.session_state.console_output,
            height=200,
            disabled=True
        )
    else:
        st.text_area(
            "Logs d'exécution",
            value="En attente de traitement...",
            height=200,
            disabled=True
        )

def process_video(uploaded_file, temperature, analyze_age, analyze_gender, 
                 analyze_emotion, analyze_ethnicity, use_gpu, detection_interval):
    """Traite la vidéo uploadée"""
    
    # Initialiser la capture de console
    console_output = io.StringIO()
    
    try:
        with redirect_stdout(console_output), redirect_stderr(console_output):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Début du traitement")
            print(f"Fichier: {uploaded_file.name}")
            print(f"Paramètres: Age={analyze_age}, Genre={analyze_gender}, Emotion={analyze_emotion}, Ethnie={analyze_ethnicity}")
            print(f"GPU: {use_gpu}")
            
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                input_path = tmp_file.name
            
            print(f"Fichier temporaire créé: {input_path}")
            
            # Initialiser le détecteur
            detector = FaceDetector(use_gpu=use_gpu)
            detector.detection_interval = detection_interval
            print("Détecteur initialisé")
            print(f"Intervalle de détection configuré: {detection_interval} frames")
            
            # Ouvrir la vidéo
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise Exception("Impossible d'ouvrir la vidéo")
            
            # Propriétés de la vidéo
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Vidéo: {total_frames} frames, {fps} FPS, {width}x{height}")
            
            # Créer le fichier de sortie
            output_path = tempfile.mktemp(suffix='_analyzed.mp4')
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # type: ignore
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            print("Début de l'analyse avec système de tracking...")
            print(f"Détection de nouveaux visages toutes les {detector.detection_interval} frames")
            
            # Barre de progression
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_detections = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculer le timestamp
                timestamp = f"{int(frame_count // fps // 3600):02d}:{int((frame_count // fps) % 3600 // 60):02d}:{int(frame_count // fps % 60):02d}"
                
                # Traiter la frame avec le système de tracking
                detections = detector.process_frame_with_tracking(
                    frame, frame_count, timestamp,
                    analyze_age, analyze_gender, analyze_emotion, analyze_ethnicity
                )
                
                if detections:
                    all_detections.extend(detections)
                    detector.detections.extend(detections)
                    
                    # Afficher les logs seulement quand il y a de nouveaux visages
                    if frame_count % detector.detection_interval == 0:
                        print(f"Frame {frame_count}: {len(detections)} visage(s) tracké(s)")
                
                # Annoter la frame avec tous les visages trackés
                annotated_frame = detector.draw_annotations(
                    frame, detections,
                    analyze_age, analyze_gender, analyze_emotion, analyze_ethnicity
                )
                
                # Écrire la frame
                out.write(annotated_frame)
                
                frame_count += 1
                
                # Mise à jour de la progression
                progress = (frame_count / total_frames)
                progress_bar.progress(progress)
                status_text.text(f"Traitement: {frame_count}/{total_frames} frames")
            
            # Nettoyer
            cap.release()
            out.release()
            
            print(f"Analyse terminée. {len(all_detections)} détections au total")
            
            # Sauvegarder les résultats
            st.session_state.video_results = {
                'detections': all_detections,
                'detector': detector,
                'output_video_path': output_path,
                'total_frames': total_frames,
                'fps': fps,
                'processing_completed': True
            }
            
            print("Résultats sauvegardés")
            print("=== TRAITEMENT TERMINÉ ===")
            
            # Afficher un message de succès
            st.success("✅ Analyse terminée avec succès ! Consultez les résultats ci-dessous.")
            
            # Nettoyer le fichier temporaire d'entrée
            os.unlink(input_path)
            
    except Exception as e:
        print(f"ERREUR: {str(e)}")
        st.error(f"Erreur lors du traitement: {str(e)}")
    
    finally:
        # Sauvegarder la sortie console
        st.session_state.console_output = console_output.getvalue()
        # Ne pas relancer automatiquement

def display_results():
    """Affiche les résultats de l'analyse"""
    results = st.session_state.video_results
    detections = results['detections']
    detector = results['detector']
    
    # Statistiques générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Frames analysées", results['total_frames'])
    
    with col2:
        st.metric("Détections totales", len(detections))
    
    with col3:
        unique_faces = len(set(d['face_id'] for d in detections))
        st.metric("Visages uniques", unique_faces)
    
    with col4:
        if 'processing_completed' in results:
            st.metric("Statut", "✅ Terminé")
        else:
            st.metric("Statut", "🔄 En cours")
    
    # Tableau des détections
    if detections:
        df = detector.get_detections_dataframe()
        
        # Supprimer la colonne bbox pour l'affichage
        display_df = df.drop('bbox', axis=1, errors='ignore')
        st.dataframe(display_df, use_container_width=True)
        
        # Boutons d'export
        col1, col2 = st.columns(2)
        
        with col1:
            # Export CSV
            csv_data = display_df.to_csv(index=False, sep=';')
            st.download_button(
                label="Télécharger CSV",
                data=csv_data,
                file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export vidéo annotée
            if os.path.exists(results['output_video_path']):
                with open(results['output_video_path'], 'rb') as f:
                    st.download_button(
                        label="Télécharger Vidéo",
                        data=f.read(),
                        file_name=f"video_analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4"
                    )
    else:
        st.warning("Aucun visage détecté dans la vidéo.") 