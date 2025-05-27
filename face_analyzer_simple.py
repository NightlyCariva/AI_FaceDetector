import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import tensorflow as tf
import logging

# Configuration du logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

class FaceAnalyzer:
    def __init__(self):
        """Initialise l'analyseur de visages avec tous les modèles nécessaires"""
        # MediaPipe pour la détection de visages
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Couleurs pour les annotations
        self.colors = {
            'face_box': (0, 255, 0),
            'text_bg': (0, 0, 0),
            'text': (255, 255, 255),
            'emotion_happy': (0, 255, 255),
            'emotion_sad': (255, 0, 0),
            'emotion_angry': (0, 0, 255),
            'emotion_neutral': (128, 128, 128)
        }
    
    def detect_faces(self, image):
        """Détecte les visages dans une image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                # Convertir les coordonnées relatives en absolues
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # S'assurer que les coordonnées sont dans les limites de l'image
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': detection.score[0]
                })
        
        return faces
    
    def analyze_face_attributes(self, image, bbox):
        """Analyse les attributs d'un visage (âge, sexe, ethnie, émotion)"""
        x, y, w, h = bbox
        
        # Extraire la région du visage avec une marge
        margin = 20
        face_region = image[
            max(0, y-margin):min(image.shape[0], y+h+margin),
            max(0, x-margin):min(image.shape[1], x+w+margin)
        ]
        
        if face_region.size == 0:
            return None
        
        try:
            # Analyse avec DeepFace pour âge, sexe, ethnie, émotion
            analysis = DeepFace.analyze(
                face_region, 
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Si analysis est une liste, prendre le premier élément
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            # Extraire les informations
            age = analysis.get('age', 'Unknown')
            gender = analysis.get('dominant_gender', 'Unknown')
            race = analysis.get('dominant_race', 'Unknown')
            emotion = analysis.get('dominant_emotion', 'Unknown')
            
            # Calculer la confiance pour l'émotion
            emotion_scores = analysis.get('emotion', {})
            emotion_confidence = emotion_scores.get(emotion, 0) if emotion_scores else 0
            
            return {
                'age': age,
                'gender': gender,
                'race': race,
                'emotion': emotion,
                'emotion_confidence': emotion_confidence
            }
            
        except Exception as e:
            print(f"Erreur lors de l'analyse du visage: {e}")
            return {
                'age': 'Unknown',
                'gender': 'Unknown', 
                'race': 'Unknown',
                'emotion': 'Unknown',
                'emotion_confidence': 0
            }
    
    def draw_annotations(self, image, faces_data):
        """Dessine les annotations sur l'image"""
        annotated_image = image.copy()
        
        for face_data in faces_data:
            bbox = face_data['bbox']
            attributes = face_data['attributes']
            
            if attributes is None:
                continue
                
            x, y, w, h = bbox
            
            # Dessiner le rectangle du visage
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), self.colors['face_box'], 2)
            
            # Préparer le texte d'annotation
            age_text = f"Age: {attributes['age']}"
            gender_text = f"Genre: {attributes['gender']}"
            race_text = f"Ethnie: {attributes['race']}"
            emotion_text = f"Emotion: {attributes['emotion']}"
            
            # Position pour le texte
            text_y = y - 10
            line_height = 25
            
            # Fonction pour dessiner du texte avec arrière-plan
            def draw_text_with_bg(img, text, pos, font_scale=0.6, thickness=1):
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Arrière-plan noir pour le texte
                cv2.rectangle(img, 
                            (pos[0], pos[1] - text_height - 5),
                            (pos[0] + text_width, pos[1] + 5),
                            self.colors['text_bg'], -1)
                
                # Texte blanc
                cv2.putText(img, text, pos, font, font_scale, self.colors['text'], thickness)
                
                return pos[1] - line_height
            
            # Dessiner toutes les annotations
            if text_y > 0:
                text_y = draw_text_with_bg(annotated_image, age_text, (x, text_y))
                text_y = draw_text_with_bg(annotated_image, gender_text, (x, text_y))
                text_y = draw_text_with_bg(annotated_image, race_text, (x, text_y))
                text_y = draw_text_with_bg(annotated_image, emotion_text, (x, text_y))
        
        return annotated_image
    
    def process_frame(self, frame):
        """Traite une frame complète"""
        # Détecter les visages
        faces = self.detect_faces(frame)
        
        # Analyser chaque visage
        faces_data = []
        for face in faces:
            attributes = self.analyze_face_attributes(frame, face['bbox'])
            faces_data.append({
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'attributes': attributes
            })
        
        # Dessiner les annotations
        annotated_frame = self.draw_annotations(frame, faces_data)
        
        return annotated_frame, faces_data
    
    def process_video(self, input_path, output_path, progress_callback=None):
        """Traite une vidéo complète"""
        cap = cv2.VideoCapture(input_path)
        
        # Obtenir les propriétés de la vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Configurer l'encodeur vidéo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Traiter la frame
            annotated_frame, faces_data = self.process_frame(frame)
            
            # Sauvegarder les détections
            all_detections.append({
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'faces': faces_data
            })
            
            # Écrire la frame annotée
            out.write(annotated_frame)
            
            frame_count += 1
            
            # Callback pour le progrès
            if progress_callback:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)
        
        # Nettoyer
        cap.release()
        out.release()
        
        return all_detections 