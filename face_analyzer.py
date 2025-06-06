import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import tensorflow as tf
import logging
import unicodedata

# Import FER avec gestion d'erreur pour moviepy
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError as e:
    print(f"Avertissement: FER non disponible - {e}")
    FER_AVAILABLE = False

# Configuration du logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

def remove_accents(text):
    """Supprime les accents d'un texte pour compatibility OpenCV"""
    if isinstance(text, str):
        # Normaliser et supprimer les accents
        normalized = unicodedata.normalize('NFD', text)
        without_accents = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        return without_accents
    return str(text)

class FaceAnalyzer:
    def __init__(self):
        """Initialise l'analyseur de visages avec tous les modèles nécessaires"""
        # MediaPipe pour la détection de visages
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Précharger les modèles DeepFace pour éviter l'erreur de lazy loading
        self._preload_deepface_models()
        
        # FER pour la détection d'émotions (si disponible)
        if FER_AVAILABLE:
            try:
                self.emotion_detector = FER(mtcnn=True)
            except Exception as e:
                print(f"Erreur lors de l'initialisation de FER: {e}")
                self.emotion_detector = None
        else:
            self.emotion_detector = None
        
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
    
    def _preload_deepface_models(self):
        """Précharge les modèles DeepFace pour éviter les erreurs de lazy loading"""
        try:
            # Créer une image de test pour forcer le chargement des modèles
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_img.fill(128)  # Image grise
            
            # Forcer le chargement des modèles avec une analyse de test
            _ = DeepFace.analyze(
                test_img, 
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False,
                silent=True
            )
            print("Modèles DeepFace préchargés avec succès")
        except Exception as e:
            print(f"Avertissement: Impossible de précharger les modèles DeepFace: {e}")
    
    
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
        
        # Extraire la région du visage avec une marge plus importante
        margin = max(30, min(w, h) // 4)  # Marge adaptative
        face_region = image[
            max(0, y-margin):min(image.shape[0], y+h+margin),
            max(0, x-margin):min(image.shape[1], x+w+margin)
        ]
        
        if face_region.size == 0 or face_region.shape[0] < 10 or face_region.shape[1] < 10:
            return None
        
        try:
            # Redimensionner la région du visage pour une meilleure analyse
            target_size = 224  # Taille optimale pour DeepFace
            if face_region.shape[0] < target_size or face_region.shape[1] < target_size:
                face_region = cv2.resize(face_region, (target_size, target_size))
            
            # Analyse avec DeepFace pour âge, sexe, ethnie avec plus de tolérance
            analysis = DeepFace.analyze(
                face_region, 
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False,  # Ne pas forcer la détection de visage
                silent=True,
                detector_backend='opencv'  # Utiliser OpenCV comme détecteur plus rapide
            )
            
            # Si analysis est une liste, prendre le premier élément
            if isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]
            elif isinstance(analysis, list) and len(analysis) == 0:
                raise Exception("Aucune analyse retournée")
            
            # Extraire les informations avec valeurs par défaut améliorées
            age = analysis.get('age', None)
            if age is None or not isinstance(age, (int, float)) or age <= 0 or age > 100:
                age = "25-35"
            else:
                age = int(age)
            
            gender = analysis.get('dominant_gender', 'Non défini')
            if gender == 'Woman':
                gender = 'Femme'
            elif gender == 'Man':
                gender = 'Homme'
            
            # Mapper les races en français (sans accents pour OpenCV)
            race_mapping = {
                'asian': 'Asiatique',
                'indian': 'Indien',
                'black': 'Noir',
                'white': 'Blanc',
                'middle eastern': 'Moyen-Orient',
                'latino hispanic': 'Latino'
            }
            race = analysis.get('dominant_race', 'Non défini')
            race = race_mapping.get(race.lower(), race.title())
            
            # Mapper les émotions en français (sans accents pour OpenCV)
            emotion_mapping = {
                'angry': 'Colere',
                'disgust': 'Degout',
                'fear': 'Peur',
                'happy': 'Joie',
                'sad': 'Tristesse',
                'surprise': 'Surprise',
                'neutral': 'Neutre'
            }
            emotion = analysis.get('dominant_emotion', 'neutre')
            emotion = emotion_mapping.get(emotion.lower(), emotion.title())
            
            # Calculer la confiance pour l'émotion
            emotion_scores = analysis.get('emotion', {})
            emotion_confidence = emotion_scores.get(analysis.get('dominant_emotion', ''), 0) if emotion_scores else 0
            
            result = {
                'age': age,
                'gender': gender,
                'race': race,
                'emotion': emotion,
                'emotion_confidence': round(emotion_confidence, 2)
            }
            
            return result
            
        except Exception as e:
            # Retourner des valeurs par défaut plus informatives (sans accents)
            return {
                'age': 'Non analyse',
                'gender': 'Non analyse', 
                'race': 'Non analyse',
                'emotion': 'Non analyse',
                'emotion_confidence': 0
            }
    
    def draw_annotations(self, image, faces_data, show_age=True, show_gender=True, show_emotion=True, show_race=True):
        """Dessine les annotations sur l'image selon les options d'affichage"""
        annotated_image = image.copy()
        
        for face_data in faces_data:
            bbox = face_data['bbox']
            attributes = face_data['attributes']
            
            if attributes is None:
                continue
                
            x, y, w, h = bbox
            
            # Dessiner le rectangle du visage (toujours visible)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), self.colors['face_box'], 2)
            
            # Préparer le texte d'annotation selon les options (sans accents pour OpenCV)
            texts_to_show = []
            
            if show_age:
                texts_to_show.append(f"Age: {attributes['age']}")
            if show_gender:
                texts_to_show.append(f"Genre: {attributes['gender']}")
            if show_race:
                texts_to_show.append(f"Ethnie: {attributes['race']}")
            if show_emotion:
                texts_to_show.append(f"Emotion: {attributes['emotion']}")
            
            # Si aucune option n'est cochée, afficher juste "Visage detecte"
            if not texts_to_show:
                texts_to_show.append("Visage detecte")
            
            # Position pour le texte
            text_y = y - 10
            line_height = 25
            
            # Fonction pour dessiner du texte avec arrière-plan (avec nettoyage des accents)
            def draw_text_with_bg(img, text, pos, font_scale=0.6, thickness=1):
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Nettoyer le texte des accents pour OpenCV
                clean_text = remove_accents(text)
                (text_width, text_height), _ = cv2.getTextSize(clean_text, font, font_scale, thickness)
                
                # Arrière-plan noir pour le texte
                cv2.rectangle(img, 
                            (pos[0], pos[1] - text_height - 5),
                            (pos[0] + text_width, pos[1] + 5),
                            self.colors['text_bg'], -1)
                
                # Texte blanc
                cv2.putText(img, clean_text, pos, font, font_scale, self.colors['text'], thickness)
                
                return pos[1] - line_height
            
            # Dessiner toutes les annotations sélectionnées
            for text in texts_to_show:
                if text_y > 0:
                    text_y = draw_text_with_bg(annotated_image, text, (x, text_y))
        
        return annotated_image
    
    def process_frame(self, frame, show_age=True, show_gender=True, show_emotion=True, show_race=True, confidence_threshold=0.5):
        """Traite une frame complète avec options d'affichage"""
        # Détecter les visages avec le seuil de confiance
        faces = self.detect_faces(frame)
        
        # Filtrer les visages selon le seuil de confiance
        filtered_faces = [face for face in faces if face['confidence'] >= confidence_threshold]
        
        # Analyser chaque visage
        faces_data = []
        for i, face in enumerate(filtered_faces):
            # Seulement analyser les visages assez grands
            x, y, w, h = face['bbox']
            if w > 50 and h > 50:  # Minimum 50x50 pixels
                attributes = self.analyze_face_attributes(frame, face['bbox'])
                faces_data.append({
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'attributes': attributes
                })
            else:
                # Ajouter quand même avec des attributs par défaut
                faces_data.append({
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'attributes': {
                        'age': 'Trop petit',
                        'gender': 'N/A',
                        'race': 'N/A',
                        'emotion': 'N/A',
                        'emotion_confidence': 0
                    }
                })
        
        # Dessiner les annotations avec les options d'affichage
        annotated_frame = self.draw_annotations(
            frame, faces_data, 
            show_age=show_age, 
            show_gender=show_gender, 
            show_emotion=show_emotion, 
            show_race=show_race
        )
        
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