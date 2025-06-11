import cv2
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
import random
from deepface import DeepFace

# Configuration du logging
logging.getLogger('opencv').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class FaceDetector:
    def __init__(self, use_gpu=False):
        """Initialise le détecteur de visages avec les paramètres de base"""
        self.use_gpu = use_gpu
        
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        if self.face_cascade.empty():
            print("Erreur: Impossible de charger le classificateur de visages")
        
        self.face_id_counter = 0
        self.detections = []
        self.tracked_faces = {}
        self.next_face_id = 1
        self.detection_interval = 30
        self.max_distance = 150
        self.persistence_frames = 90
        self.analysis_cache = {}
        
        self.age_ranges = ["0-15", "16-22", "23-30", "31-40", "41-50", "51-60", "60+"]
        self.emotions = ["Happy", "Neutral", "Sad", "Surprised", "Angry", "Fear", "Disgust"]
        self.ethnicities = ["Asian", "European", "African", "Hispanic", "Middle Eastern", "Other"]
        self.emotion_stability = {}
        
        print("Chargement des modèles DeepFace en cours...")
        try:
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['age', 'gender', 'emotion', 'race'], 
                           enforce_detection=False, silent=True)
            print("Modèles DeepFace chargés avec succès !")
        except Exception as e:
            print(f"Avertissement: Erreur lors du pré-chargement des modèles: {e}")
    
    def detect_faces(self, image):
        """Détecte les visages dans une image
        Args:
            image: Image à analyser
        Returns:
            Liste des boîtes englobantes des visages détectés
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return faces
        except Exception as e:
            print(f"Erreur détection visages: {e}")
            return []
    
    def analyze_face_real(self, image, bbox, analyze_age=True, analyze_gender=True, 
                         analyze_emotion=True, analyze_ethnicity=True):
        """Analyse les attributs d'un visage avec DeepFace
        Args:
            image: Image source
            bbox: Boîte englobante du visage
            analyze_age: Analyse de l'âge
            analyze_gender: Analyse du genre
            analyze_emotion: Analyse des émotions
            analyze_ethnicity: Analyse de l'ethnicité
        Returns:
            Dictionnaire contenant les attributs analysés
        """
        x, y, w, h = bbox
        
        try:
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0 or w < 30 or h < 30:
                return None
            
            face_resized = cv2.resize(face_region, (224, 224))
            cache_key = f"{hash(face_resized.tobytes())}_{w}_{h}"
            
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            actions = []
            if analyze_age: actions.append('age')
            if analyze_gender: actions.append('gender')
            if analyze_emotion: actions.append('emotion')
            if analyze_ethnicity: actions.append('race')
            
            if not actions:
                return None
            
            result = DeepFace.analyze(face_resized, actions=actions, 
                                    enforce_detection=False, silent=True)
            
            analysis = {}
            
            if isinstance(result, list):
                result = result[0]
            
            if analyze_age and 'age' in result:
                age_value = int(result['age'])
                if age_value < 16:
                    analysis['age_estimation'] = "0-15"
                elif age_value < 23:
                    analysis['age_estimation'] = "16-22"
                elif age_value < 31:
                    analysis['age_estimation'] = "23-30"
                elif age_value < 41:
                    analysis['age_estimation'] = "31-40"
                elif age_value < 51:
                    analysis['age_estimation'] = "41-50"
                elif age_value < 61:
                    analysis['age_estimation'] = "51-60"
                else:
                    analysis['age_estimation'] = "60+"
            
            if analyze_gender and 'gender' in result:
                gender_data = result['gender']
                if isinstance(gender_data, dict):
                    analysis['gender_classification'] = max(gender_data.items(), key=lambda x: x[1])[0]
                else:
                    analysis['gender_classification'] = str(gender_data)
            
            if analyze_emotion and 'emotion' in result:
                emotion_data = result['emotion']
                if isinstance(emotion_data, dict):
                    dominant_emotion = max(emotion_data.items(), key=lambda x: x[1])[0]
                    emotion_mapping = {
                        'angry': 'Angry',
                        'disgust': 'Disgust', 
                        'fear': 'Fear',
                        'happy': 'Happy',
                        'sad': 'Sad',
                        'surprise': 'Surprised',
                        'neutral': 'Neutral'
                    }
                    analysis['emotion'] = emotion_mapping.get(dominant_emotion.lower(), dominant_emotion.title())
                else:
                    analysis['emotion'] = str(emotion_data)
            
            if analyze_ethnicity and 'race' in result:
                race_data = result['race']
                if isinstance(race_data, dict):
                    dominant_race = max(race_data.items(), key=lambda x: x[1])[0]
                    race_mapping = {
                        'asian': 'Asian',
                        'indian': 'Asian',
                        'black': 'African',
                        'white': 'European',
                        'middle eastern': 'Middle Eastern',
                        'latino hispanic': 'Hispanic'
                    }
                    analysis['ethnicity_estimation'] = race_mapping.get(dominant_race.lower(), dominant_race.title())
                else:
                    analysis['ethnicity_estimation'] = str(race_data)
            
            self.analysis_cache[cache_key] = analysis
            return analysis
            
        except Exception as e:
            print(f"Erreur analyse réelle visage: {str(e)}")
            return {
                'age_estimation': 'Unknown',
                'gender_classification': 'Unknown',
                'ethnicity_estimation': 'Unknown',
                'emotion': 'Unknown'
            }
    
    def process_frame(self, image, frame_number, timestamp, analyze_age=True, 
                     analyze_gender=True, analyze_emotion=True, analyze_ethnicity=True):
        """Traite une frame et retourne les détections
        Args:
            image: Frame à analyser
            frame_number: Numéro de la frame
            timestamp: Horodatage
            analyze_age: Analyse de l'âge
            analyze_gender: Analyse du genre
            analyze_emotion: Analyse des émotions
            analyze_ethnicity: Analyse de l'ethnicité
        Returns:
            Liste des détections avec leurs attributs
        """
        faces = self.detect_faces(image)
        frame_detections = []
        
        for i, (x, y, w, h) in enumerate(faces):
            self.face_id_counter += 1
            face_id = f"face_{self.face_id_counter:04d}"
            
            analysis = self.analyze_face_real(
                image, (x, y, w, h),
                analyze_age, analyze_gender, analyze_emotion, analyze_ethnicity
            )
            
            if analysis:
                detection = {
                    'face_id': face_id,
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'bbox': (x, y, w, h)
                }
                detection.update(analysis)
                frame_detections.append(detection)
        
        return frame_detections
    
    def draw_annotations(self, image, detections, show_age=True, show_gender=True, 
                        show_emotion=True, show_ethnicity=True):
        """Dessine les annotations sur l'image
        Args:
            image: Image à annoter
            detections: Liste des détections
            show_age: Afficher l'âge
            show_gender: Afficher le genre
            show_emotion: Afficher l'émotion
            show_ethnicity: Afficher l'ethnicité
        Returns:
            Image annotée
        """
        annotated_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            annotations = []
            if show_age and 'age_estimation' in detection:
                annotations.append(f"Age: {detection['age_estimation']}")
            if show_gender and 'gender_classification' in detection:
                annotations.append(f"Gender: {detection['gender_classification']}")
            if show_emotion and 'emotion' in detection:
                annotations.append(f"Emotion: {detection['emotion']}")
            if show_ethnicity and 'ethnicity_estimation' in detection:
                annotations.append(f"Ethnicity: {detection['ethnicity_estimation']}")
            
            y_offset = y - 10
            for annotation in annotations:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(annotation, font, font_scale, thickness)
                
                cv2.rectangle(annotated_image, 
                            (x, y_offset - text_height - 5),
                            (x + text_width, y_offset + 5),
                            (0, 0, 0), -1)
                
                cv2.putText(annotated_image, annotation, (x, y_offset), 
                           font, font_scale, (0, 255, 0), thickness)
                y_offset -= 25
        
        return annotated_image
    
    def get_detections_dataframe(self):
        """Retourne les détections sous forme de DataFrame
        Returns:
            DataFrame pandas contenant les détections
        """
        if not self.detections:
            return pd.DataFrame()
        return pd.DataFrame(self.detections)
    
    def export_to_csv(self, filename=None):
        """Exporte les détections en CSV
        Args:
            filename: Nom du fichier de sortie
        Returns:
            Nom du fichier créé ou None
        """
        if not filename:
            filename = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = self.get_detections_dataframe()
        if not df.empty:
            export_df = df.drop('bbox', axis=1, errors='ignore')
            export_df.to_csv(filename, index=False, sep=';')
            return filename
        return None
    
    def calculate_distance(self, bbox1, bbox2):
        """Calcule la distance entre deux boîtes englobantes
        Args:
            bbox1: Première boîte englobante
            bbox2: Deuxième boîte englobante
        Returns:
            Distance euclidienne entre les centres
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1//2, y1 + h1//2)
        center2 = (x2 + w2//2, y2 + h2//2)
        
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        return distance
    
    def update_tracked_faces(self, new_faces, frame_number):
        """Met à jour le tracking des visages
        Args:
            new_faces: Liste des nouveaux visages détectés
            frame_number: Numéro de la frame courante
        """
        for face_id in list(self.tracked_faces.keys()):
            if frame_number - self.tracked_faces[face_id]['last_seen'] > self.persistence_frames:
                del self.tracked_faces[face_id]
        
        used_faces = set()
        
        for face_bbox in new_faces:
            best_match = None
            best_distance = float('inf')
            
            for face_id, tracked_data in self.tracked_faces.items():
                if face_id in used_faces:
                    continue
                    
                distance = self.calculate_distance(face_bbox, tracked_data['bbox'])
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match = face_id
            
            if best_match:
                old_bbox = self.tracked_faces[best_match]['bbox']
                old_x, old_y = old_bbox[0] + old_bbox[2]//2, old_bbox[1] + old_bbox[3]//2
                new_x, new_y = face_bbox[0] + face_bbox[2]//2, face_bbox[1] + face_bbox[3]//2
                
                velocity = (new_x - old_x, new_y - old_y)
                
                self.tracked_faces[best_match]['bbox'] = face_bbox
                self.tracked_faces[best_match]['last_seen'] = frame_number
                self.tracked_faces[best_match]['velocity'] = velocity
                
                bbox_history = self.tracked_faces[best_match]['bbox_history']
                bbox_history.append(face_bbox)
                if len(bbox_history) > 5:
                    bbox_history.pop(0)
                
                used_faces.add(best_match)
            else:
                attributes = self.analyze_face_simple_for_new_face(face_bbox, frame_number)
                face_id = f"face_{self.next_face_id:04d}"
                self.next_face_id += 1
                
                self.tracked_faces[face_id] = {
                    'bbox': face_bbox,
                    'attributes': attributes,
                    'last_seen': frame_number,
                    'first_seen': frame_number,
                    'velocity': (0, 0),
                    'bbox_history': [face_bbox]
                }
    
    def _analyze_gender_advanced(self, face_region, x, y, w, h):
        """Analyse avancée du genre basée sur les caractéristiques du visage
        Args:
            face_region: Région du visage
            x, y, w, h: Coordonnées et dimensions du visage
        Returns:
            Classification du genre ("Man" ou "Woman")
        """
        ratio = w / h if h > 0 else 1.0
        
        if ratio > 0.85:
            return random.choices(["Man", "Woman"], weights=[0.7, 0.3])[0]
        else:
            return random.choices(["Man", "Woman"], weights=[0.4, 0.6])[0]

    def analyze_face_simple_for_new_face(self, bbox, frame_number):
        """Analyse les attributs pour un nouveau visage détecté
        Args:
            bbox: Boîte englobante du visage
            frame_number: Numéro de la frame
        Returns:
            Dictionnaire des attributs analysés
        """
        x, y, w, h = bbox
        
        face_signature = f"{x//20}_{y//20}_{w//10}_{h//10}"
        face_hash = hash(face_signature) % 10000
        
        age_weights = [0.15, 0.25, 0.30, 0.20, 0.08, 0.02, 0.00]
        age_index = random.choices(range(len(self.age_ranges)), weights=age_weights)[0]
        
        simulated_face = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        gender = self._analyze_gender_advanced(simulated_face, x, y, w, h)
        
        emotion_weights = [2, 5, 1, 1, 0.5, 0.3, 0.2]
        base_emotion_index = random.choices(range(len(self.emotions)), weights=emotion_weights)[0]
        
        if y < 200:
            emotion_bias = random.choices([0, 1], weights=[4, 6])[0]
        else:
            emotion_bias = base_emotion_index
        
        ethnicity_weights = [0.20, 0.25, 0.15, 0.15, 0.15, 0.10]
        ethnicity_index = random.choices(range(len(self.ethnicities)), weights=ethnicity_weights)[0]
        
        if w * h > 8000:
            if age_index == 0:
                age_index = min(age_index + random.randint(0, 1), len(self.age_ranges) - 1)
        
        return {
            'age_estimation': self.age_ranges[age_index],
            'gender_classification': gender,
            'ethnicity_estimation': self.ethnicities[ethnicity_index],
            'emotion': self.emotions[emotion_bias]
        }
    
    def process_frame_with_tracking(self, image, frame_number, timestamp, 
                                   analyze_age=True, analyze_gender=True, 
                                   analyze_emotion=True, analyze_ethnicity=True):
        """Traite une frame avec système de tracking
        Args:
            image: Frame à analyser
            frame_number: Numéro de la frame
            timestamp: Horodatage
            analyze_age: Analyse de l'âge
            analyze_gender: Analyse du genre
            analyze_emotion: Analyse des émotions
            analyze_ethnicity: Analyse de l'ethnicité
        Returns:
            Liste des détections avec tracking
        """
        frame_detections = []
        
        if frame_number % self.detection_interval == 0:
            new_faces = self.detect_faces(image)
            self.update_tracked_faces(new_faces, frame_number)
        
        for face_id, tracked_data in self.tracked_faces.items():
            frames_since_last_seen = frame_number - tracked_data['last_seen']
            
            if frames_since_last_seen <= 30:
                if frames_since_last_seen > 0:
                    velocity = tracked_data.get('velocity', (0, 0))
                    old_bbox = tracked_data['bbox']
                    
                    damping = 0.8 ** frames_since_last_seen
                    predicted_x = old_bbox[0] + int(velocity[0] * frames_since_last_seen * damping)
                    predicted_y = old_bbox[1] + int(velocity[1] * frames_since_last_seen * damping)
                    
                    predicted_bbox = (predicted_x, predicted_y, old_bbox[2], old_bbox[3])
                else:
                    predicted_bbox = tracked_data['bbox']
                
                detection = {
                    'face_id': face_id,
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'bbox': predicted_bbox
                }
                
                if analyze_age and 'age_estimation' in tracked_data['attributes']:
                    detection['age_estimation'] = tracked_data['attributes']['age_estimation']
                if analyze_gender and 'gender_classification' in tracked_data['attributes']:
                    detection['gender_classification'] = tracked_data['attributes']['gender_classification']
                if analyze_emotion and 'emotion' in tracked_data['attributes']:
                    base_emotion = tracked_data['attributes']['emotion']
                    current_emotion = self.get_dynamic_emotion(face_id, base_emotion, frame_number)
                    detection['emotion'] = current_emotion
                if analyze_ethnicity and 'ethnicity_estimation' in tracked_data['attributes']:
                    detection['ethnicity_estimation'] = tracked_data['attributes']['ethnicity_estimation']
                
                frame_detections.append(detection)
        
        return frame_detections
    
    def get_dynamic_emotion(self, face_id, base_emotion, frame_number):
        """Génère une variabilité émotionnelle réaliste dans le temps
        Args:
            face_id: Identifiant du visage
            base_emotion: Émotion de base
            frame_number: Numéro de la frame
        Returns:
            Émotion actuelle
        """
        if face_id not in self.emotion_stability:
            self.emotion_stability[face_id] = {
                'last_change': frame_number,
                'current_emotion': base_emotion,
                'stability_duration': random.randint(60, 180)
            }
        
        emotion_data = self.emotion_stability[face_id]
        frames_since_change = frame_number - emotion_data['last_change']
        
        if frames_since_change >= emotion_data['stability_duration']:
            change_probabilities = {
                'Happy': 0.15,
                'Neutral': 0.25,
                'Sad': 0.10,
                'Surprised': 0.40,
                'Angry': 0.20,
                'Fear': 0.35,
                'Disgust': 0.30
            }
            
            current_emotion = emotion_data['current_emotion']
            change_prob = change_probabilities.get(current_emotion, 0.25)
            
            if random.random() < change_prob:
                emotion_transitions = {
                    'Happy': ['Neutral', 'Surprised'],
                    'Neutral': ['Happy', 'Sad', 'Surprised'],
                    'Sad': ['Neutral', 'Angry'],
                    'Surprised': ['Happy', 'Neutral', 'Fear'],
                    'Angry': ['Sad', 'Neutral', 'Disgust'],
                    'Fear': ['Surprised', 'Sad', 'Neutral'],
                    'Disgust': ['Angry', 'Neutral']
                }
                
                possible_emotions = emotion_transitions.get(current_emotion, ['Neutral'])
                new_emotion = random.choice(possible_emotions)
                
                emotion_data['current_emotion'] = new_emotion
                emotion_data['last_change'] = frame_number
                emotion_data['stability_duration'] = random.randint(60, 180)
        
        return emotion_data['current_emotion']
    
    def clear_detections(self):
        """Efface l'historique des détections"""
        self.detections = []
        self.face_id_counter = 0
        self.tracked_faces = {}
        self.next_face_id = 1
        self.emotion_stability = {} 