import cv2
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
import random

# Configuration du logging
logging.getLogger('opencv').setLevel(logging.ERROR)

class FaceDetector:
    def __init__(self, use_gpu=False):
        """Initialise le détecteur de visages"""
        self.use_gpu = use_gpu
        
        # Initialiser le détecteur de visages OpenCV
        try:
            # Essayer d'utiliser le chemin des cascades OpenCV
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            # Fallback si cv2.data n'est pas disponible
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Vérifier que le cascade est bien chargé
        if self.face_cascade.empty():
            print("Erreur: Impossible de charger le classificateur de visages")
        
        # Compteur pour les IDs de visages
        self.face_id_counter = 0
        
        # Historique des détections
        self.detections = []
        
        # Système de tracking
        self.tracked_faces = {}  # ID -> {bbox, attributes, last_seen}
        self.next_face_id = 1
        self.detection_interval = 30  # Détecter les nouveaux visages toutes les 30 frames
        self.max_distance = 150  # Distance max pour associer un visage (augmentée)
        self.persistence_frames = 90  # Frames de persistance après perte (3 secondes à 30fps)
        
        # Listes pour simulation intelligente - Plus réalistes
        self.age_ranges = ["16-22", "23-30", "31-40", "41-50", "51-60", "60+"]
        self.emotions = ["Happy", "Neutral", "Sad", "Surprised", "Angry", "Fear", "Disgust"]
        self.ethnicities = ["European", "Asian", "African", "Hispanic", "Middle Eastern", "Mixed"]
        self.genders = ["Male", "Female"]
        
        # Variabilité temporelle pour les émotions
        self.emotion_stability = {}  # face_id -> emotion_timer
    
    def detect_faces(self, image):
        """Détecte les visages dans une image"""
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
    
    def analyze_face_simple(self, image, bbox, analyze_age=True, analyze_gender=True, 
                           analyze_emotion=True, analyze_ethnicity=True):
        """Analyse simplifiée des attributs d'un visage avec simulation intelligente"""
        x, y, w, h = bbox
        
        try:
            # Extraire la région du visage
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return None
            
            # Simulation intelligente basée sur des caractéristiques de l'image
            # Utiliser des propriétés de l'image pour créer une cohérence
            face_hash = hash(str(face_region.mean())) % 1000
            
            analysis = {}
            
            if analyze_age:
                # Simuler l'âge basé sur des caractéristiques de l'image
                age_index = (face_hash + int(face_region.mean())) % len(self.age_ranges)
                analysis['age_estimation'] = self.age_ranges[age_index]
            
            if analyze_gender:
                # Simuler le genre
                gender_index = face_hash % len(self.genders)
                analysis['gender_classification'] = self.genders[gender_index]
            
            if analyze_ethnicity:
                # Simuler l'ethnie
                ethnicity_index = (face_hash * 2) % len(self.ethnicities)
                analysis['ethnicity_estimation'] = self.ethnicities[ethnicity_index]
            
            if analyze_emotion:
                # Simuler l'émotion avec une tendance vers "Happy" et "Neutral"
                emotion_weights = [3, 4, 1, 1, 1, 1, 1]  # Plus de chances pour Happy et Neutral
                emotion_index = random.choices(range(len(self.emotions)), weights=emotion_weights)[0]
                analysis['emotion'] = self.emotions[emotion_index]
            
            return analysis
            
        except Exception as e:
            print(f"Erreur analyse visage: {str(e)}")
            return {
                'age_estimation': 'Unknown',
                'gender_classification': 'Unknown',
                'ethnicity_estimation': 'Unknown',
                'emotion': 'Unknown'
            }
    
    def process_frame(self, image, frame_number, timestamp, analyze_age=True, 
                     analyze_gender=True, analyze_emotion=True, analyze_ethnicity=True):
        """Traite une frame et retourne les détections"""
        faces = self.detect_faces(image)
        frame_detections = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Générer un ID unique pour ce visage
            self.face_id_counter += 1
            face_id = f"face_{self.face_id_counter:04d}"
            
            # Analyser le visage
            analysis = self.analyze_face_simple(
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
        """Dessine les annotations sur l'image"""
        annotated_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # Dessiner le rectangle du visage
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Préparer le texte d'annotation
            annotations = []
            if show_age and 'age_estimation' in detection:
                annotations.append(f"Age: {detection['age_estimation']}")
            if show_gender and 'gender_classification' in detection:
                annotations.append(f"Gender: {detection['gender_classification']}")
            if show_emotion and 'emotion' in detection:
                annotations.append(f"Emotion: {detection['emotion']}")
            if show_ethnicity and 'ethnicity_estimation' in detection:
                annotations.append(f"Ethnicity: {detection['ethnicity_estimation']}")
            
            # Dessiner le texte avec arrière-plan
            y_offset = y - 10
            for annotation in annotations:
                # Mesurer le texte
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(annotation, font, font_scale, thickness)
                
                # Dessiner l'arrière-plan
                cv2.rectangle(annotated_image, 
                            (x, y_offset - text_height - 5),
                            (x + text_width, y_offset + 5),
                            (0, 0, 0), -1)
                
                # Dessiner le texte
                cv2.putText(annotated_image, annotation, (x, y_offset), 
                           font, font_scale, (0, 255, 0), thickness)
                y_offset -= 25
        
        return annotated_image
    
    def get_detections_dataframe(self):
        """Retourne les détections sous forme de DataFrame"""
        if not self.detections:
            return pd.DataFrame()
        
        return pd.DataFrame(self.detections)
    
    def export_to_csv(self, filename=None):
        """Exporte les détections en CSV"""
        if not filename:
            filename = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = self.get_detections_dataframe()
        if not df.empty:
            # Supprimer la colonne bbox pour l'export
            export_df = df.drop('bbox', axis=1, errors='ignore')
            export_df.to_csv(filename, index=False, sep=';')
            return filename
        return None
    
    def calculate_distance(self, bbox1, bbox2):
        """Calcule la distance entre deux boîtes englobantes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Centres des boîtes
        center1 = (x1 + w1//2, y1 + h1//2)
        center2 = (x2 + w2//2, y2 + h2//2)
        
        # Distance euclidienne
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        return distance
    
    def update_tracked_faces(self, new_faces, frame_number):
        """Met à jour le tracking des visages avec persistance améliorée"""
        # Nettoyer les visages trop anciens seulement
        for face_id in list(self.tracked_faces.keys()):
            if frame_number - self.tracked_faces[face_id]['last_seen'] > self.persistence_frames:
                del self.tracked_faces[face_id]
        
        # Associer les nouveaux visages aux visages trackés
        used_faces = set()
        
        for face_bbox in new_faces:
            best_match = None
            best_distance = float('inf')
            
            # Chercher le visage tracké le plus proche
            for face_id, tracked_data in self.tracked_faces.items():
                if face_id in used_faces:
                    continue
                    
                distance = self.calculate_distance(face_bbox, tracked_data['bbox'])
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match = face_id
            
            if best_match:
                # Calculer la vitesse de déplacement
                old_bbox = self.tracked_faces[best_match]['bbox']
                old_x, old_y = old_bbox[0] + old_bbox[2]//2, old_bbox[1] + old_bbox[3]//2
                new_x, new_y = face_bbox[0] + face_bbox[2]//2, face_bbox[1] + face_bbox[3]//2
                
                velocity = (new_x - old_x, new_y - old_y)
                
                # Mettre à jour le visage existant
                self.tracked_faces[best_match]['bbox'] = face_bbox
                self.tracked_faces[best_match]['last_seen'] = frame_number
                self.tracked_faces[best_match]['velocity'] = velocity
                
                # Maintenir un historique des 5 dernières positions
                bbox_history = self.tracked_faces[best_match]['bbox_history']
                bbox_history.append(face_bbox)
                if len(bbox_history) > 5:
                    bbox_history.pop(0)
                
                used_faces.add(best_match)
            else:
                # Nouveau visage - analyser ses attributs
                attributes = self.analyze_face_simple_for_new_face(face_bbox, frame_number)
                face_id = f"face_{self.next_face_id:04d}"
                self.next_face_id += 1
                
                self.tracked_faces[face_id] = {
                    'bbox': face_bbox,
                    'attributes': attributes,
                    'last_seen': frame_number,
                    'first_seen': frame_number,
                    'velocity': (0, 0),  # Vitesse de déplacement
                    'bbox_history': [face_bbox]  # Historique des positions
                }
    
    def analyze_face_simple_for_new_face(self, bbox, frame_number):
        """Analyse les attributs pour un nouveau visage détecté"""
        x, y, w, h = bbox
        
        # Créer un hash stable basé sur la position et taille du visage
        face_signature = f"{x//20}_{y//20}_{w//10}_{h//10}"  # Réduire la sensibilité aux petits mouvements
        face_hash = hash(face_signature) % 10000
        
        # === SIMULATION AMÉLIORÉE DES ATTRIBUTS ===
        
        # AGE - Distribution plus réaliste
        age_weights = [0.15, 0.25, 0.30, 0.20, 0.08, 0.02]  # Plus de jeunes adultes
        age_index = random.choices(range(len(self.age_ranges)), weights=age_weights)[0]
        
        # GENRE - Equilibré avec légère variation basée sur la position
        gender_index = (face_hash + x) % 2
        
        # EMOTION - Distribution plus réaliste
        # Pondération: Neutral > Happy > Autres émotions
        emotion_weights = [2, 5, 1, 1, 0.5, 0.3, 0.2]  # Happy, Neutral dominant
        base_emotion_index = random.choices(range(len(self.emotions)), weights=emotion_weights)[0]
        
        # Ajouter une variabilité basée sur la position Y (visages en haut plus heureux)
        if y < 200:  # Visages en haut de l'image
            emotion_bias = random.choices([0, 1], weights=[4, 6])[0]  # Plus Happy/Neutral
        else:
            emotion_bias = base_emotion_index
        
        # ETHNIE - Distribution plus équilibrée
        ethnicity_weights = [0.20, 0.25, 0.15, 0.15, 0.15, 0.10]  # Distribution équilibrée
        ethnicity_index = random.choices(range(len(self.ethnicities)), weights=ethnicity_weights)[0]
        
        # Logique de cohérence : certaines combinaisons sont plus probables
        # Par exemple, ajuster l'âge selon la taille du visage
        if w * h > 8000:  # Grand visage (proche)
            # Plus de chances d'être un adulte
            if age_index == 0:  # Si jeune, parfois vieillir
                age_index = min(age_index + random.randint(0, 1), len(self.age_ranges) - 1)
        
        return {
            'age_estimation': self.age_ranges[age_index],
            'gender_classification': self.genders[gender_index],
            'ethnicity_estimation': self.ethnicities[ethnicity_index],
            'emotion': self.emotions[emotion_bias]
        }
    
    def process_frame_with_tracking(self, image, frame_number, timestamp, 
                                   analyze_age=True, analyze_gender=True, 
                                   analyze_emotion=True, analyze_ethnicity=True):
        """Traite une frame avec système de tracking"""
        frame_detections = []
        
        # Détecter de nouveaux visages seulement à intervalle régulier
        if frame_number % self.detection_interval == 0:
            new_faces = self.detect_faces(image)
            self.update_tracked_faces(new_faces, frame_number)
        
        # Utiliser tous les visages trackés actifs avec persistance améliorée
        for face_id, tracked_data in self.tracked_faces.items():
            # Calculer l'âge depuis la dernière détection
            frames_since_last_seen = frame_number - tracked_data['last_seen']
            
            # Maintenir les cadres plus longtemps avec décroissance progressive
            if frames_since_last_seen <= 30:  # Tolérance augmentée à 30 frames (1 seconde)
                
                # Prédire la position si le visage n'est pas détecté récemment
                if frames_since_last_seen > 0:
                    # Utiliser la vitesse pour prédire la nouvelle position
                    velocity = tracked_data.get('velocity', (0, 0))
                    old_bbox = tracked_data['bbox']
                    
                    # Prédiction avec amortissement de la vitesse
                    damping = 0.8 ** frames_since_last_seen  # Réduction progressive
                    predicted_x = old_bbox[0] + int(velocity[0] * frames_since_last_seen * damping)
                    predicted_y = old_bbox[1] + int(velocity[1] * frames_since_last_seen * damping)
                    
                    predicted_bbox = (predicted_x, predicted_y, old_bbox[2], old_bbox[3])
                else:
                    predicted_bbox = tracked_data['bbox']
                
                # Créer la détection avec les attributs stockés
                detection = {
                    'face_id': face_id,
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'bbox': predicted_bbox
                }
                
                # Ajouter les attributs selon les options
                if analyze_age and 'age_estimation' in tracked_data['attributes']:
                    detection['age_estimation'] = tracked_data['attributes']['age_estimation']
                if analyze_gender and 'gender_classification' in tracked_data['attributes']:
                    detection['gender_classification'] = tracked_data['attributes']['gender_classification']
                if analyze_emotion and 'emotion' in tracked_data['attributes']:
                    # Variabilité émotionnelle dans le temps
                    base_emotion = tracked_data['attributes']['emotion']
                    current_emotion = self.get_dynamic_emotion(face_id, base_emotion, frame_number)
                    detection['emotion'] = current_emotion
                if analyze_ethnicity and 'ethnicity_estimation' in tracked_data['attributes']:
                    detection['ethnicity_estimation'] = tracked_data['attributes']['ethnicity_estimation']
                
                frame_detections.append(detection)
        
        return frame_detections
    
    def get_dynamic_emotion(self, face_id, base_emotion, frame_number):
        """Génère une variabilité émotionnelle réaliste dans le temps"""
        
        # Initialiser le timer pour ce visage si nécessaire
        if face_id not in self.emotion_stability:
            self.emotion_stability[face_id] = {
                'last_change': frame_number,
                'current_emotion': base_emotion,
                'stability_duration': random.randint(60, 180)  # 2-6 secondes à 30fps
            }
        
        emotion_data = self.emotion_stability[face_id]
        frames_since_change = frame_number - emotion_data['last_change']
        
        # Changer d'émotion si assez de temps s'est écoulé
        if frames_since_change >= emotion_data['stability_duration']:
            
            # Probabilité de changement selon l'émotion actuelle
            change_probabilities = {
                'Happy': 0.15,      # Les gens heureux restent généralement heureux
                'Neutral': 0.25,    # Neutral change plus facilement
                'Sad': 0.10,        # Tristesse assez stable
                'Surprised': 0.40,  # Surprise change vite
                'Angry': 0.20,      # Colère modérément stable
                'Fear': 0.35,       # Peur change assez vite
                'Disgust': 0.30     # Dégoût change assez vite
            }
            
            current_emotion = emotion_data['current_emotion']
            change_prob = change_probabilities.get(current_emotion, 0.25)
            
            if random.random() < change_prob:
                # Changer vers une émotion liée
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
                
                # Mettre à jour
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