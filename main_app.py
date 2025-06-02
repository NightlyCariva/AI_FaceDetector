import streamlit as st
import subprocess
import sys
import os

st.set_page_config(
    page_title="Analyseur de Visages - Protection des Données",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 4rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .mode-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .mode-card h3 {
        color: white;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 2rem 0;
        border-left: 5px solid #f39c12;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 2rem 0;
        border-left: 5px solid #17a2b8;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    .feature-item {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .gdpr-section {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎭 Analyseur de Visages IA</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Projet de Démonstration - Protection des Données</p>', unsafe_allow_html=True)
    
    # Avertissement principal RGPD
    st.markdown("""
    <div class="warning-box">
        <h2>⚠️ AVERTISSEMENT IMPORTANT - RGPD</h2>
        <p><strong>Cette application traite des données biométriques sensibles.</strong></p>
        <p>Selon le Règlement Général sur la Protection des Données (RGPD), les données biométriques 
        sont considérées comme des <strong>données à caractère personnel de catégorie particulière</strong> 
        (Article 9 du RGPD).</p>
        <h4>Avant d'utiliser cette application :</h4>
        <ul>
            <li>✅ Assurez-vous d'avoir le <strong>consentement explicite</strong> de toutes les personnes filmées</li>
            <li>✅ Informez clairement sur la <strong>finalité du traitement</strong></li>
            <li>✅ Respectez les principes de <strong>minimisation des données</strong></li>
            <li>✅ Garantissez la <strong>sécurité</strong> des données collectées</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation dans la sidebar
    with st.sidebar:
        st.header("🧭 Navigation")
        
        mode = st.radio(
            "Choisissez un mode :",
            ["🏠 Accueil", "📤 Mode 1: Upload Vidéo", "📹 Mode 2: Temps Réel"],
            index=0
        )
        
        st.markdown("---")
        
        st.header("📊 Informations Projet")
        st.markdown("""
        **Objectif :** Démonstration des enjeux de protection des données dans l'IA
        
        **Technologies utilisées :**
        - OpenCV
        - MediaPipe
        - DeepFace
        - Streamlit
        - TensorFlow
        """)
        
        st.markdown("---")
        
        st.header("⚖️ Conformité RGPD")
        st.markdown("""
        **Articles concernés :**
        - Art. 6 : Licéité du traitement
        - Art. 9 : Traitement portant sur des catégories particulières de données à caractère personnel
        - Art. 13-14: Informations à fournir lorsque des données à caractère personnel sont collectées auprès de la personne concernée
        - Art. 25 : Protection des données dès la conception et protection des données par défaut
        """)
    
    if mode == "🏠 Accueil":
        show_home_page()
    elif mode == "📤 Mode 1: Upload Vidéo":
        show_mode1_info()
    elif mode == "📹 Mode 2: Temps Réel":
        show_mode2_info()

def show_home_page():
    """Affiche la page d'accueil avec présentation du projet"""
    
    # Présentation du projet
    st.markdown("## 🎯 Objectif du Projet")
    st.markdown("""
    Ce projet a été développé dans le cadre d'un cours sur la **Protection des Données** pour illustrer 
    les défis et enjeux liés au traitement de données biométriques par l'Intelligence Artificielle.
    """)
    
    # Fonctionnalités
    st.markdown("## 🚀 Fonctionnalités")
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-item">
            <h4>👤 Détection de Visages</h4>
            <p>Identification automatique des visages dans les images/vidéos</p>
        </div>
        <div class="feature-item">
            <h4>🎂 Estimation d'Âge</h4>
            <p>Prédiction de la tranche d'âge des personnes détectées</p>
        </div>
        <div class="feature-item">
            <h4>⚧️ Classification Genre</h4>
            <p>Identification du genre apparent des individus</p>
        </div>
        <div class="feature-item">
            <h4>🌍 Origine Ethnique</h4>
            <p>Estimation de l'origine ethnique (données sensibles)</p>
        </div>
        <div class="feature-item">
            <h4>😊 Reconnaissance Émotions</h4>
            <p>Analyse de l'état émotionnel des personnes</p>
        </div>
        <div class="feature-item">
            <h4>📊 Analyse Statistique</h4>
            <p>Génération de rapports et visualisations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎛️ Modes Disponibles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="mode-card">
            <h3>📤 Mode 1: Upload Vidéo</h3>
            <p>Analysez des vidéos pré-enregistrées</p>
            <ul style="text-align: left;">
                <li>Upload de fichiers MP4, AVI, MOV</li>
                <li>Traitement complet de la vidéo</li>
                <li>Génération de rapport détaillé</li>
                <li>Téléchargement des résultats</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="mode-card">
            <h3>📹 Mode 2: Temps Réel</h3>
            <p>Analyse en direct via webcam</p>
            <ul style="text-align: left;">
                <li>Flux vidéo en temps réel</li>
                <li>Détection instantanée</li>
                <li>Statistiques live</li>
                <li>Compatible téléphone</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="gdpr-section">
        <h2>🔒 Enjeux de Protection des Données</h2>
        <h3>Pourquoi ce projet ?</h3>
        <p>L'Intelligence Artificielle soulève de nombreuses questions en matière de protection des données, 
        particulièrement quand elle traite des <strong>données biométriques</strong>. Ce projet permet de :</p>
        <ul>
            <li>🎓 <strong>Comprendre</strong> les implications du RGPD dans l'IA</li>
            <li>⚖️ <strong>Identifier</strong> les données sensibles collectées</li>
            <li>🛡️ <strong>Appliquer</strong> les principes de privacy by design</li>
            <li>📋 <strong>Documenter</strong> les traitements de données</li>
        </ul>
        <h3>Données Sensibles Traitées</h3>
        <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <p><strong>Article 9 RGPD - Catégories particulières :</strong></p>
            <ul>
                <li>🧬 <strong>Données biométriques</strong> (géométrie faciale)</li>
                <li>🌍 <strong>Origine raciale/ethnique</strong> (estimation)</li>
                <li>⚧️ <strong>Données relatives au genre</strong></li>
                <li>🧠 <strong>État psychologique</strong> (émotions)</li>
            </ul>
        </div>
        <h3>Mesures de Protection Implémentées</h3>
        <ul>
            <li>✅ <strong>Traitement local</strong> - Aucune donnée envoyée vers des serveurs externes</li>
            <li>✅ <strong>Transparence</strong> - Information claire sur les données collectées</li>
            <li>✅ <strong>Minimisation</strong> - Collecte uniquement des données nécessaires</li>
            <li>✅ <strong>Limitation de conservation</strong> - Historique limité dans le temps</li>
            <li>✅ <strong>Droit à l'effacement</strong> - Possibilité de supprimer les données</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def show_mode1_info():
    """Affiche les informations et lance le mode 1"""
    st.markdown("## 📤 Mode 1: Analyse de Vidéos Uploadées")
    
    st.markdown("""
    <div class="info-box">
        <h3>Fonctionnement du Mode 1</h3>
        <p>Ce mode permet d'analyser des vidéos pré-enregistrées en uploadant le fichier. 
        L'analyse complète est effectuée et vous pouvez télécharger les résultats.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Avantages et inconvénients
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Avantages
        - Analyse complète de la vidéo
        - Rapport détaillé avec statistiques
        - Possibilité de télécharger les résultats
        - Traitement optimisé pour la qualité
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Considérations RGPD
        - Stockage temporaire du fichier uploadé
        - Génération d'un fichier de sortie annoté
        - Conservation des données d'analyse
        - Nécessité du consentement pour les personnes filmées
        """)
    
    # Bouton pour lancer le mode 1
    if st.button("🚀 Lancer le Mode 1", type="primary", use_container_width=True):
        st.info("Redirection vers le Mode 1...")
        st.markdown("""
        <script>
        window.open('http://localhost:8501', '_blank');
        </script>
        """, unsafe_allow_html=True)
        
        # Instructions pour lancer manuellement
        st.markdown("""
        ### 📋 Instructions de Lancement Manuel
        
        Ouvrez un nouveau terminal et exécutez :
        ```bash
        streamlit run app_mode1.py --server.port 8502
        ```
        
        Puis ouvrez votre navigateur à l'adresse : `http://localhost:8502`
        """)

def show_mode2_info():
    """Affiche les informations et lance le mode 2"""
    st.markdown("## 📹 Mode 2: Analyse en Temps Réel")
    
    st.markdown("""
    <div class="info-box">
        <h3>Fonctionnement du Mode 2</h3>
        <p>Ce mode permet l'analyse en temps réel via votre webcam ou caméra de téléphone. 
        Les détections sont affichées instantanément avec des statistiques live.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Avantages et inconvénients
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Avantages
        - Analyse instantanée
        - Pas de stockage de fichiers vidéo
        - Statistiques en temps réel
        - Compatible avec caméra de téléphone
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Considérations RGPD
        - Traitement continu de données biométriques
        - Historique temporaire des détections
        - Nécessité d'autorisation caméra
        - Consentement requis pour les personnes filmées
        """)
    
    # Instructions pour utiliser la caméra du téléphone
    st.markdown("### 📱 Utilisation avec Téléphone")
    
    tab1, tab2 = st.tabs(["DroidCam", "IP Webcam"])
    
    with tab1:
        st.markdown("""
        #### Option 1: DroidCam
        1. **Téléchargez DroidCam** sur votre téléphone (Android/iOS)
        2. **Installez DroidCam Client** sur votre PC
        3. **Connectez** via WiFi ou USB
        4. **Sélectionnez** DroidCam comme source vidéo dans votre navigateur
        
        [📥 Télécharger DroidCam](https://www.dev47apps.com/)
        """)
    
    with tab2:
        st.markdown("""
        #### Option 2: IP Webcam (Android)
        1. **Installez IP Webcam** depuis Google Play Store
        2. **Lancez l'application** et démarrez le serveur
        3. **Notez l'adresse IP** affichée (ex: 192.168.1.100:8080)
        4. **Utilisez cette URL** dans votre navigateur
        
        [📥 Télécharger IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)
        """)
    
    # Bouton pour lancer le mode 2
    if st.button("🚀 Lancer le Mode 2", type="primary", use_container_width=True):
        st.info("Redirection vers le Mode 2...")
        
        # Instructions pour lancer manuellement
        st.markdown("""
        ### 📋 Instructions de Lancement Manuel
        
        Ouvrez un nouveau terminal et exécutez :
        ```bash
        streamlit run app_mode2.py --server.port 8503
        ```
        
        Puis ouvrez votre navigateur à l'adresse : `http://localhost:8503`
        
        ⚠️ **Important :** Autorisez l'accès à votre caméra quand le navigateur vous le demande.
        """)

if __name__ == "__main__":
    main() 