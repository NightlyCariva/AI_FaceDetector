import streamlit as st
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title=os.getenv('APP_TITLE', 'Analyseur de Visages IA'),
    page_icon=os.getenv('APP_ICON', '🎭'),
    layout="wide"
)

def main():
    """Page d'accueil principale"""
    
    # Titre principal
    st.title("Analyseur de Visages IA")
    st.markdown("---")
    
    # Zone d'avertissement
    st.error("""
    **⚠️ AVERTISSEMENT IMPORTANT**
    
    Ce projet a été développé dans le cadre d'un cours sur la **Protection des Données** à l'UPJV 
    pour illustrer les défis et enjeux liés au traitement de données biométriques par l'Intelligence Artificielle. 
    Il démontre concrètement le potentiel de l'IA et les raisons de l'implications du RGPD dans le développement 
    d'applications d'IA.
    """)
    
    # Radio button de consentement
    consent = st.radio(
        "Consentement d'utilisation:",
        ["Je n'ai pas pris connaissance", 
         "J'ai pris connaissance de la finalité du projet et j'atteste ne l'utiliser que dans le cadre pédagogique autorisé"],
        index=0
    )
    
    st.markdown("---")
    
    # Grille des fonctionnalités
    st.header("Fonctionnalités")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("""
        **Détection de Visages**
        
        Identification automatique des visages dans les images et vidéos
        """)
    
    with col2:
        st.info("""
        **Estimation d'Age**
        
        Prédiction de la tranche d'âge des personnes détectées
        """)
    
    with col3:
        st.info("""
        **Classification Genre**
        
        Identification du genre apparent des individus
        """)
    
    with col4:
        st.info("""
        **Analyse Emotions**
        
        Reconnaissance des expressions faciales et émotions
        """)
    
    st.markdown("---")
    
    # Modes disponibles
    st.header("Modes Disponibles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mode 1: Upload Vidéo")
        st.write("""
        Analysez des vidéos pré-enregistrées en uploadant vos fichiers. 
        Idéal pour l'analyse complète et détaillée de contenus vidéo existants.
        
        - Upload de vidéos (MP4, AVI, MOV, MKV)
        - Analyse complète frame par frame
        - Export des résultats en CSV
        - Téléchargement de la vidéo annotée
        """)
        
        if st.button("Accéder au Mode 1", key="mode1"):
            if consent.startswith("J'ai pris connaissance"):
                st.session_state.page = "mode1"
                st.rerun()
            else:
                st.error("Vous devez prendre connaissance de la finalité du projet avant d'accéder aux modes d'analyse.")
    
    with col2:
        st.subheader("Mode 2: Temps Réel")
        st.write("""
        Analysez en temps réel via votre webcam pour des démonstrations 
        interactives et des tests instantanés.
        
        - Analyse en temps réel via webcam
        - Détection instantanée des visages
        - Statistiques live
        - Export des sessions en CSV
        """)
        
        if st.button("Accéder au Mode 2", key="mode2"):
            if consent.startswith("J'ai pris connaissance"):
                st.session_state.page = "mode2"
                st.rerun()
            else:
                st.error("Vous devez prendre connaissance de la finalité du projet avant d'accéder aux modes d'analyse.")

if __name__ == "__main__":
    # Initialisation de la session
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Navigation entre les pages
    if st.session_state.page == "home":
        main()
    elif st.session_state.page == "mode1":
        # Bouton retour
        if st.button("← Retour à l'accueil"):
            st.session_state.page = "home"
            st.rerun()
        
        # Importer et exécuter le mode 1
        from mode1_upload import run_mode1
        run_mode1()
        
    elif st.session_state.page == "mode2":
        # Bouton retour
        if st.button("← Retour à l'accueil"):
            st.session_state.page = "home"
            st.rerun()
        
        # Importer et exécuter le mode 2
        from mode2_realtime import run_mode2
        run_mode2() 