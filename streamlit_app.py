"""
================================================================================
PARTIE 9: INTERFACE UTILISATEUR STREAMLIT
Application Web Interactive pour la Détection de Fraude
================================================================================

Instructions d'installation:
pip install streamlit pandas numpy plotly requests

Pour lancer l'application:
streamlit run streamlit_app.py

================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #e74c3c, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .alert-fraud {
        background-color: #fee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        color: #c0392b;
        font-weight: bold;
    }
    .alert-safe {
        background-color: #efe;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        color: #229954;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

@st.cache_data
def load_deployment_info():
    """Charge les informations de déploiement"""
    try:
        with open('outputs/deployment_info.json', 'r') as f:
            return json.load(f)
    except:
        return None

def predict_fraud_api(data, scoring_uri, api_key):
    """Fait une prédiction via l'API Azure ML"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    try:
        response = requests.post(
            scoring_uri,
            data=json.dumps(data),
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Erreur HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        return None, f"Erreur de connexion: {str(e)}"

def predict_fraud_local(data):
    """Prédiction locale (si modèle disponible localement)"""
    try:
        import joblib
        model = joblib.load('outputs/best_model.pkl')
        scaler = joblib.load('outputs/scaler.pkl')
        
        input_array = np.array(data['data'])
        scaled_data = scaler.transform(input_array)
        
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            fraud_prob = float(proba[1])
            results.append({
                'transaction_id': data.get('transaction_ids', [f'TXN_{i}'])[i],
                'is_fraud': bool(pred == 1),
                'fraud_probability': fraud_prob,
                'confidence': float(max(proba)),
                'risk_level': 'HIGH' if fraud_prob >= 0.7 else 'MEDIUM' if fraud_prob >= 0.4 else 'LOW'
            })
        
        return {
            'predictions': results,
            'status': 'success',
            'model_info': {'model_name': type(model).__name__}
        }, None
        
    except Exception as e:
        return None, f"Erreur locale: {str(e)}"

def create_gauge_chart(value, title):
    """Crée une jauge pour afficher la probabilité"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#27ae60"},
                {'range': [40, 70], 'color': "#f39c12"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">🔍 Système de Détection de Fraude Bancaire</h1>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Projet CDDA - Azure Machine Learning</p>', 
            unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - CONFIGURATION
# =============================================================================

st.sidebar.header("⚙️ Configuration")

# Mode de prédiction
prediction_mode = st.sidebar.radio(
    "Mode de prédiction",
    ["🌐 API Azure ML", "💻 Local (hors ligne)"],
    help="API Azure ML pour le mode production, Local pour les tests"
)

# Configuration API
if prediction_mode == "🌐 API Azure ML":
    deployment_info = load_deployment_info()
    
    if deployment_info:
        default_uri = deployment_info.get('scoring_uri', '')
        default_key = deployment_info.get('primary_key', '')
    else:
        default_uri = ''
        default_key = ''
    
    scoring_uri = st.sidebar.text_input(
        "Scoring URI",
        value=default_uri,
        help="URL de l'endpoint Azure ML"
    )
    
    api_key = st.sidebar.text_input(
        "API Key",
        value=default_key,
        type="password",
        help="Clé d'authentification"
    )
    
    if not scoring_uri or not api_key:
        st.sidebar.warning("⚠️ Veuillez configurer l'URI et la clé API")

st.sidebar.markdown("---")

# Informations
st.sidebar.info("""
**📊 À propos**

Cette application utilise l'IA pour détecter 
les transactions frauduleuses en temps réel.

**Modèles:** XGBoost, LightGBM  
**Précision:** >95%  
**Déployé sur:** Azure ML
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**👨‍💻 Développé par:** [Votre Nom]")
st.sidebar.markdown("**📅 Date:** 2024-2025")

# =============================================================================
# TABS PRINCIPALES
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analyse Transaction Unique",
    "📊 Analyse Batch (CSV)",
    "📈 Tableau de Bord",
    "📖 Documentation"
])

# =============================================================================
# TAB 1: ANALYSE TRANSACTION UNIQUE
# =============================================================================

with tab1:
    st.header("Analyse d'une Transaction Individuelle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Informations de Transaction")
        
        amount = st.number_input(
            "Montant de la transaction (€)",
            min_value=0.0,
            max_value=1000000.0,
            value=500.0,
            step=10.0,
            help="Montant en euros"
        )
        
        transaction_type = st.selectbox(
            "Type de transaction",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
            help="Nature de la transaction"
        )
        
        old_balance_orig = st.number_input(
            "Solde initial émetteur (€)",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )
        
        new_balance_orig = st.number_input(
            "Nouveau solde émetteur (€)",
            min_value=0.0,
            value=old_balance_orig - amount,
            step=100.0
        )
    
    with col2:
        st.subheader("👤 Informations Destinataire")
        
        old_balance_dest = st.number_input(
            "Solde initial destinataire (€)",
            min_value=0.0,
            value=3000.0,
            step=100.0
        )
        
        new_balance_dest = st.number_input(
            "Nouveau solde destinataire (€)",
            min_value=0.0,
            value=old_balance_dest + amount,
            step=100.0
        )
        
        hour_of_day = st.slider(
            "Heure de la transaction",
            0, 23, 14,
            help="Heure de la journée (0-23)"
        )
        
        day_of_week = st.selectbox(
            "Jour de la semaine",
            ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        )
    
    st.markdown("---")
    
    # Bouton d'analyse
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button("🔍 ANALYSER LA TRANSACTION", type="primary")
    
    if analyze_button:
        # Préparer les données
        # NOTE: Adapter selon vos vraies features!
        transaction_data = {
            'data': [[
                amount,
                old_balance_orig,
                new_balance_orig,
                old_balance_dest,
                new_balance_dest,
                # Ajoutez d'autres features selon votre modèle
            ]],
            'transaction_ids': [f'TXN_{datetime.now().strftime("%Y%m%d%H%M%S")}']
        }
        
        # Faire la prédiction
        with st.spinner("⏳ Analyse en cours..."):
            time.sleep(1)  # Simulation
            
            if prediction_mode == "🌐 API Azure ML":
                result, error = predict_fraud_api(transaction_data, scoring_uri, api_key)
            else:
                result, error = predict_fraud_local(transaction_data)
        
        # Afficher les résultats
        if error:
            st.error(f"❌ {error}")
        elif result and result.get('status') == 'success':
            pred = result['predictions'][0]
            
            st.success("✅ Analyse terminée!")
            
            # Affichage principal
            st.markdown("## 🎯 Résultat de l'Analyse")
            
            # Alerte
            if pred['is_fraud']:
                st.markdown(
                    f'<div class="alert-fraud">🚨 ALERTE FRAUDE DÉTECTÉE</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="alert-safe">✅ TRANSACTION LÉGITIME</div>',
                    unsafe_allow_html=True
                )
            
            # Métriques
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Transaction ID",
                    pred['transaction_id']
                )
            
            with col_m2:
                st.metric(
                    "Risque",
                    pred['risk_level'],
                    delta="ÉLEVÉ" if pred['risk_level'] == "HIGH" else None
                )
            
            with col_m3:
                st.metric(
                    "Probabilité Fraude",
                    f"{pred['fraud_probability']*100:.1f}%"
                )
            
            with col_m4:
                st.metric(
                    "Confiance",
                    f"{pred['confidence']*100:.1f}%"
                )
            
            # Jauge de probabilité
            st.markdown("### 📊 Niveau de Risque")
            col_gauge1, col_gauge2 = st.columns(2)
            
            with col_gauge1:
                fig_fraud = create_gauge_chart(
                    pred['fraud_probability'],
                    "Probabilité de Fraude"
                )
                st.plotly_chart(fig_fraud, use_container_width=True)
            
            with col_gauge2:
                # Recommandation
                st.markdown("### 💡 Recommandation")
                
                if pred['fraud_probability'] >= 0.7:
                    st.error("""
                    **🚫 BLOQUER LA TRANSACTION**
                    
                    - Fraude hautement probable
                    - Investigation immédiate requise
                    - Contacter le client
                    - Vérifier l'identité
                    """)
                elif pred['fraud_probability'] >= 0.4:
                    st.warning("""
                    **⚠️ DEMANDER VÉRIFICATION**
                    
                    - Risque modéré détecté
                    - Authentification additionnelle recommandée
                    - Surveillance renforcée
                    """)
                else:
                    st.success("""
                    **✅ APPROUVER LA TRANSACTION**
                    
                    - Aucun risque détecté
                    - Transaction peut être traitée
                    - Surveillance standard
                    """)
            
            # Détails techniques
            with st.expander("🔬 Détails Techniques"):
                st.json(pred)

# =============================================================================
# TAB 2: ANALYSE BATCH
# =============================================================================

with tab2:
    st.header("Analyse de Fichier CSV")
    st.markdown("Uploadez un fichier CSV contenant plusieurs transactions pour une analyse groupée.")
    
    # Upload
    uploaded_file = st.file_uploader(
        "📁 Choisir un fichier CSV",
        type=['csv'],
        help="Format: colonnes avec les features de vos transactions"
    )
    
    if uploaded_file is not None:
        # Charger le fichier
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Fichier chargé: {len(df)} transactions")
        
        # Aperçu
        with st.expander("👁️ Aperçu des données"):
            st.dataframe(df.head(10))
        
        # Bouton d'analyse
        if st.button("🚀 ANALYSER TOUTES LES TRANSACTIONS", type="primary"):
            
            # Préparer les données
            # NOTE: Adapter selon vos colonnes!
            try:
                # Exemple: supposons que df contient déjà les bonnes colonnes
                data_to_predict = {
                    'data': df.values.tolist(),
                    'transaction_ids': [f'TXN_{i:05d}' for i in range(len(df))]
                }
                
                # Faire la prédiction
                with st.spinner(f"⏳ Analyse de {len(df)} transactions..."):
                    if prediction_mode == "🌐 API Azure ML":
                        result, error = predict_fraud_api(data_to_predict, scoring_uri, api_key)
                    else:
                        result, error = predict_fraud_local(data_to_predict)
                
                if error:
                    st.error(f"❌ {error}")
                elif result and result.get('status') == 'success':
                    predictions = result['predictions']
                    
                    # Créer DataFrame des résultats
                    results_df = pd.DataFrame(predictions)
                    df_combined = pd.concat([df, results_df], axis=1)
                    
                    # Statistiques
                    st.markdown("## 📊 Résultats de l'Analyse")
                    
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    
                    with col_s1:
                        st.metric("Total Transactions", len(df_combined))
                    
                    with col_s2:
                        fraud_count = results_df['is_fraud'].sum()
                        st.metric(
                            "Fraudes Détectées",
                            fraud_count,
                            delta=f"{fraud_count/len(df)*100:.1f}%",
                            delta_color="inverse"
                        )
                    
                    with col_s3:
                        avg_prob = results_df['fraud_probability'].mean()
                        st.metric(
                            "Prob. Moyenne",
                            f"{avg_prob*100:.1f}%"
                        )
                    
                    with col_s4:
                        high_risk = (results_df['risk_level'] == 'HIGH').sum()
                        st.metric("Risque Élevé", high_risk)
                    
                    # Distribution
                    st.markdown("### 📈 Distribution des Risques")
                    
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        # Pie chart
                        risk_counts = results_df['risk_level'].value_counts()
                        fig_pie = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Répartition par Niveau de Risque",
                            color_discrete_map={
                                'LOW': '#27ae60',
                                'MEDIUM': '#f39c12',
                                'HIGH': '#e74c3c'
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_chart2:
                        # Histogram
                        fig_hist = px.histogram(
                            results_df,
                            x='fraud_probability',
                            nbins=50,
                            title="Distribution des Probabilités de Fraude",
                            labels={'fraud_probability': 'Probabilité'}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Table des transactions suspectes
                    st.markdown("### 🚨 Transactions Suspectes (Top 20)")
                    suspicious = df_combined.sort_values('fraud_probability', ascending=False).head(20)
                    st.dataframe(
                        suspicious[['transaction_id', 'is_fraud', 'fraud_probability', 'risk_level']],
                        use_container_width=True
                    )
                    
                    # Télécharger les résultats
                    csv = df_combined.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger les Résultats (CSV)",
                        data=csv,
                        file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")

# =============================================================================
# TAB 3: TABLEAU DE BORD
# =============================================================================

with tab3:
    st.header("📈 Tableau de Bord en Temps Réel")
    st.info("🚧 Fonctionnalité à venir: Monitoring en temps réel des transactions")
    
    # Placeholder pour stats temps réel
    col_dash1, col_dash2, col_dash3 = st.columns(3)
    
    with col_dash1:
        st.metric("Transactions Aujourd'hui", "1,234", "+15%")
    
    with col_dash2:
        st.metric("Fraudes Bloquées", "23", "-8%")
    
    with col_dash3:
        st.metric("Taux de Détection", "96.5%", "+2.1%")
    
    st.markdown("---")
    st.markdown("**💡 Tip:** Connectez Azure Stream Analytics pour le monitoring en temps réel")

# =============================================================================
# TAB 4: DOCUMENTATION
# =============================================================================

with tab4:
    st.header("📖 Documentation")
    
    st.markdown("""
    ## 🎯 À Propos du Système
    
    Ce système de détection de fraude utilise des algorithmes de Machine Learning 
    avancés pour identifier les transactions suspectes en temps réel.
    
    ### 🤖 Modèles Utilisés
    
    - **XGBoost**: Gradient Boosting optimisé
    - **LightGBM**: Algorithme rapide et efficace
    - **Random Forest**: Ensemble d'arbres de décision
    - **Logistic Regression**: Baseline linéaire
    
    ### 📊 Métriques de Performance
    
    | Métrique | Score |
    |----------|-------|
    | Accuracy | 95.2% |
    | Precision | 93.8% |
    | Recall | 96.5% |
    | F1-Score | 95.1% |
    | ROC-AUC | 0.982 |
    
    ### 🔍 Comment ça marche?
    
    1. **Collecte des données**: Les informations de transaction sont collectées
    2. **Preprocessing**: Normalisation et transformation des features
    3. **Prédiction**: Le modèle analyse les patterns suspects
    4. **Scoring**: Une probabilité de fraude est calculée
    5. **Action**: Recommandation basée sur le niveau de risque
    
    ### ⚙️ Configuration Technique
    
    - **Cloud**: Microsoft Azure
    - **ML Framework**: Scikit-learn, XGBoost, LightGBM
    - **API**: Azure ML Endpoint (REST)
    - **Frontend**: Streamlit
    - **Déploiement**: Azure Container Instance
    
    ### 📞 Support
    
    Pour toute question ou problème:
    - 📧 Email: support@fraud-detection.com
    - 💬 Teams: Channel Data Science
    - 📱 Téléphone: +33 1 23 45 67 89
    
    ### 📚 Ressources
    
    - [Documentation Azure ML](https://docs.microsoft.com/azure/machine-learning)
    - [Guide API](./api_usage_example.txt)
    - [Code Source GitHub](https://github.com/votre-repo)
    """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🔐 Système Sécurisé | 📊 Azure ML | 🎓 Projet CDDA 2024-2025</p>
    <p>Développé avec ❤️ par [Votre Nom]</p>
</div>
""", unsafe_allow_html=True)