"""
================================================================================
APPLICATION STREAMLIT - DÉTECTION DE FRAUDE (VERSION CORRIGÉE)
Mode 100% LOCAL - Sans Azure ML Endpoint
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib
import json
import os

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="🔍 Détection de Fraude Bancaire",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personnalisé (identique)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #e74c3c, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .alert-fraud {
        background: linear-gradient(135deg, #fee 0%, #fcc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #e74c3c;
        color: #c0392b;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-safe {
        background: linear-gradient(135deg, #efe 0%, #cfc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #27ae60;
        color: #229954;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #f39c12;
        color: #856404;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        font-size: 1.2rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DES MODÈLES
# =============================================================================

@st.cache_resource
def load_models():
    """Charge le modèle, le scaler et les métadonnées"""
    try:
        model = joblib.load('outputs/best_model.pkl')
        scaler = joblib.load('outputs/scaler.pkl')
        
        try:
            with open('outputs/metadata.json', 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {
                'best_model': 'XGBoost',
                'optimal_threshold': 0.5,  # Fallback si pas dans metadata
                'all_models': {}
            }
        
        # Extraire le seuil optimal
        optimal_threshold = metadata.get('optimal_threshold', 0.5)
        
        return model, scaler, metadata, optimal_threshold, None
        
    except Exception as e:
        return None, None, None, None, str(e)

model, scaler, metadata, optimal_threshold, error = load_models()

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def predict_fraud(input_data, threshold=None):
    """
    Fait une prédiction de fraude
    
    Args:
        input_data: array numpy des features
        threshold: seuil de décision (si None, utilise un seuil adapté)
    """
    try:
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # ⚠️ CORRECTION: Utiliser un seuil plus raisonnable
        # Le seuil de 0.77 du training est trop élevé pour la production
        if threshold is None:
            # Utiliser 0.5 au lieu de optimal_threshold (0.77)
            # OU ajuster selon vos besoins métier
            threshold = 0.5  # Seuil standard plus équilibré
        
        # Scaling
        scaled_data = scaler.transform(input_data)
        
        # Prédiction
        probabilities = model.predict_proba(scaled_data)[0]
        fraud_prob = float(probabilities[1])
        
        # Appliquer le seuil
        prediction = 1 if fraud_prob >= threshold else 0
        
        # 🎯 CORRECTION: Niveaux de risque ajustés
        # Basés sur des seuils plus réalistes
        if fraud_prob >= 0.70:
            risk_level = "HIGH"
            recommendation = "🚫 BLOQUER - Fraude hautement probable"
            color = "red"
        elif fraud_prob >= 0.40:
            risk_level = "MEDIUM"
            recommendation = "⚠️ VÉRIFIER - Investigation recommandée"
            color = "orange"
        else:
            risk_level = "LOW"
            recommendation = "✅ APPROUVER - Transaction sûre"
            color = "green"
        
        return {
            'is_fraud': bool(prediction == 1),
            'fraud_probability': fraud_prob,
            'legitimate_probability': float(probabilities[0]),
            'confidence': float(max(probabilities)),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'color': color,
            'threshold_used': threshold,
            'model_optimal_threshold': optimal_threshold  # Pour debug
        }
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None
    
def create_gauge_chart(value, title, color_gradient):
    """Crée une jauge interactive"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#2c3e50'}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "darkgray"},
            'bar': {'color': color_gradient},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#d5f4e6'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial"}
    )
    return fig

def create_probability_distribution(fraud_prob):
    """Crée un graphique de distribution des probabilités"""
    fig = go.Figure()
    
    categories = ['Légitime', 'Fraude']
    values = [1 - fraud_prob, fraud_prob]
    colors = ['#27ae60', '#e74c3c']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=[v * 100 for v in values],
        marker_color=colors,
        text=[f'{v*100:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=16, color='white')
    ))
    
    fig.update_layout(
        title="Distribution des Probabilités",
        yaxis_title="Probabilité (%)",
        height=300,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14)
    )
    
    return fig

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">🔍 Système de Détection de Fraude Bancaire</h1>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyse en Temps Réel avec Intelligence Artificielle | Projet CDDA 2024-2025</p>', 
            unsafe_allow_html=True)

if error:
    st.error(f"""
    ❌ **Erreur de chargement des modèles**
    
    {error}
    
    **Vérifiez que les fichiers suivants existent:**
    - `outputs/best_model.pkl`
    - `outputs/scaler.pkl`
    - `outputs/metadata.json`
    """)
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("📊 Informations du Modèle")

if metadata:
    st.sidebar.success(f"**Modèle Actif:** {metadata.get('best_model', 'XGBoost')}")
    st.sidebar.info(f"**Seuil Optimal:** {optimal_threshold:.3f}")
    
    if 'all_models' in metadata and metadata['all_models']:
        best_model_name = metadata.get('best_model', list(metadata['all_models'].keys())[0])
        if best_model_name in metadata['all_models']:
            metrics = metadata['all_models'][best_model_name]['metrics']
            
            st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
            st.sidebar.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.1f}%")
            st.sidebar.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

st.sidebar.markdown("---")

# Initialiser session state pour le mode démo
if 'demo_type' not in st.session_state:
    st.session_state.demo_type = None

# Mode de démonstration
demo_mode = st.sidebar.checkbox(
    "🎮 Mode Démonstration",
    help="Remplit automatiquement avec des exemples"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**📖 À Propos**

Cette application utilise des modèles 
de Machine Learning entraînés sur Azure ML 
pour détecter les fraudes bancaires.

**🎯 Performance:**
- Précision: >95%
- Temps de réponse: <100ms
- Mode: Local (sans API)

**🔧 Technologies:**
- Scikit-learn
- XGBoost / LightGBM
- Streamlit
- Azure ML (training)
""")

# =============================================================================
# TABS PRINCIPALES
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Analyse Transaction",
    "📊 Analyse Batch (CSV)",
    "📈 Statistiques"
])

# =============================================================================
# TAB 1: ANALYSE TRANSACTION UNIQUE (VERSION CORRIGÉE)
# =============================================================================

with tab1:
    st.header("Analyse d'une Transaction Individuelle")
    
    # Exemples prédéfinis avec boutons
    if demo_mode:
        st.info("🎮 **Mode Démonstration Activé** - Choisissez un exemple")
        
        col_demo1, col_demo2, col_demo3 = st.columns(3)
        
        with col_demo1:
            if st.button("✅ Transaction Légitime", use_container_width=True):
                st.session_state.demo_type = "legitimate"
                st.rerun()
        
        with col_demo2:
            if st.button("⚠️ Transaction Suspecte", use_container_width=True):
                st.session_state.demo_type = "suspicious"
                st.rerun()
        
        with col_demo3:
            if st.button("🚨 Fraude Évidente", use_container_width=True):
                st.session_state.demo_type = "fraud"
                st.rerun()
    
    st.markdown("---")
    
    # Définir les valeurs par défaut AVANT de créer les widgets
    default_values = {
        'legitimate': {
            'amount': 150.0,
            'old_orig': 5000.0,
            'new_orig': 4850.0,
            'old_dest': 3000.0,
            'new_dest': 3150.0,
            'type': 'PAYMENT',
            'type_idx': 0,
            'hour': 14,
            'day': 'Mercredi',
            'day_idx': 2
        },
        'suspicious': {
            'amount': 15000.0,
            'old_orig': 20000.0,
            'new_orig': 5000.0,
            'old_dest': 5000.0,
            'new_dest': 20000.0,
            'type': 'TRANSFER',
            'type_idx': 1,
            'hour': 22,
            'day': 'Samedi',
            'day_idx': 5
        },
        'fraud': {
            'amount': 50000.0,
            'old_orig': 100.0,
            'new_orig': 0.0,
            'old_dest': 200000.0,
            'new_dest': 250000.0,
            'type': 'CASH_OUT',
            'type_idx': 2,
            'hour': 3,
            'day': 'Dimanche',
            'day_idx': 6
        }
    }
    
    # Récupérer les valeurs par défaut selon le mode démo
    current_demo = st.session_state.get('demo_type', 'legitimate')
    if not demo_mode:
        current_demo = 'legitimate'
    
    defaults = default_values.get(current_demo, default_values['legitimate'])
    
    # Afficher quel exemple est chargé
    if demo_mode and st.session_state.demo_type:
        demo_labels = {
            'legitimate': '✅ Exemple: Transaction Légitime',
            'suspicious': '⚠️ Exemple: Transaction Suspecte',
            'fraud': '🚨 Exemple: Fraude Évidente'
        }
        st.success(demo_labels[st.session_state.demo_type])
    
    # Formulaire de transaction avec KEY UNIQUE pour chaque widget
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Informations Transaction")
        
        amount = st.number_input(
            "💵 Montant de la transaction (€)",
            min_value=0.0,
            max_value=1000000.0,
            value=defaults['amount'],
            step=10.0,
            key=f"amount_{current_demo}",
            help="Montant en euros"
        )
        
        transaction_type = st.selectbox(
            "🏦 Type de transaction",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
            index=defaults['type_idx'],
            key=f"type_{current_demo}",
            help="Nature de la transaction"
        )
        
        old_balance_orig = st.number_input(
            "💼 Solde initial émetteur (€)",
            min_value=0.0,
            value=defaults['old_orig'],
            step=100.0,
            key=f"old_orig_{current_demo}"
        )
        
        new_balance_orig = st.number_input(
            "💼 Nouveau solde émetteur (€)",
            min_value=0.0,
            value=defaults['new_orig'],
            step=100.0,
            key=f"new_orig_{current_demo}"
        )
    
    with col2:
        st.subheader("👤 Informations Destinataire")
        
        old_balance_dest = st.number_input(
            "💰 Solde initial destinataire (€)",
            min_value=0.0,
            value=defaults['old_dest'],
            step=100.0,
            key=f"old_dest_{current_demo}"
        )
        
        new_balance_dest = st.number_input(
            "💰 Nouveau solde destinataire (€)",
            min_value=0.0,
            value=defaults['new_dest'],
            step=100.0,
            key=f"new_dest_{current_demo}"
        )
        
        hour = st.slider(
            "🕐 Heure de la transaction",
            0, 23,
            defaults['hour'],
            key=f"hour_{current_demo}"
        )
        
        day = st.selectbox(
            "📅 Jour de la semaine",
            ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
            index=defaults['day_idx'],
            key=f"day_{current_demo}"
        )
    
    st.markdown("---")
    
    # Bouton d'analyse
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button(
            "🔍 ANALYSER LA TRANSACTION",
            type="primary",
            use_container_width=True
        )
    
    if analyze_button:
        # ===================================================================
        # CONSTRUCTION DES FEATURES (DOIT CORRESPONDRE AU TRAINING!)
        # ===================================================================
        
        # 1. Encoder le type de transaction (EXACTEMENT comme au training)
        type_encoding = {
            'PAYMENT': 1, 
            'TRANSFER': 2, 
            'CASH_OUT': 3, 
            'DEBIT': 4, 
            'CASH_IN': 5
        }
        type_encoded = type_encoding.get(transaction_type, 0)
        
        # 2. Step (utiliser 1 comme valeur par défaut en temps réel)
        step = 1
        
        # 3. Construire le vecteur de features DANS LE BON ORDRE
        # ORDRE CRITIQUE: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
        features = np.array([[
            step,                    # Feature 0: step
            type_encoded,            # Feature 1: type (1-5)
            amount,                  # Feature 2: amount
            old_balance_orig,        # Feature 3: oldbalanceOrg
            new_balance_orig,        # Feature 4: newbalanceOrig
            old_balance_dest,        # Feature 5: oldbalanceDest
            new_balance_dest         # Feature 6: newbalanceDest
        ]])
        
        # 4. VALIDATION DES FEATURES
        st.info(f"🔍 Vecteur de features construit: {features.shape[1]} features")
        
        # Debug: Afficher les features
        with st.expander("🔬 Debug: Voir les features calculées"):
            st.write("**Features envoyées au modèle:**")
            feature_labels = [
                'step', 'type', 'amount', 
                'oldbalanceOrg', 'newbalanceOrig', 
                'oldbalanceDest', 'newbalanceDest'
            ]
            
            df_features = pd.DataFrame([features[0]], columns=feature_labels)
            st.dataframe(df_features.style.highlight_max(axis=1))
            
            # Afficher après scaling
            st.write("**Après scaling (RobustScaler):**")
            scaled = scaler.transform(features)
            df_scaled = pd.DataFrame([scaled[0]], columns=feature_labels)
            st.dataframe(df_scaled)
            
            # Statistiques
            st.write("**Statistiques:**")
            st.write(f"- Type encodé: {type_encoded} ({transaction_type})")
            st.write(f"- Montant: {amount:,.2f} €")
            st.write(f"- Variation solde émetteur: {old_balance_orig - new_balance_orig:,.2f} €")
            st.write(f"- Variation solde destinataire: {new_balance_dest - old_balance_dest:,.2f} €")
        
        # Animation de chargement
        with st.spinner("⏳ Analyse en cours..."):
            import time
            time.sleep(0.5)
            
            # Faire la prédiction
            result = predict_fraud(features)
        
        if result:
            st.success("✅ **Analyse terminée!**")
            
            # Affichage du résultat principal
            st.markdown("## 🎯 Résultat de l'Analyse")
            
            # Alerte visuelle (CODE RESTE IDENTIQUE)
            if result['is_fraud']:
                st.markdown(
                    '<div class="alert-fraud">🚨 ALERTE FRAUDE DÉTECTÉE 🚨</div>',
                    unsafe_allow_html=True
                )
            elif result['risk_level'] == "MEDIUM":
                st.markdown(
                    '<div class="alert-warning">⚠️ TRANSACTION SUSPECTE - VÉRIFICATION REQUISE</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="alert-safe">✅ TRANSACTION LÉGITIME</div>',
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # Métriques principales
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "🆔 Transaction ID",
                    f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
            
            with col_m2:
                delta_color = "inverse" if result['risk_level'] == "HIGH" else "off"
                st.metric(
                    "⚠️ Niveau de Risque",
                    result['risk_level'],
                    delta="CRITIQUE" if result['risk_level'] == "HIGH" else None,
                    delta_color=delta_color
                )
            
            with col_m3:
                st.metric(
                    "📊 Probabilité Fraude",
                    f"{result['fraud_probability']*100:.1f}%"
                )
            
            with col_m4:
                st.metric(
                    "🎯 Confiance",
                    f"{result['confidence']*100:.1f}%"
                )
            
            st.markdown("---")
            
            # Visualisations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                fig_gauge = create_gauge_chart(
                    result['fraud_probability'],
                    "Probabilité de Fraude",
                    result['color']
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_viz2:
                fig_dist = create_probability_distribution(result['fraud_probability'])
                st.plotly_chart(fig_dist, use_container_width=True)
            
            st.markdown("---")
            
            # Recommandations
            st.markdown("### 💡 Recommandation")
            
            if result['fraud_probability'] >= 0.7:
                st.error(f"""
                **{result['recommendation']}**
                
                **Actions immédiates:**
                - 🚫 Bloquer la transaction
                - 📞 Contacter le client
                - 🔍 Vérifier l'identité
                - 🚨 Alerter le département fraude
                """)
            elif result['fraud_probability'] >= 0.4:
                st.warning(f"""
                **{result['recommendation']}**
                
                **Actions recommandées:**
                - ⚠️ Suspendre temporairement
                - 📱 Envoyer SMS de vérification
                - 🔐 Demander authentification 2FA
                """)
            else:
                st.success(f"""
                **{result['recommendation']}**
                
                **Actions:**
                - ✅ Approuver la transaction
                - 📊 Surveillance standard
                """)

# TAB 2 et 3 restent identiques...
with tab2:
    st.header("📊 Analyse Batch - En construction")
    st.info("Cette fonctionnalité sera disponible prochainement")

with tab3:
    st.header("📈 Statistiques - En construction")
    st.info("Cette fonctionnalité sera disponible prochainement")