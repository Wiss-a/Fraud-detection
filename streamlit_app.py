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
    st.sidebar.info(f"**Seuil Optimal1111:** {optimal_threshold:.3f}")
    
    if 'all_models' in metadata and metadata['all_models']:
        best_model_name = metadata.get('best_model', list(metadata['all_models'].keys())[0])
        if best_model_name in metadata['all_models']:
            metrics = metadata['all_models'][best_model_name]['metrics']
            
            st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
            st.sidebar.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.1f}%")
            st.sidebar.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

st.sidebar.markdown("---")
# À ajouter dans la SIDEBAR (après les métriques du modèle)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuration")

# Sélecteur de seuil
st.sidebar.markdown("### 🎯 Seuil de Décision")

threshold_option = st.sidebar.radio(
    "Choisir le mode de seuil:",
    ["Standard (0.5)", "Optimal Training (0.77)", "Personnalisé"],
    help="Le seuil détermine à partir de quelle probabilité une transaction est classée comme fraude"
)

if threshold_option == "Standard (0.5)":
    custom_threshold = 0.5
    st.sidebar.info("✅ Seuil équilibré recommandé")
elif threshold_option == "Optimal Training (0.77)":
    custom_threshold = optimal_threshold
    st.sidebar.warning("⚠️ Seuil très élevé - Peut manquer des fraudes")
else:
    custom_threshold = st.sidebar.slider(
        "Seuil personnalisé:",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Plus le seuil est élevé, moins il y aura de fausses alertes, mais plus de fraudes manquées"
    )
    
    # Indicateurs visuels
    if custom_threshold < 0.3:
        st.sidebar.error("🚨 Très sensible - Beaucoup de fausses alertes")
    elif custom_threshold < 0.5:
        st.sidebar.warning("⚠️ Sensible - Plus d'alertes")
    elif custom_threshold < 0.7:
        st.sidebar.success("✅ Équilibré - Recommandé")
    else:
        st.sidebar.warning("⚠️ Strict - Risque de manquer des fraudes")

st.sidebar.markdown("---")

# Afficher les explications
with st.sidebar.expander("📖 Comprendre le seuil"):
    st.write("""
    **Seuil de décision:**
    
    - **0.5 (Standard)**: Équilibre entre détection et fausses alertes
    - **0.77 (Optimal Training)**: Optimisé pour maximiser F1-score sur données d'entraînement, mais peut être trop strict en production
    - **Personnalisé**: Ajustez selon vos besoins métier
    
    **Impact:**
    - ⬇️ Seuil bas → Détecte plus de fraudes, mais plus de fausses alertes
    - ⬆️ Seuil haut → Moins de fausses alertes, mais risque de manquer des fraudes
    """)
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
        st.markdown("---")
        st.markdown("## 🔬 DIAGNOSTIC COMPLET")
        
        # ===================================================================
        # 1. CONSTRUCTION DES FEATURES
        # ===================================================================
        st.subheader("1️⃣ Construction du Vecteur de Features")
        
        # Encoder le type
        type_encoding = {
            'PAYMENT': 1, 
            'TRANSFER': 2, 
            'CASH_OUT': 3, 
            'DEBIT': 4, 
            'CASH_IN': 5
        }
        type_encoded = type_encoding.get(transaction_type, 0)
        
        # Construire features
        features = np.array([[
            1,                      # step
            type_encoded,           # type
            amount,                 # amount
            old_balance_orig,       # oldbalanceOrg
            new_balance_orig,       # newbalanceOrig
            old_balance_dest,       # oldbalanceDest
            new_balance_dest        # newbalanceDest
        ]])
        
        # Afficher les features BRUTES
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Features BRUTES:**")
            df_raw = pd.DataFrame({
                'Feature': ['step', 'type', 'amount', 'oldbalanceOrg', 
                           'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'],
                'Valeur': features[0]
            })
            st.dataframe(df_raw, use_container_width=True)
        
        with col2:
            st.write("**Informations:**")
            st.metric("Type Transaction", f"{transaction_type} (code: {type_encoded})")
            st.metric("Montant", f"{amount:,.2f} €")
            st.metric("Δ Solde Émetteur", f"{old_balance_orig - new_balance_orig:,.2f} €")
            st.metric("Δ Solde Destinataire", f"{new_balance_dest - old_balance_dest:,.2f} €")
        
        # ===================================================================
        # 2. SCALING
        # ===================================================================
        st.markdown("---")
        st.subheader("2️⃣ Application du Scaling")
        
        try:
            scaled_data = scaler.transform(features)
            st.success("✅ Scaling appliqué avec succès")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Features APRÈS Scaling:**")
                df_scaled = pd.DataFrame({
                    'Feature': ['step', 'type', 'amount', 'oldbalanceOrg', 
                               'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'],
                    'Valeur Scalée': scaled_data[0]
                })
                st.dataframe(df_scaled, use_container_width=True)
            
            with col2:
                st.write("**Statistiques du Scaling:**")
                st.write(f"Min: {scaled_data[0].min():.4f}")
                st.write(f"Max: {scaled_data[0].max():.4f}")
                st.write(f"Mean: {scaled_data[0].mean():.4f}")
                st.write(f"Std: {scaled_data[0].std():.4f}")
                
        except Exception as e:
            st.error(f"❌ Erreur lors du scaling: {str(e)}")
            st.stop()
        
        # ===================================================================
        # 3. PRÉDICTION BRUTE
        # ===================================================================
        st.markdown("---")
        st.subheader("3️⃣ Prédiction du Modèle")
        
        try:
            # Probabilités
            probabilities = model.predict_proba(scaled_data)[0]
            fraud_prob = float(probabilities[1])
            legit_prob = float(probabilities[0])
            
            # Prédiction binaire avec différents seuils
            pred_050 = 1 if fraud_prob >= 0.50 else 0
            pred_077 = 1 if fraud_prob >= 0.77 else 0
            pred_030 = 1 if fraud_prob >= 0.30 else 0
            
            st.success("✅ Prédiction réussie")
            
            # Affichage des probabilités
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Probabilité FRAUDE",
                    f"{fraud_prob*100:.2f}%",
                    delta=f"{(fraud_prob - 0.5)*100:+.1f}% vs seuil 0.5"
                )
            
            with col2:
                st.metric(
                    "Probabilité LÉGITIME",
                    f"{legit_prob*100:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Confiance",
                    f"{max(probabilities)*100:.2f}%"
                )
            
            # Tableau de décision selon les seuils
            st.write("**Décision selon différents seuils:**")
            decision_df = pd.DataFrame({
                'Seuil': ['0.30 (Sensible)', '0.50 (Standard)', '0.77 (Training Optimal)'],
                'Probabilité Fraude': [f"{fraud_prob*100:.2f}%"] * 3,
                'Décision': [
                    '🚨 FRAUDE' if pred_030 == 1 else '✅ LÉGITIME',
                    '🚨 FRAUDE' if pred_050 == 1 else '✅ LÉGITIME',
                    '🚨 FRAUDE' if pred_077 == 1 else '✅ LÉGITIME'
                ],
                'Dépasse Seuil?': [
                    '✅ OUI' if fraud_prob >= 0.30 else '❌ NON',
                    '✅ OUI' if fraud_prob >= 0.50 else '❌ NON',
                    '✅ OUI' if fraud_prob >= 0.77 else '❌ NON'
                ]
            })
            st.dataframe(decision_df, use_container_width=True)
            
            # ⚠️ ALERTE SI PROBABILITÉ ÉLEVÉE MAIS PAS DÉTECTÉE
            if fraud_prob >= 0.60 and pred_050 == 0:
                st.error("""
                ⚠️ **INCOHÉRENCE DÉTECTÉE!**
                
                La probabilité de fraude est élevée ({:.1f}%) mais la transaction 
                n'est pas classée comme fraude avec le seuil standard de 0.5.
                
                **Cela ne devrait PAS arriver!**
                """.format(fraud_prob*100))
            
            # ===================================================================
            # 4. ANALYSE DES FEATURES IMPORTANTES
            # ===================================================================
            st.markdown("---")
            st.subheader("4️⃣ Analyse des Features")
            
            # Vérifier si le modèle a feature_importances_
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 
                               'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances,
                    'Valeur Brute': features[0],
                    'Valeur Scalée': scaled_data[0]
                }).sort_values('Importance', ascending=False)
                
                st.write("**Importance des Features (selon le modèle):**")
                st.dataframe(importance_df, use_container_width=True)
                
                # Graphique
                fig = px.bar(
                    importance_df, 
                    x='Feature', 
                    y='Importance',
                    title='Importance des Features dans le Modèle'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ===================================================================
            # 5. VÉRIFICATIONS DE COHÉRENCE
            # ===================================================================
            st.markdown("---")
            st.subheader("5️⃣ Vérifications de Cohérence")
            
            checks = []
            
            # Check 1: Cohérence des soldes
            delta_orig = old_balance_orig - new_balance_orig
            if abs(delta_orig - amount) > 0.01:
                checks.append({
                    'Check': 'Cohérence Solde Émetteur',
                    'Status': '⚠️ INCOHÉRENT',
                    'Détail': f'Δ solde ({delta_orig:.2f}) ≠ montant ({amount:.2f})'
                })
            else:
                checks.append({
                    'Check': 'Cohérence Solde Émetteur',
                    'Status': '✅ OK',
                    'Détail': f'Δ solde = montant'
                })
            
            # Check 2: Soldes négatifs
            if new_balance_orig < 0 or new_balance_dest < 0:
                checks.append({
                    'Check': 'Soldes Positifs',
                    'Status': '⚠️ SOLDE NÉGATIF',
                    'Détail': 'Un solde est négatif (suspect)'
                })
            else:
                checks.append({
                    'Check': 'Soldes Positifs',
                    'Status': '✅ OK',
                    'Détail': 'Tous les soldes sont positifs'
                })
            
            # Check 3: Transaction suspecte
            if amount > old_balance_orig * 1.5:
                checks.append({
                    'Check': 'Montant vs Solde',
                    'Status': '⚠️ SUSPECT',
                    'Détail': f'Montant ({amount:.0f}€) > 150% du solde initial'
                })
            else:
                checks.append({
                    'Check': 'Montant vs Solde',
                    'Status': '✅ OK',
                    'Détail': 'Montant cohérent avec le solde'
                })
            
            # Check 4: Type de transaction
            if transaction_type in ['CASH_OUT', 'TRANSFER'] and amount > 10000:
                checks.append({
                    'Check': 'Type & Montant',
                    'Status': '⚠️ RISQUE ÉLEVÉ',
                    'Détail': f'{transaction_type} de {amount:,.0f}€ (suspect)'
                })
            else:
                checks.append({
                    'Check': 'Type & Montant',
                    'Status': '✅ OK',
                    'Détail': 'Combinaison normale'
                })
            
            checks_df = pd.DataFrame(checks)
            st.dataframe(checks_df, use_container_width=True)
            
            # ===================================================================
            # 6. RÉSULTAT FINAL
            # ===================================================================
            st.markdown("---")
            st.markdown("## 🎯 RÉSULTAT FINAL")
            
            # Utiliser seuil 0.5
            final_decision = 1 if fraud_prob >= 0.5 else 0
            
            if final_decision == 1:
                st.markdown(
                    '<div class="alert-fraud">🚨 ALERTE FRAUDE DÉTECTÉE 🚨</div>',
                    unsafe_allow_html=True
                )
            else:
                if fraud_prob >= 0.3:
                    st.markdown(
                        '<div class="alert-warning">⚠️ TRANSACTION SUSPECTE</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="alert-safe">✅ TRANSACTION LÉGITIME</div>',
                        unsafe_allow_html=True
                    )
            
            # Recommandation
            st.markdown("### 💡 Recommandation")
            if fraud_prob >= 0.7:
                st.error("""
                **🚫 BLOQUER LA TRANSACTION**
                
                **Actions immédiates:**
                - Bloquer la transaction
                - Contacter le client immédiatement
                - Vérifier l'identité
                - Alerter le département fraude
                """)
            elif fraud_prob >= 0.4:
                st.warning("""
                **⚠️ SUSPENDRE ET VÉRIFIER**
                
                **Actions recommandées:**
                - Suspendre temporairement
                - Envoyer SMS de vérification
                - Demander authentification 2FA
                """)
            else:
                st.success("""
                **✅ APPROUVER**
                
                Transaction sûre - Surveillance standard
                """)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            st.exception(e)

