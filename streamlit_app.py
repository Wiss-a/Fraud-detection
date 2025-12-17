"""
================================================================================
APPLICATION STREAMLIT - DÉTECTION DE FRAUDE (VERSION FINALE CORRIGÉE)
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

st.set_page_config(
    page_title="🔍 Détection de Fraude Bancaire",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
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
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DES MODÈLES
# =============================================================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('outputs/best_model.pkl')
        scaler = joblib.load('outputs/scaler.pkl')
        try:
            with open('outputs/metadata.json', 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {
                'best_model': 'XGBoost',
                'optimal_threshold': 0.5,
                'all_models': {}
            }
        
        optimal_threshold = metadata.get('optimal_threshold', 0.5)
        return model, scaler, metadata, optimal_threshold, None
        
    except Exception as e:
        return None, None, None, None, str(e)

model, scaler, metadata, optimal_threshold, error = load_models()

if error:
    st.error(f"❌ Erreur: {error}")
    st.stop()

# =============================================================================
# FONCTIONS
# =============================================================================

def create_gauge_chart(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': '#d5f4e6'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 70}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=80, b=20))
    return fig

def predict_fraud_with_rules(features_raw, threshold=None):
    """
    Prédiction avec règles métier ET ML
    
    Args:
        features_raw: [amount, old_orig, new_orig, old_dest, new_dest, 
                      balance_change_orig, balance_change_dest, type_encoded, 
                      hour_normalized, day_encoded]
        threshold: seuil de décision ML
    
    Returns:
        dict avec résultats
    """
    if threshold is None:
        threshold = optimal_threshold
    
    # Extraire les valeurs
    amount = features_raw[0][0]
    old_orig = features_raw[0][1]
    new_orig = features_raw[0][2]
    old_dest = features_raw[0][3]
    new_dest = features_raw[0][4]
    balance_change_orig = features_raw[0][5]
    balance_change_dest = features_raw[0][6]
    type_encoded = features_raw[0][7]
    
    # =========================================================================
    # RÈGLES MÉTIER (PRIORITAIRES)
    # =========================================================================
    
    rule_triggered = False
    rule_reason = []
    
    # Règle 1: Incohérence des soldes
    if abs(balance_change_orig - amount) > 0.01:
        rule_triggered = True
        rule_reason.append(f"Incohérence: Δ solde émetteur ({balance_change_orig:.2f}€) ≠ montant ({amount:.2f}€)")
    
    # Règle 2: Montant énorme vs solde
    ratio_amount_orig = amount / (old_orig + 1e-5)
    if ratio_amount_orig > 10:
        rule_triggered = True
        rule_reason.append(f"Montant ({amount:,.0f}€) >> solde initial ({old_orig:,.0f}€) - Ratio: {ratio_amount_orig:.1f}x")
    
    # Règle 3: CASH_OUT suspect
    if type_encoded == 3 and amount > 10000 and old_orig < amount * 0.5:
        rule_triggered = True
        rule_reason.append(f"CASH_OUT de {amount:,.0f}€ avec solde insuffisant ({old_orig:,.0f}€)")
    
    # Règle 4: Solde négatif
    if new_orig < 0 or new_dest < 0:
        rule_triggered = True
        rule_reason.append("Solde négatif détecté")
    
    # =========================================================================
    # PRÉDICTION ML
    # =========================================================================
    
    # Scaling
    scaled_data = scaler.transform(features_raw)
    
    # Prédiction
    probabilities = model.predict_proba(scaled_data)[0]
    fraud_prob = float(probabilities[1])
    
    # Décision ML pure
    ml_decision = 1 if fraud_prob >= threshold else 0
    
    # =========================================================================
    # DÉCISION FINALE (RÈGLES > ML)
    # =========================================================================
    
    if rule_triggered:
        # RÈGLE MÉTIER DÉTECTE FRAUDE → FORCER À FRAUDE
        final_decision = 1
        risk_level = "HIGH"
        recommendation = "🚫 BLOQUER - Fraude détectée par règles métier"
        color = "red"
        decision_source = "RÈGLES MÉTIER"
    else:
        # PAS DE RÈGLE → UTILISER ML
        final_decision = ml_decision
        decision_source = "MACHINE LEARNING"
        
        if fraud_prob >= 0.7:
            risk_level = "HIGH"
            recommendation = "🚫 BLOQUER - Fraude hautement probable"
            color = "red"
        elif fraud_prob >= 0.4:
            risk_level = "MEDIUM"
            recommendation = "⚠️ VÉRIFIER - Investigation recommandée"
            color = "orange"
        else:
            risk_level = "LOW"
            recommendation = "✅ APPROUVER - Transaction sûre"
            color = "green"
    
    return {
        'is_fraud': bool(final_decision),
        'fraud_probability': fraud_prob,
        'legitimate_probability': float(probabilities[0]),
        'confidence': float(max(probabilities)),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'color': color,
        'threshold_used': threshold,
        'ml_decision': bool(ml_decision),
        'rule_triggered': rule_triggered,
        'rule_reason': rule_reason,
        'decision_source': decision_source,
        'ratio_amount_orig': ratio_amount_orig,
        'features_raw': features_raw[0].tolist(),
        'features_scaled': scaled_data[0].tolist()
    }

# =============================================================================
# INTERFACE
# =============================================================================

st.title("🔍 Système de Détection de Fraude Bancaire")
st.markdown("*Analyse en Temps Réel avec IA + Règles Métier*")

# SIDEBAR
st.sidebar.header("📊 Informations du Modèle")
st.sidebar.success(f"**Modèle:** {metadata.get('best_model', 'XGBoost')}")
st.sidebar.info(f"**Seuil Optimal:** {optimal_threshold:.3f}")

if 'all_models' in metadata and metadata['all_models']:
    best_model_name = metadata.get('best_model')
    if best_model_name in metadata['all_models']:
        metrics = metadata['all_models'][best_model_name]['metrics']
        st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
        st.sidebar.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.1f}%")

st.sidebar.markdown("---")

# Initialiser session state
if 'demo_type' not in st.session_state:
    st.session_state.demo_type = None

demo_mode = st.sidebar.checkbox("🎮 Mode Démonstration")

# =============================================================================
# FORMULAIRE
# =============================================================================

st.header("📝 Saisie de la Transaction")

# Exemples
if demo_mode:
    st.info("🎮 **Mode Démonstration** - Choisissez un exemple")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Transaction Légitime", use_container_width=True):
            st.session_state.demo_type = "legitimate"
            st.rerun()
    
    with col2:
        if st.button("⚠️ Transaction Suspecte", use_container_width=True):
            st.session_state.demo_type = "suspicious"
            st.rerun()
    
    with col3:
        if st.button("🚨 Fraude Évidente", use_container_width=True):
            st.session_state.demo_type = "fraud"
            st.rerun()

st.markdown("---")

# Valeurs par défaut
default_values = {
    'legitimate': {
        'amount': 150.0, 'old_orig': 5000.0, 'new_orig': 4850.0,
        'old_dest': 3000.0, 'new_dest': 3150.0, 'type': 'PAYMENT',
        'type_idx': 0, 'hour': 14, 'day_idx': 2
    },
    'suspicious': {
        'amount': 15000.0, 'old_orig': 20000.0, 'new_orig': 5000.0,
        'old_dest': 5000.0, 'new_dest': 20000.0, 'type': 'TRANSFER',
        'type_idx': 1, 'hour': 22, 'day_idx': 5
    },
    'fraud': {
        'amount': 50000.0, 'old_orig': 100.0, 'new_orig': 0.0,
        'old_dest': 200000.0, 'new_dest': 250000.0, 'type': 'CASH_OUT',
        'type_idx': 2, 'hour': 3, 'day_idx': 6
    }
}

current_demo = st.session_state.get('demo_type', 'legitimate')
if not demo_mode:
    current_demo = 'legitimate'

defaults = default_values[current_demo]

if demo_mode and st.session_state.demo_type:
    demo_labels = {
        'legitimate': '✅ Exemple: Transaction Légitime',
        'suspicious': '⚠️ Exemple: Transaction Suspecte',
        'fraud': '🚨 Exemple: Fraude Évidente'
    }
    st.success(demo_labels[st.session_state.demo_type])

# Formulaire
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Informations Transaction")
    
    amount = st.number_input(
        "💵 Montant (€)",
        min_value=0.0,
        max_value=1000000.0,
        value=defaults['amount'],
        step=10.0,
        key=f"amount_{current_demo}"
    )
    
    transaction_type = st.selectbox(
        "🏦 Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
        index=defaults['type_idx'],
        key=f"type_{current_demo}"
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
        "🕐 Heure",
        0, 23,
        defaults['hour'],
        key=f"hour_{current_demo}"
    )
    
    day_idx = st.selectbox(
        "📅 Jour",
        ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
        index=defaults['day_idx'],
        key=f"day_{current_demo}"
    ).split()[0]
    
    day_encoding = {
        'Lundi': 0, 'Mardi': 1, 'Mercredi': 2, 'Jeudi': 3,
        'Vendredi': 4, 'Samedi': 5, 'Dimanche': 6
    }
    day_encoded = day_encoding.get(day_idx, 0)

st.markdown("---")

# Bouton
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 ANALYSER LA TRANSACTION", type="primary", use_container_width=True)

if analyze_button:
    st.markdown("---")
    st.markdown("## 🔬 DIAGNOSTIC COMPLET")
    
    # =========================================================================
    # 1. CONSTRUCTION DES FEATURES (10 FEATURES!)
    # =========================================================================
    st.subheader("1️⃣ Construction du Vecteur de Features")
    
    # Encoder type
    type_encoding = {
        'PAYMENT': 1, 'TRANSFER': 2, 'CASH_OUT': 3, 
        'DEBIT': 4, 'CASH_IN': 5
    }
    type_encoded = type_encoding.get(transaction_type, 0)
    
    # Calculer features dérivées
    balance_change_orig = old_balance_orig - new_balance_orig
    balance_change_dest = new_balance_dest - old_balance_dest
    hour_normalized = hour / 23.0
    
    # ⚠️ IMPORTANT: 10 FEATURES EXACTEMENT!
    features = np.array([[
        amount,                  # 1
        old_balance_orig,       # 2
        new_balance_orig,       # 3
        old_balance_dest,       # 4
        new_balance_dest,       # 5
        balance_change_orig,    # 6
        balance_change_dest,    # 7
        type_encoded,           # 8
        hour_normalized,        # 9
        day_encoded             # 10
    ]])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Features BRUTES:**")
        df_raw = pd.DataFrame({
            'Feature': [
                'amount', 'old_balance_orig', 'new_balance_orig',
                'old_balance_dest', 'new_balance_dest', 'balance_change_orig',
                'balance_change_dest', 'type_encoded', 'hour_normalized', 'day_encoded'
            ],
            'Valeur': features[0]
        })
        st.dataframe(df_raw, use_container_width=True)
    
    with col2:
        st.write("**Informations:**")
        st.metric("Type", f"{transaction_type} (code: {type_encoded})")
        st.metric("Montant", f"{amount:,.2f} €")
        st.metric("Δ Émetteur", f"{balance_change_orig:,.2f} €")
        st.metric("Δ Destinataire", f"{balance_change_dest:,.2f} €")
    
    # =========================================================================
    # 2. PRÉDICTION AVEC RÈGLES
    # =========================================================================
    st.markdown("---")
    st.subheader("2️⃣ Analyse & Prédiction")
    
    with st.spinner("⏳ Analyse en cours..."):
        result = predict_fraud_with_rules(features, threshold=optimal_threshold)
    
    st.success("✅ Analyse terminée")
    
    # Afficher features scalées
    with st.expander("🔬 Voir Features Scalées"):
        df_scaled = pd.DataFrame({
            'Feature': [
                'amount', 'old_balance_orig', 'new_balance_orig',
                'old_balance_dest', 'new_balance_dest', 'balance_change_orig',
                'balance_change_dest', 'type_encoded', 'hour_normalized', 'day_encoded'
            ],
            'Valeur Scalée': result['features_scaled']
        })
        st.dataframe(df_scaled, use_container_width=True)
    
    # =========================================================================
    # 3. RÉSULTATS
    # =========================================================================
    st.markdown("---")
    st.markdown("## 🎯 RÉSULTAT FINAL")
    
    # Alerte visuelle
    if result['is_fraud']:
        st.markdown('<div class="alert-fraud">🚨 ALERTE FRAUDE DÉTECTÉE 🚨</div>', unsafe_allow_html=True)
    elif result['risk_level'] == "MEDIUM":
        st.markdown('<div class="alert-warning">⚠️ TRANSACTION SUSPECTE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-safe">✅ TRANSACTION LÉGITIME</div>', unsafe_allow_html=True)
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Probabilité ML", f"{result['fraud_probability']*100:.1f}%")
    
    with col2:
        st.metric("Niveau Risque", result['risk_level'])
    
    with col3:
        st.metric("Décision Source", result['decision_source'])
    
    with col4:
        st.metric("Seuil Utilisé", f"{result['threshold_used']:.3f}")
    
    # Graphique
    fig = create_gauge_chart(result['fraud_probability'], "Probabilité de Fraude", result['color'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Règles métier
    if result['rule_triggered']:
        st.markdown("---")
        st.markdown("### 🚨 Règles Métier Déclenchées")
        for reason in result['rule_reason']:
            st.error(f"• {reason}")
    
    # Recommandation
    st.markdown("---")
    st.markdown("### 💡 Recommandation")
    
    if result['fraud_probability'] >= 0.7 or result['rule_triggered']:
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
        - 📱 SMS de vérification
        - 🔐 Authentification 2FA
        """)
    else:
        st.success(f"""
        **{result['recommendation']}**
        
        Transaction sûre - Surveillance standard
        """)
    
    # Détails techniques
    with st.expander("🔬 Détails Techniques"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "ml_decision": result['ml_decision'],
                "fraud_probability": round(result['fraud_probability'], 4),
                "confidence": round(result['confidence'], 4),
                "threshold": result['threshold_used']
            })
        
        with col2:
            st.json({
                "rule_triggered": result['rule_triggered'],
                "ratio_amount_orig": round(result['ratio_amount_orig'], 2),
                "balance_change_orig": balance_change_orig,
                "balance_change_dest": balance_change_dest
            })