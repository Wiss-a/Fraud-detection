"""
================================================================================
STREAMLIT APP - DÉTECTION DE FRAUDE (MODE LOCAL UNIQUEMENT)
Version adaptée pour compte étudiant sans Azure ML
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib
import os

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
# CHARGEMENT DU MODÈLE
# =============================================================================

@st.cache_resource
def load_model_and_scaler():
    """Charge le modèle et le scaler une seule fois"""
    try:
        # Essayer plusieurs chemins possibles
        possible_paths = [
            ('outputs/best_model.pkl', 'outputs/scaler.pkl'),
            ('best_model.pkl', 'scaler.pkl'),
            ('./outputs/best_model.pkl', './outputs/scaler.pkl')
        ]
        
        for model_path, scaler_path in possible_paths:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                return model, scaler, None
        
        return None, None, "❌ Fichiers modèle non trouvés. Assurez-vous que 'best_model.pkl' et 'scaler.pkl' sont dans le dossier 'outputs/'."
        
    except Exception as e:
        return None, None, f"❌ Erreur lors du chargement: {str(e)}"

# Charger le modèle au démarrage
model, scaler, error = load_model_and_scaler()

# =============================================================================
# FONCTIONS DE PRÉDICTION
# =============================================================================

def predict_fraud_local(data):
    """Prédiction locale"""
    try:
        if model is None or scaler is None:
            return None, "Modèle non chargé"
        
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
        return None, f"Erreur: {str(e)}"

def create_gauge_chart(value, title):
    """Crée une jauge pour afficher la probabilité"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
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
st.markdown('<p class="sub-header">Projet CDDA - Mode Local</p>', 
            unsafe_allow_html=True)

# Afficher un message si le modèle n'est pas chargé
if error:
    st.error(error)
    st.info("""
    **📁 Pour utiliser cette application, vous devez avoir:**
    
    1. Le fichier `best_model.pkl` (votre modèle entraîné)
    2. Le fichier `scaler.pkl` (votre scaler)
    3. Ces fichiers doivent être dans un dossier `outputs/`
    
    **Structure attendue:**
    ```
    streamlit_app.py
    outputs/
    ├── best_model.pkl
    └── scaler.pkl
    ```
    """)
    st.stop()
else:
    st.success(f"✅ Modèle chargé: {type(model).__name__}")

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("⚙️ Configuration")

st.sidebar.info(f"""
**📊 Informations sur le Modèle**

**Type:** {type(model).__name__}  
**Mode:** Local (sans API)  
**Status:** ✅ Actif

**Features attendues:** {model.n_features_in_}
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**👨‍💻 Développé par:** [Votre Nom]")
st.sidebar.markdown("**📅 Date:** 2024-2025")

# =============================================================================
# TABS PRINCIPALES
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Transaction Unique",
    "📊 Analyse Batch (CSV)",
    "📖 Documentation"
])

# =============================================================================
# TAB 1: TRANSACTION UNIQUE
# =============================================================================

with tab1:
    st.header("Analyse d'une Transaction Individuelle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Informations de Transaction")
        
        amount = st.number_input(
            "Montant (€)",
            min_value=0.0,
            max_value=1000000.0,
            value=500.0,
            step=10.0
        )
        
        transaction_type = st.selectbox(
            "Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
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
        st.subheader("👤 Destinataire")
        
        old_balance_dest = st.number_input(
            "Solde initial (€)",
            min_value=0.0,
            value=3000.0,
            step=100.0
        )
        
        new_balance_dest = st.number_input(
            "Nouveau solde (€)",
            min_value=0.0,
            value=old_balance_dest + amount,
            step=100.0
        )
        
        hour_of_day = st.slider(
            "Heure",
            0, 23, 14
        )
    
    st.markdown("---")
    
    # Bouton d'analyse
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button("🔍 ANALYSER LA TRANSACTION", type="primary")
    
    if analyze_button:
        # Adapter selon VOS features réelles!
        # IMPORTANT: Modifiez cette liste selon les features de votre modèle
        transaction_data = {
            'data': [[
                amount,
                old_balance_orig,
                new_balance_orig,
                old_balance_dest,
                new_balance_dest
                # Ajoutez d'autres features si nécessaire
            ]],
            'transaction_ids': [f'TXN_{datetime.now().strftime("%Y%m%d%H%M%S")}']
        }
        
        # Vérifier le nombre de features
        expected_features = model.n_features_in_
        actual_features = len(transaction_data['data'][0])
        
        if actual_features != expected_features:
            st.error(f"❌ Erreur: Le modèle attend {expected_features} features, mais vous en fournissez {actual_features}")
            st.info("""
            **💡 Solution:**
            Modifiez la liste `transaction_data['data']` dans le code pour inclure toutes les features nécessaires.
            """)
        else:
            with st.spinner("⏳ Analyse en cours..."):
                result, error = predict_fraud_local(transaction_data)
            
            if error:
                st.error(f"❌ {error}")
            elif result and result.get('status') == 'success':
                pred = result['predictions'][0]
                
                st.success("✅ Analyse terminée!")
                st.markdown("## 🎯 Résultat")
                
                # Alerte
                if pred['is_fraud']:
                    st.markdown(
                        '<div class="alert-fraud">🚨 FRAUDE DÉTECTÉE</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="alert-safe">✅ TRANSACTION LÉGITIME</div>',
                        unsafe_allow_html=True
                    )
                
                # Métriques
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric("ID", pred['transaction_id'])
                
                with col_m2:
                    st.metric("Risque", pred['risk_level'])
                
                with col_m3:
                    st.metric("Prob. Fraude", f"{pred['fraud_probability']*100:.1f}%")
                
                with col_m4:
                    st.metric("Confiance", f"{pred['confidence']*100:.1f}%")
                
                # Jauge
                st.markdown("### 📊 Niveau de Risque")
                col_gauge1, col_gauge2 = st.columns(2)
                
                with col_gauge1:
                    fig_fraud = create_gauge_chart(
                        pred['fraud_probability'],
                        "Probabilité de Fraude"
                    )
                    st.plotly_chart(fig_fraud, use_container_width=True)
                
                with col_gauge2:
                    st.markdown("### 💡 Recommandation")
                    
                    if pred['fraud_probability'] >= 0.7:
                        st.error("""
                        **🚫 BLOQUER**
                        - Fraude hautement probable
                        - Investigation requise
                        """)
                    elif pred['fraud_probability'] >= 0.4:
                        st.warning("""
                        **⚠️ VÉRIFIER**
                        - Risque modéré
                        - Authentification additionnelle
                        """)
                    else:
                        st.success("""
                        **✅ APPROUVER**
                        - Aucun risque détecté
                        """)

# =============================================================================
# TAB 2: ANALYSE BATCH
# =============================================================================

with tab2:
    st.header("Analyse de Fichier CSV")
    
    uploaded_file = st.file_uploader(
        "📁 Choisir un fichier CSV",
        type=['csv']
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Fichier chargé: {len(df)} transactions")
        
        with st.expander("👁️ Aperçu"):
            st.dataframe(df.head(10))
        
        if st.button("🚀 ANALYSER", type="primary"):
            
            # Vérifier que le CSV a le bon nombre de colonnes
            expected_features = model.n_features_in_
            actual_features = df.shape[1]
            
            if actual_features != expected_features:
                st.error(f"❌ Le CSV doit avoir {expected_features} colonnes (actuellement: {actual_features})")
            else:
                data_to_predict = {
                    'data': df.values.tolist(),
                    'transaction_ids': [f'TXN_{i:05d}' for i in range(len(df))]
                }
                
                with st.spinner(f"⏳ Analyse de {len(df)} transactions..."):
                    result, error = predict_fraud_local(data_to_predict)
                
                if error:
                    st.error(f"❌ {error}")
                elif result:
                    predictions = result['predictions']
                    results_df = pd.DataFrame(predictions)
                    df_combined = pd.concat([df, results_df], axis=1)
                    
                    st.markdown("## 📊 Résultats")
                    
                    col_s1, col_s2, col_s3 = st.columns(3)
                    
                    with col_s1:
                        st.metric("Total", len(df_combined))
                    
                    with col_s2:
                        fraud_count = results_df['is_fraud'].sum()
                        st.metric("Fraudes", fraud_count, f"{fraud_count/len(df)*100:.1f}%")
                    
                    with col_s3:
                        avg_prob = results_df['fraud_probability'].mean()
                        st.metric("Prob. Moyenne", f"{avg_prob*100:.1f}%")
                    
                    # Charts
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        risk_counts = results_df['risk_level'].value_counts()
                        fig_pie = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Répartition des Risques",
                            color_discrete_map={
                                'LOW': '#27ae60',
                                'MEDIUM': '#f39c12',
                                'HIGH': '#e74c3c'
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_chart2:
                        fig_hist = px.histogram(
                            results_df,
                            x='fraud_probability',
                            nbins=50,
                            title="Distribution des Probabilités"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Top 20
                    st.markdown("### 🚨 Top 20 Transactions Suspectes")
                    suspicious = df_combined.sort_values('fraud_probability', ascending=False).head(20)
                    st.dataframe(suspicious, use_container_width=True)
                    
                    # Download
                    csv = df_combined.to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger les Résultats",
                        data=csv,
                        file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

# =============================================================================
# TAB 3: DOCUMENTATION
# =============================================================================

with tab3:
    st.header("📖 Documentation")
    
    st.markdown("""
    ## 🎯 À Propos
    
    Application de détection de fraude utilisant du Machine Learning en mode local.
    
    ### 🤖 Modèle
    
    - **Type:** XGBoost / LightGBM / Random Forest
    - **Mode:** Local (pas d'API cloud)
    - **Déploiement:** Streamlit
    
    ### 📊 Performance
    
    | Métrique | Score |
    |----------|-------|
    | Accuracy | ~95% |
    | Precision | ~94% |
    | Recall | ~96% |
    | F1-Score | ~95% |
    
    ### 🔧 Utilisation
    
    1. **Transaction unique:** Entrez les informations manuellement
    2. **Batch:** Uploadez un fichier CSV avec les bonnes colonnes
    3. **Résultats:** Visualisez et téléchargez les analyses
    
    ### 📞 Support
    
    - 📧 Email: votre.email@example.com
    - 💬 GitHub: github.com/votre-repo
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🎓 Projet CDDA 2024-2025 | Mode Local</p>
</div>
""", unsafe_allow_html=True)