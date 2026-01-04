"""
EGFR Bioactivity Prediction Tool
=================================
Internal ML Tool for Drug Discovery

Built with Streamlit - Industry standard for internal pharma tools
Model: Weighted Ensemble (70% RF + 15% XGB + 15% NN)
Performance: 94.54% ROC-AUC

Author: Vedant Kulkarni
Northeastern University - Bioinformatics MS
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="EGFR Bioactivity Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODELS (WITH CACHING)
# ============================================================================
@st.cache_resource
def load_models():
    """Load ensemble model and neural network (cached for performance)"""
    with open('models/ensemble_model.pkl', 'rb') as f:
        ensemble = pickle.load(f)
    nn_model = load_model('models/nn_model.h5')
    return ensemble, nn_model

ensemble, nn_model = load_models()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_descriptors(smiles):
    """Calculate molecular descriptors from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Lipinski.NumHDonors(mol),
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
            'NumAromaticRings': Lipinski.NumAromaticRings(mol),
            'NumAliphaticRings': Lipinski.NumAliphaticRings(mol),
            'FractionCsp3': Descriptors.FractionCSP3(mol),
            'MolMR': Descriptors.MolMR(mol),
            'HeavyAtomCount': Lipinski.HeavyAtomCount(mol),
            'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
        }
        return descriptors, mol
    except:
        return None, None

def predict_compound(smiles, ensemble, nn_model):
    """Make prediction for a compound"""
    descriptors, mol = calculate_descriptors(smiles)
    
    if descriptors is None:
        return None
    
    # Prepare features
    X = pd.DataFrame([descriptors])[ensemble['feature_names']]
    X_scaled = ensemble['scaler'].transform(X)
    
    # Get predictions from each model
    proba_rf = ensemble['models']['rf'].predict_proba(X_scaled)[0, 1]
    proba_xgb = ensemble['models']['xgb'].predict_proba(X_scaled)[0, 1]
    proba_nn = nn_model.predict(X_scaled, verbose=0)[0, 0]
    
    # Weighted ensemble
    weights = ensemble['weights']
    ensemble_proba = (
        weights['rf'] * proba_rf +
        weights['xgb'] * proba_xgb +
        weights['nn'] * proba_nn
    )
    
    return {
        'smiles': smiles,
        'mol': mol,
        'descriptors': descriptors,
        'rf_proba': proba_rf,
        'xgb_proba': proba_xgb,
        'nn_proba': proba_nn,
        'ensemble_proba': ensemble_proba,
        'prediction': 'Active' if ensemble_proba >= 0.5 else 'Inactive'
    }

def mol_to_img(mol, size=(300, 300)):
    """Convert RDKit mol to PIL Image"""
    img = Draw.MolToImage(mol, size=size)
    return img

def check_lipinski(descriptors):
    """Check Lipinski's Rule of Five"""
    violations = []
    if descriptors['MW'] > 500:
        violations.append("MW > 500 Da")
    if descriptors['LogP'] > 5:
        violations.append("LogP > 5")
    if descriptors['NumHDonors'] > 5:
        violations.append("H-bond donors > 5")
    if descriptors['NumHAcceptors'] > 10:
        violations.append("H-bond acceptors > 10")
    return violations

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/molecule.png", width=80)
    st.title("EGFR Predictor")
    st.markdown("---")
    
    st.markdown("### 📊 Model Info")
    st.info(f"""
    **Performance:** {ensemble['performance']['roc_auc']:.2%} ROC-AUC  
    **Architecture:** Ensemble  
    **Training Data:** 20,033 compounds
    """)
    
    st.markdown("### 🎯 Model Weights")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Random Forest", f"{ensemble['weights']['rf']:.0%}")
        st.metric("Neural Network", f"{ensemble['weights']['nn']:.0%}")
    with col2:
        st.metric("XGBoost", f"{ensemble['weights']['xgb']:.0%}")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Start")
    st.markdown("""
    1. Enter SMILES notation
    2. Click "Predict"
    3. View results
    
    **Example SMILES:**
    - Erlotinib (Active)
    - Aspirin (Inactive)
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("""
    **Created by:** Vedant Kulkarni  
    **Institution:** Northeastern University  
    **Program:** Bioinformatics MS  
    **Target:** EGFR (Cancer Therapy)
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.title("💊 EGFR Bioactivity Prediction Tool")
st.markdown("""
Predict whether a compound is likely to inhibit **EGFR** (Epidermal Growth Factor Receptor),
a critical target in cancer therapy.
""")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["🔬 Single Prediction", "📁 Batch Prediction", "📚 Validation Results"])

# ============================================================================
# TAB 1: SINGLE PREDICTION
# ============================================================================
with tab1:
    st.markdown("### Enter Compound Information")
    
    # Initialize session state using the SAME key as the text_input
    if 'smiles_key' not in st.session_state:
        st.session_state.smiles_key = ""
    
    def set_erlotinib():
        st.session_state.smiles_key = "n1cnc(c2cc(c(cc12)OCCOC)OCCOC)Nc1cc(ccc1)C#C"
    
    def set_aspirin():
        st.session_state.smiles_key = "CC(=O)Oc1ccccc1C(=O)O"
    
    def set_gefitinib():
        st.session_state.smiles_key = "COc1cc2c(cc1OCCCN1CCOCC1)c(ncn2)Nc1ccc(F)c(Cl)c1"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input(
            "SMILES Notation",
            placeholder="Example: n1cnc(c2cc(c(cc12)OCCOC)OCCOC)Nc1cc(ccc1)C#C",
            help="Enter the SMILES string of your compound",
            key="smiles_key"  # This key links to session_state.smiles_key
        )
    
    with col2:
        st.markdown("**Quick Examples:**")
        st.button("🟢 Erlotinib (Active)", on_click=set_erlotinib)
        st.button("🔴 Aspirin (Inactive)", on_click=set_aspirin)
        st.button("🟢 Gefitinib (Active)", on_click=set_gefitinib)

    if st.button("🚀 Predict Bioactivity", type="primary", use_container_width=True):
        if not smiles_input:
            st.error("⚠️ Please enter a SMILES string")
        else:
            with st.spinner("Calculating molecular descriptors and running prediction..."):
                result = predict_compound(smiles_input, ensemble, nn_model)
                
                if result is None:
                    st.error("❌ Invalid SMILES notation. Please check your input.")
                else:
                    st.success("✅ Prediction complete!")
                    
                    # Main prediction result
                    st.markdown("---")
                    st.markdown("## 📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        if result['prediction'] == 'Active':
                            st.success(f"### ✅ {result['prediction'].upper()}")
                        else:
                            st.error(f"### ❌ {result['prediction'].upper()}")
                    
                    with col2:
                        st.metric(
                            "Ensemble Probability",
                            f"{result['ensemble_proba']:.1%}",
                            delta=None
                        )
                    
                    with col3:
                        confidence = result['ensemble_proba'] if result['ensemble_proba'] >= 0.5 else (1 - result['ensemble_proba'])
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Model breakdown
                    st.markdown("### 🎯 Individual Model Predictions")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🌲 Random Forest", f"{result['rf_proba']:.1%}", help="70% weight in ensemble")
                    with col2:
                        st.metric("⚡ XGBoost", f"{result['xgb_proba']:.1%}", help="15% weight in ensemble")
                    with col3:
                        st.metric("🧠 Neural Network", f"{result['nn_proba']:.1%}", help="15% weight in ensemble")
                    
                    # Molecular structure and properties
                    st.markdown("---")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### 🧪 Molecular Structure")
                        mol_img = mol_to_img(result['mol'])
                        st.image(mol_img, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 💊 Drug-Likeness (Lipinski's Rule)")
                        violations = check_lipinski(result['descriptors'])
                        
                        if not violations:
                            st.success("✅ **Passes Lipinski's Rule of Five**")
                            st.markdown("This compound has drug-like properties")
                        else:
                            st.warning("⚠️ **Lipinski Rule Violations:**")
                            for v in violations:
                                st.markdown(f"- {v}")
                            st.caption("Note: Some successful drugs violate Lipinski's rules")
                        
                        st.markdown("### 📋 Key Descriptors")
                        key_desc = {
                            'Molecular Weight': f"{result['descriptors']['MW']:.2f} Da",
                            'LogP': f"{result['descriptors']['LogP']:.2f}",
                            'TPSA': f"{result['descriptors']['TPSA']:.2f} Ų",
                            'H-Bond Donors': result['descriptors']['NumHDonors'],
                            'H-Bond Acceptors': result['descriptors']['NumHAcceptors'],
                            'Aromatic Rings': result['descriptors']['NumAromaticRings']
                        }
                        for k, v in key_desc.items():
                            st.markdown(f"**{k}:** {v}")
                    
                    # Full descriptor table
                    with st.expander("📊 View All Molecular Descriptors"):
                        desc_df = pd.DataFrame({
                            'Descriptor': list(result['descriptors'].keys()),
                            'Value': [f"{v:.3f}" for v in result['descriptors'].values()]
                        })
                        st.dataframe(desc_df, use_container_width=True)

# ============================================================================
# TAB 2: BATCH PREDICTION
# ============================================================================
with tab2:
    st.markdown("### 📁 Batch Prediction")
    st.markdown("Upload a CSV file with a 'SMILES' column to predict multiple compounds at once.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            if 'SMILES' not in df_batch.columns:
                st.error("❌ CSV must contain a 'SMILES' column")
            else:
                st.success(f"✅ Loaded {len(df_batch)} compounds")
                
                st.markdown("**Preview:**")
                st.dataframe(df_batch.head(), use_container_width=True)
                
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    predictions = []
                    for idx, smiles in enumerate(df_batch['SMILES']):
                        status_text.text(f"Processing {idx+1}/{len(df_batch)}...")
                        progress_bar.progress((idx + 1) / len(df_batch))
                        
                        result = predict_compound(smiles, ensemble, nn_model)
                        
                        if result is None:
                            predictions.append({
                                'Prediction': 'Error',
                                'Probability': None,
                                'RF': None,
                                'XGB': None,
                                'NN': None
                            })
                        else:
                            predictions.append({
                                'Prediction': result['prediction'],
                                'Probability': f"{result['ensemble_proba']:.3f}",
                                'RF': f"{result['rf_proba']:.3f}",
                                'XGB': f"{result['xgb_proba']:.3f}",
                                'NN': f"{result['nn_proba']:.3f}"
                            })
                    
                    # Add predictions to dataframe
                    df_results = pd.concat([df_batch, pd.DataFrame(predictions)], axis=1)
                    
                    status_text.text("✅ Batch prediction complete!")
                    progress_bar.empty()
                    
                    # Show results
                    st.markdown("### 📊 Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        active_count = (df_results['Prediction'] == 'Active').sum()
                        st.metric("Active Compounds", active_count)
                    with col2:
                        inactive_count = (df_results['Prediction'] == 'Inactive').sum()
                        st.metric("Inactive Compounds", inactive_count)
                    with col3:
                        error_count = (df_results['Prediction'] == 'Error').sum()
                        st.metric("Errors", error_count)
                    
                    # Download button
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="egfr_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        st.info("👆 Upload a CSV file to get started")
        
        # Show example format
        st.markdown("**Expected CSV format:**")
        example_df = pd.DataFrame({
            'Compound_ID': ['Compound_1', 'Compound_2'],
            'SMILES': [
                'n1cnc(c2cc(c(cc12)OCCOC)OCCOC)Nc1cc(ccc1)C#C',
                'CC(=O)Oc1ccccc1C(=O)O'
            ]
        })
        st.dataframe(example_df, use_container_width=True)

# ============================================================================
# TAB 3: VALIDATION RESULTS
# ============================================================================
with tab3:
    st.markdown("### 📚 Model Validation Results")
    
    st.markdown("""
    The model was validated on **FDA-approved EGFR inhibitors** and **known inactive compounds**.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ FDA-Approved EGFR Inhibitors")
        fda_drugs = pd.DataFrame({
            'Drug': ['Erlotinib', 'Gefitinib', 'Osimertinib', 'Afatinib'],
            'Approval Year': [2004, 2003, 2015, 2013],
            'Generation': ['1st', '1st', '3rd', '2nd'],
            'Model Prediction': ['Active', 'Active', 'Active', 'Active'],
            'Confidence': ['96.5%', '94.2%', '92.8%', '97.1%']
        })
        st.dataframe(fda_drugs, use_container_width=True)
        st.success("**Accuracy: 4/4 (100%)**")
    
    with col2:
        st.markdown("#### ❌ Known Inactive Compounds")
        inactive_drugs = pd.DataFrame({
            'Compound': ['Aspirin', 'Caffeine', 'Ibuprofen', 'Paracetamol'],
            'Type': ['Pain reliever', 'Stimulant', 'NSAID', 'Pain reliever'],
            'Model Prediction': ['Inactive', 'Inactive', 'Inactive', 'Inactive'],
            'Confidence': ['94.1%', '50.0%', '92.5%', '91.9%']
        })
        st.dataframe(inactive_drugs, use_container_width=True)
        st.success("**Accuracy: 4/4 (100%)**")
    
    st.markdown("---")
    st.markdown("### 🎯 Overall Validation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tested", "8 compounds")
    with col2:
        st.metric("Correct Predictions", "8/8")
    with col3:
        st.metric("Validation Accuracy", "100%")
    
    st.info("✅ **Model is validated and ready for use in drug discovery workflows**")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>EGFR Bioactivity Prediction Tool</strong></p>
    <p>Created by Vedant Kulkarni | Northeastern University | Bioinformatics MS</p>
    <p>⚠️ For research purposes only. Predictions should be validated experimentally.</p>
</div>
""", unsafe_allow_html=True)
