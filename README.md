# EGFR Bioactivity Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6-orange.svg)](https://scikit-learn.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2025.03-green.svg)](https://www.rdkit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

An end-to-end machine learning pipeline to predict small molecule bioactivity against **EGFR** (Epidermal Growth Factor Receptor), a critical target in cancer therapy. This project demonstrates the application of ensemble learning and cheminformatics for drug discovery.

**Model Performance:** 94.54% ROC-AUC  
**Dataset:** 20,033 compounds from ChEMBL database  
**Deployment:** Interactive Streamlit web application

---

## 🧬 Biological Context

**EGFR (Epidermal Growth Factor Receptor):**
- Receptor tyrosine kinase involved in cell growth and proliferation
- Mutated/overexpressed in multiple cancers:
  - Non-small cell lung cancer (NSCLC): 15-30% have EGFR mutations
  - Colorectal cancer: 40-50% EGFR overexpression
  
**FDA-Approved EGFR Inhibitors:**
- Erlotinib (Tarceva) - 2004
- Gefitinib (Iressa) - 2003
- Osimertinib (Tagrisso) - 2015
- Afatinib (Gilotrif) - 2013

---

## 🔬 Methods

### Data Collection
- **Source:** ChEMBL database (CHEMBL203 - Human EGFR)
- **Data points:** 25,758 IC50 bioactivity measurements
- **Final dataset:** 20,033 compounds after preprocessing

### Molecular Descriptors
Calculated 12 physicochemical descriptors using RDKit:
- **Lipinski descriptors:** MW, LogP, H-bond donors/acceptors
- **Extended features:** TPSA, rotatable bonds, aromatic rings, Csp3 fraction

### Machine Learning Pipeline
1. **Data preprocessing:** IC50 → pIC50 conversion, binary classification
2. **Feature engineering:** RDKit molecular descriptors
3. **Class balancing:** SMOTE oversampling (17,488 active : 2,545 inactive)
4. **Models trained:**
   - Random Forest (baseline & optimized)
   - XGBoost
   - Neural Network (256→128→64→32→1)
   - **Weighted Ensemble** (70% RF + 15% XGB + 15% NN)

---

## 📊 Results

### Model Performance

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| **Ensemble** | **0.9454** | 0.9294 | 0.9556 | 0.9603 | 0.9596 |
| Random Forest | 0.9435 | 0.9269 | 0.9562 | 0.9603 | 0.9582 |
| Neural Network | 0.9282 | 0.9114 | 0.9645 | 0.9328 | 0.9484 |
| XGBoost | 0.9234 | 0.9194 | 0.9514 | 0.9565 | 0.9540 |

### Feature Importance (Top 5)
1. **LogP** (18.2%) - Lipophilicity
2. **TPSA** (15.7%) - Polar surface area
3. **NumAromaticRings** (13.4%) - Aromatic character
4. **Molecular Weight** (11.8%)
5. **NumRotatableBonds** (9.3%)

### Validation Results
- **FDA-approved drugs:** 4/4 correctly predicted as active (100%)
- **Known inactive compounds:** 4/4 correctly predicted as inactive (100%)
- **Overall validation accuracy:** 100%

---

## 💻 Installation

### Prerequisites
- Python 3.9+
- Conda or virtualenv

### Setup
```bash
# Clone repository
git clone https://github.com/your-username/egfr-bioactivity-ml.git
cd egfr-bioactivity-ml

# Create conda environment
conda create -n drug_ml python=3.9 -y
conda activate drug_ml

# Install dependencies
conda install -c conda-forge rdkit scikit-learn xgboost pandas numpy matplotlib seaborn -y
pip install tensorflow streamlit chembl_webresource_client imbalanced-learn
```

---

## 🚀 Usage

### Web Application
```bash
streamlit run streamlit_app.py
```

Open browser to `http://localhost:8501`

### Python API
```python
import pickle
from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load models
with open('models/ensemble_model.pkl', 'rb') as f:
    ensemble = pickle.load(f)
nn_model = load_model('models/nn_model.h5')

# Make prediction
smiles = "n1cnc(c2cc(c(cc12)OCCOC)OCCOC)Nc1cc(ccc1)C#C"  # Erlotinib
# ... calculate descriptors and predict
```

---

## 📁 Repository Structure
```
egfr-bioactivity-ml/
├── README.md
├── requirements.txt
├── streamlit_app.py          # Web application
├── data/
│   ├── raw/                  # ChEMBL data
│   └── processed/            # Processed with descriptors
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_validation.ipynb
├── models/
│   ├── ensemble_model.pkl    # Weighted ensemble
│   ├── nn_model.h5           # Neural network
│   ├── rf_best.pkl           # Random Forest
│   ├── xgb_model.pkl         # XGBoost
│   └── scaler.pkl            # Feature scaler
├── results/
│   ├── figures/              # Visualizations
│   └── validation/           # Validation reports
└── src/
    └── ensemble_predictor.py # Prediction utilities
```

---

## 🧪 Key Features

- ✅ **Single compound prediction** with molecular structure visualization
- ✅ **Batch prediction** via CSV upload
- ✅ **Drug-likeness analysis** (Lipinski's Rule of Five)
- ✅ **Model interpretability** (feature importance, individual model breakdown)
- ✅ **Validated on FDA-approved drugs** (100% accuracy)

---

## 📈 Skills Demonstrated

### Bioinformatics & Cheminformatics
- ChEMBL database querying and API integration
- Molecular descriptor calculation (RDKit)
- Structure-activity relationship (SAR) analysis
- Drug-likeness assessment (Lipinski's Rule)

### Machine Learning
- Ensemble learning (Random Forest, XGBoost, Neural Networks)
- Hyperparameter optimization (GridSearchCV)
- Class imbalance handling (SMOTE)
- Model evaluation (ROC-AUC, confusion matrices)

### Software Engineering
- Clean, modular code architecture
- Interactive web application (Streamlit)
- Model serialization and deployment
- Version control (Git/GitHub)

---

## 🎓 Project Insights

### Structure-Activity Relationships
**Optimal active compound profile:**
- Molecular Weight: 350-450 Da
- LogP: 2.5-4.0 (balanced lipophilicity)
- TPSA: 70-110 Ų (good oral bioavailability)
- Aromatic rings: 2-3 (π-π stacking with EGFR)
- Lipinski violations: 0-1 (drug-like)

### Model Interpretability
- **LogP** is the most predictive feature (lipophilicity matters!)
- **TPSA** separates drug-like from non-drug-like compounds
- **Aromatic rings** critical for EGFR binding pocket interactions
- Ensemble approach reduces individual model biases

---

## 🚀 Future Enhancements

- [ ] Add Morgan fingerprints for improved feature representation
- [ ] Implement SHAP values for explainable AI
- [ ] Multi-task learning (predict multiple targets simultaneously)
- [ ] ADMET property prediction (absorption, distribution, metabolism)
- [ ] Deploy to cloud platform (AWS/GCP)
- [ ] Add molecular docking scores as features

---

## 📚 References

1. ChEMBL Database: https://www.ebi.ac.uk/chembl/
2. RDKit Cheminformatics: https://www.rdkit.org/
3. Lipinski's Rule of Five: Lipinski et al. (1997) Adv Drug Deliv Rev
4. EGFR Inhibitors Review: Gazdar (2009) Oncogene

---

## 📧 Contact

**Vedant Kulkarni and Haozhe Zhao**  
Bioinformatics MS, Northeastern University  
Email: kulkarni.vedan@northeastern.edu , zhao.haozh@northeastern.edu
LinkedIn: www.linkedin.com/in/ved-kulk , https://www.linkedin.com/in/haozhe-zhao/  

---

## 📄 License

MIT License - feel free to use for research and educational purposes.

---

## ⚠️ Disclaimer

This tool is for **research purposes only**. Predictions should be validated through experimental assays before any pharmaceutical development.

---

**⭐ If you find this project useful, please star the repository!**
