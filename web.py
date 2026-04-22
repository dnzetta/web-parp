import streamlit as st
import pandas as pd
import torch
from torch_geometric.data import Batch
from model import Multimodal, tokenize_smiles
from utils import compute_descriptors, compute_maccs, mol_to_graph

# --- Unified Styling ---
st.markdown("""
<style>
/* ===== PAGE BACKGROUND ===== */
.stApp {
background-color: #eef6fa !important;
font-family: 'sans-serif' !important;
}
            
/* ===== GLOBAL TEXT ===== */
p, span, div, label, h1, h2, h3, h4, h5 {
    color: #002244 !important;
}
            
/* ===== AUTHOR SECTION ===== */
.author {
    background-color: #cce0ff !important;
    color: #003366 !important;
    font-style: italic;
    font-size: 16px;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
    margin-top: 30px;
}

/* ===== TABS ===== */
[data-testid="stTabs"] div[role="tablist"] div[role="tab"] button,
[data-testid="stTabs"] div[role="tablist"] div[role="tab"] button span,
[data-testid="stTabs"] div[role="tablist"] div[role="tab"] button div {
    color: #002244 !important;
    background-color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    opacity: 1 !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stTabs"] div[data-baseweb="tab-panel"] * {
    color: #002244 !important;
}

/* ===== INPUT BOXES ===== */
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #002244 !important;
    border: 1px solid #4da6ff !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
}
div[data-baseweb="input"] input::placeholder {
    color: #666666 !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    color: #002244 !important;
    border: 1px dashed #4da6ff !important;
    border-radius: 8px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"] button {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: 1px solid #000000 !important;
    padding: 8px 16px !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:focus {
    background-color: #f0f0f0 !important;
    color: #000000 !important;
}

/* ===== ALL BUTTONS ===== */
div.stButton > button {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: 1px solid #000000 !important;
    padding: 8px 16px !important;
    }
/* Hover/focus state */ 
div.stButton > button:hover, 
div.stButton > button:focus { 
    background-color: #f0f0f0 !important; /* 
    Slight gray on hover */ 
    color: #000000 !important; 
    }
            
   
</style>
""", unsafe_allow_html=True)

# -------------------
# Load BOTH models
# -------------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_div = Multimodal(desc_dim=13, maccs_dim=167)
    model_unc = Multimodal(desc_dim=13, maccs_dim=167)

    model_div.load_state_dict(torch.load("al_diversity_last_round.pth", map_location=device))
    model_unc.load_state_dict(torch.load("al_uncertainty_last_round.pth", map_location=device))

    model_div.to(device).eval()
    model_unc.to(device).eval()

    return model_div, model_unc, device

model_div, model_unc, device = load_models()

# -------------------
# UI
# -------------------
st.title("🧪 Multimodal PARP Predictor")

tab1, tab2 = st.tabs(["Single SMILES", "CSV Batch"])

with tab1:
    smiles = st.text_input("Enter SMILES")

    if st.button("Predict"):

        if not smiles:
            st.warning("Invalid SMILES")
        else:
            try:
                # --- Feature generation ---
                desc = compute_descriptors(smiles)
                maccs = compute_maccs(smiles)
                graph = mol_to_graph(smiles)
                tokens = tokenize_smiles(smiles)

                # --- Tensor ---
                desc = torch.tensor(desc, dtype=torch.float32).unsqueeze(0).to(device)
                maccs = torch.tensor(maccs, dtype=torch.float32).unsqueeze(0).to(device)
                tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
                graph = Batch.from_data_list([graph]).to(device)

                # --- Predictions ---
                prob_div = model_div.predict_proba(desc, maccs, graph)[0][1]
                prob_unc = model_unc.predict_proba(desc, maccs, graph)[0][1]

                # --- Ensemble ---
                prob_ens = (prob_div + prob_unc) / 2

                # --- Output ---
                st.subheader("🔍 Models Predictions")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Diversity Probability", f"{prob_div:.4f}")

                with col2:
                    st.metric("Uncertainty Probability", f"{prob_unc:.4f}")

                st.subheader("Consensus Result")

                prediction = "Active" if prob_ens > 0.5 else "Inactive"

                st.success(f"Consensus Prediction: {prediction}")
                st.write(f"Consensus Probability: {prob_ens:.4f}")

            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    uploaded_file = st.file_uploader("📂 Upload CSV with SMILES column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "SMILES" not in df.columns:
            st.error("CSV must contain 'SMILES' column")
        else:
            div_probs = []
            unc_probs = []
            ens_probs = []

            for smiles in df["SMILES"]:
                try:
                    desc = compute_descriptors(smiles)
                    maccs = compute_maccs(smiles)
                    graph = mol_to_graph(smiles)

                    desc = torch.tensor(desc, dtype=torch.float32).unsqueeze(0).to(device)
                    maccs = torch.tensor(maccs, dtype=torch.float32).unsqueeze(0).to(device)
                    graph = Batch.from_data_list([graph]).to(device)

                    prob_div = model_div.predict_proba(desc, maccs, graph)[0][1]
                    prob_unc = model_unc.predict_proba(desc, maccs, graph)[0][1]
                    prob_ens = (prob_div + prob_unc) / 2

                    div_probs.append(prob_div)
                    unc_probs.append(prob_unc)
                    ens_probs.append(prob_ens)

                except:
                    div_probs.append(None)
                    unc_probs.append(None)
                    ens_probs.append(None)

            df["Diversity Probability"] = div_probs
            df["Uncertainty Probability"] = unc_probs
            df["Consensus Probability"] = ens_probs

            st.dataframe(df)

# =========================
# Footer
# =========================
# --- Spacer before author section ---
st.markdown("<br><br><br>", unsafe_allow_html=True)

# --- Author Section ---
st.markdown("""
<div class="author">
Authors\n
Andi Endang Kusuma Intan<sup>1</sup>, Darlene Nabila Zetta<sup>1</sup>, and Tarapong Srisongkram<sup>2*</sup>

<sup>1</sup>*Graduate School in the Program of Pharmaceutical Sciences, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen 40002, Thailand*
            
<sup>2</sup>*Division of Pharmaceutical Chemistry, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen 40002, Thailand* </div>
            

""", unsafe_allow_html=True)