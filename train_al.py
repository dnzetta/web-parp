import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem, DataStructs
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, f1_score, precision_recall_curve, pairwise_distances, confusion_matrix, precision_score, recall_score, auc, matthews_corrcoef
from scipy.stats import entropy
from tqdm import tqdm
import umap
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")


# === 1. SMILES Tokenizer and Vocabulary ===
SMILES_CHARS = [
        '<pad>', '<sos>', '<eos>', '<SEP>', '<MASK>', 'c', 'C', '(', ')', 'O', '1', '2', '=', 'N', '.', 
        'n', '3', 'F', 'Cl', '>>', '~', '-', '4', '[C@H]', 'S', '[C@@H]', '[O-]', 'Br', '#', '/', '[nH]', 
        '[N+]', 's', '5', 'o', 'P', '[Na+]', '[Si]', 'I', '[Na]', '[Pd]', '[K+]', '[K]', '[P]', 'B', '[C@]', 
        '[C@@]', '[Cl-]', '6', '[OH-]', '\\', '[N-]', '[Li]', '[H]', '[2H]', '[NH4+]', '[c-]', '[P-]', '[Cs+]',
        '[Li+]', '[Cs]', '[NaH]', '[H-]', '[O+]', '[BH4-]', '[Cu]', '7', '[Mg]', '[Fe+2]', '[n+]', '[Sn]', 
        '[BH-]', '[Pd+2]', '[CH]', '[I-]', '[Br-]', '[C-]', '[Zn]', '[B-]', '[F-]', '[Al]', '[P+]', '[BH3-]',
        '[Fe]', '[C]', '[AlH4]', '[Ni]', '[SiH]', '8', '[Cu+2]', '[Mn]', '[AlH]', '[nH+]', '[AlH4-]', '[O-2]',
        '[Cr]', '[Mg+2]', '[NH3+]', '[S@]', '[Pt]', '[Al+3]', '[S@@]', '[S-]', '[Ti]', '[Zn+2]', '[PH]', 
        '[NH2+]', '[Ru]', '[Ag+]', '[S+]', '[I+3]', '[NH+]', '[Ca+2]', '[Ag]', '9', '[Os]', '[Se]', '[SiH2]',
        '[Ca]', '[Ti+4]', '[Ac]', '[Cu+]', '[S]', '[Rh]', '[Cl+3]', '[cH-]', '[Zn+]', '[O]', '[Cl+]', '[SH]', 
        '[H+]', '[Pd+]', '[se]', '[PH+]', '[I]', '[Pt+2]', '[C+]', '[Mg+]', '[Hg]', '[W]', '[SnH]', '[SiH3]',
        '[Fe+3]', '[NH]', '[Mo]', '[CH2+]', '%10', '[CH2-]', '[CH2]', '[n-]', '[Ce+4]', '[NH-]', '[Co]', 
        '[I+]', '[PH2]', '[Pt+4]', '[Ce]', '[B]', '[Sn+2]', '[Ba+2]', '%11', '[Fe-3]', '[18F]', '[SH-]', 
        '[Pb+2]', '[Os-2]', '[Zr+4]', '[N]', '[Ir]', '[Bi]', '[Ni+2]', '[P@]', '[Co+2]', '[s+]', '[As]', 
        '[P+3]', '[Hg+2]', '[Yb+3]', '[CH-]', '[Zr+2]', '[Mn+2]', '[CH+]', '[In]', '[KH]', '[Ce+3]', '[Zr]',
        '[AlH2-]', '[OH2+]', '[Ti+3]', '[Rh+2]', '[Sb]', '[S-2]', '%12', '[P@@]', '[Si@H]', '[Mn+4]', 'p', 
        '[Ba]', '[NH2-]', '[Ge]', '[Pb+4]', '[Cr+3]', '[Au]', '[LiH]', '[Sc+3]', '[o+]', '[Rh-3]', '%13', 
        '[Br]', '[Sb-]', '[S@+]', '[I+2]', '[Ar]', '[V]', '[Cu-]', '[Al-]', '[Te]', '[13c]', '[13C]', '[Cl]', 
        '[PH4+]', '[SiH4]', '[te]', '[CH3-]', '[S@@+]', '[Rh+3]', '[SH+]', '[Bi+3]', '[Br+2]', '[La]', 
        '[La+3]', '[Pt-2]', '[N@@]', '[PH3+]', '[N@]', '[Si+4]', '[Sr+2]', '[Al+]', '[Pb]', '[SeH]', '[Si-]', 
        '[V+5]', '[Y+3]', '[Re]', '[Ru+]', '[Sm]', '*', '[3H]', '[NH2]', '[Ag-]', '[13CH3]', '[OH+]', '[Ru+3]',
        '[OH]', '[Gd+3]', '[13CH2]', '[In+3]', '[Si@@]', '[Si@]', '[Ti+2]', '[Sn+]', '[Cl+2]', '[AlH-]', 
        '[Pd-2]', '[SnH3]', '[B+3]', '[Cu-2]', '[Nd+3]', '[Pb+3]', '[13cH]', '[Fe-4]', '[Ga]', '[Sn+4]', 
        '[Hg+]', '[11CH3]', '[Hf]', '[Pr]', '[Y]', '[S+2]', '[Cd]', '[Cr+6]', '[Zr+3]', '[Rh+]', '[CH3]', 
        '[N-3]', '[Hf+2]', '[Th]', '[Sb+3]', '%14', '[Cr+2]', '[Ru+2]', '[Hf+4]', '[14C]', '[Ta]', '[Tl+]', 
        '[B+]', '[Os+4]', '[PdH2]', '[Pd-]', '[Cd+2]', '[Co+3]', '[S+4]', '[Nb+5]', '[123I]', '[c+]', '[Rb+]',
        '[V+2]', '[CH3+]', '[Ag+2]', '[cH+]', '[Mn+3]', '[Se-]', '[As-]', '[Eu+3]', '[SH2]', '[Sm+3]', '[IH+]',
        '%15', '[OH3+]', '[PH3]', '[IH2+]', '[SH2+]', '[Ir+3]', '[AlH3]', '[Sc]', '[Yb]', '[15NH2]', '[Lu]', 
        '[sH+]', '[Gd]', '[18F-]', '[SH3+]', '[SnH4]', '[TeH]', '[Si@@H]', '[Ga+3]', '[CaH2]', '[Tl]', 
        '[Ta+5]', '[GeH]', '[Br+]', '[Sr]', '[Tl+3]', '[Sm+2]', '[PH5]', '%16', '[N@@+]', '[Au+3]', '[C-4]',
        '[Nd]', '[Ti+]', '[IH]', '[N@+]', '[125I]', '[Eu]', '[Sn+3]', '[Nb]', '[Er+3]', '[123I-]', '[14c]',
        '%17', '[SnH2]', '[YH]', '[Sb+5]', '[Pr+3]', '[Ir+]', '[N+3]', '[AlH2]', '[19F]', '%18', '[Tb]', 
        '[14CH]', '[Mo+4]', '[Si+]', '[BH]', '[Be]', '[Rb]', '[pH]', '%19', '%20', '[Xe]', '[Ir-]', '[Be+2]', 
        '[C+4]', '[RuH2]', '[15NH]', '[U+2]', '[Au-]', '%21', '%22', '[Au+]', '[15n]', '[Al+2]', '[Tb+3]', 
        '[15N]', '[V+3]', '[W+6]', '[14CH3]', '[Cr+4]', '[ClH+]', 'b', '[Ti+6]', '[Nd+]', '[Zr+]', '[PH2+]', 
        '[Fm]', '[N@H+]', '[RuH]', '[Dy+3]', '%23', '[Hf+3]', '[W+4]', '[11C]', '[13CH]', '[Er]', '[124I]', 
        '[LaH]', '[F]', '[siH]', '[Ga+]', '[Cm]', '[GeH3]', '[IH-]', '[U+6]', '[SeH+]', '[32P]', '[SeH-]',
        '[Pt-]', '[Ir+2]', '[se+]', '[U]', '[F+]', '[BH2]', '[As+]', '[Cf]', '[ClH2+]', '[Ni+]', '[TeH3]',
        '[SbH2]', '[Ag+3]', '%24', '[18O]', '[PH4]', '[Os+2]', '[Na-]', '[Sb+2]', '[V+4]', '[Ho+3]', '[68Ga]',
        '[PH-]', '[Bi+2]', '[Ce+2]', '[Pd+3]', '[99Tc]', '[13C@@H]', '[Fe+6]', '[c]', '[GeH2]', '[10B]',
        '[Cu+3]', '[Mo+2]', '[Cr+]', '[Pd+4]', '[Dy]', '[AsH]', '[Ba+]', '[SeH2]', '[In+]', '[TeH2]', '[BrH+]',
        '[14cH]', '[W+]', '[13C@H]', '[AsH2]', '[In+2]', '[N+2]', '[N@@H+]', '[SbH]', '[60Co]', '[AsH4+]',
        '[AsH3]', '[18OH]', '[Ru-2]', '[Na-2]', '[CuH2]', '[31P]', '[Ti+5]', '[35S]', '[P@@H]', '[ArH]', 
        '[Co+]', '[Zr-2]', '[BH2-]', '[131I]', '[SH5]', '[VH]', '[B+2]', '[Yb+2]', '[14C@H]', '[211At]', 
        '[NH3+2]', '[IrH]', '[IrH2]', '[Rh-]', '[Cr-]', '[Sb+]', '[Ni+3]', '[TaH3]', '[Tl+2]', '[64Cu]',
        '[Tc]', '[Cd+]', '[1H]', '[15nH]', '[AlH2+]', '[FH+2]', '[BiH3]', '[Ru-]', '[Mo+6]', '[AsH+]',
        '[BaH2]', '[BaH]', '[Fe+4]', '[229Th]', '[Th+4]', '[As+3]', '[NH+3]', '[P@H]', '[Li-]', '[7NaH]',
        '[Bi+]', '[PtH+2]', '[p-]', '[Re+5]', '[NiH]', '[Ni-]', '[Xe+]', '[Ca+]', '[11c]', '[Rh+4]', '[AcH]',
        '[HeH]', '[Sc+2]', '[Mn+]', '[UH]', '[14CH2]', '[SiH4+]', '[18OH2]', '[Ac-]', '[Re+4]', '[118Sn]',
        '[153Sm]', '[P+2]', '[9CH]', '[9CH3]', '[Y-]', '[NiH2]', '[Si+2]', '[Mn+6]', '[ZrH2]', '[C-2]',
        '[Bi+5]', '[24NaH]', '[Fr]', '[15CH]', '[Se+]', '[At]', '[P-3]', '[124I-]', '[CuH2-]', '[Nb+4]',
        '[Nb+3]', '[MgH]', '[Ir+4]', '[67Ga+3]', '[67Ga]', '[13N]', '[15OH2]', '[2NH]', '[Ho]', '[Cn]'
    ]
SMILES_VOCAB_SIZE = len(SMILES_CHARS)
char_to_idx = {char: i for i, char in enumerate(SMILES_CHARS)}
idx_to_char = {i: char for i, char in enumerate(SMILES_CHARS)}
MAX_SMILES_LEN = 200  # Max length for padding

def tokenize_smiles(smiles):
    """Tokenizes a SMILES string."""
    tokens = ['<sos>'] + list(smiles) + ['<eos>']
    return [char_to_idx.get(char, char_to_idx['.']) for char in tokens]

# === 1. Dataset and Collation ===
class MolecularDataset(Dataset):
    """Custom PyTorch Dataset for multimodal molecular data."""
    def __init__(self, desc, maccs, smiles, graphs, labels):
        self.desc = torch.tensor(desc.values if hasattr(desc, 'values') else desc, dtype=torch.float32)
        self.maccs = torch.tensor(maccs.values if hasattr(maccs, 'values') else maccs, dtype=torch.float32)
        self.smiles = smiles
        self.graphs = graphs  # Pre-calculated list of PyG Data objects
        if isinstance(labels, (pd.Series, np.ndarray, list)):
             self.labels = torch.tensor(labels, dtype=torch.long)
        else:
             self.labels = labels.clone().detach().long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        smiles_str = self.smiles[idx]
        
        # Tokenize SMILES
        tokenized = tokenize_smiles(smiles_str)
        # Pad or truncate
        if len(tokenized) > MAX_SMILES_LEN:
             tokenized = tokenized[:MAX_SMILES_LEN]
        else:
             tokenized = tokenized + [char_to_idx['<pad>']] * (MAX_SMILES_LEN - len(tokenized))
        
        # Ensure correct type
        smiles_tokens = torch.tensor(tokenized, dtype=torch.long)

        # Get pre-calculated graph
        graph_data = self.graphs[idx]
        
        if graph_data is None:
             # Handle invalid SMILES gracefully by returning a dummy or skipping
             # Ideally we should filter these out beforehand
             return None 

        return {
            'desc': self.desc[idx],
            'maccs': self.maccs[idx],
            'graph_data': graph_data,
            'smiles': smiles_str,
            'smiles_tokens': smiles_tokens,
            'label': self.labels[idx]
        }

def collate_fn(batch):
    """Custom collate function to handle different data types and None values."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Standard tensors
    desc = torch.stack([item['desc'] for item in batch])
    maccs = torch.stack([item['maccs'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    smiles_tokens = torch.stack([item['smiles_tokens'] for item in batch])
    
    # PyG graphs
    graph_data = Batch.from_data_list([item['graph_data'] for item in batch])
    
    # SMILES strings
    smiles = [item['smiles'] for item in batch]

    return {
        'desc': desc, 'maccs': maccs,
        'graph_data': graph_data, 'smiles': smiles,
        'smiles_tokens': smiles_tokens, 'label': labels
    }


def extract_embeddings(model, dataset, device, batch_size=128):
    """Extract latent embeddings from the model for all samples in the dataset."""
    model.eval()
    embeddings = []
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass to get latent features
            _, latent_features, _ = model(
                batch['desc'],
                batch['maccs'],
                batch['graph_data'],
                batch['smiles_tokens']
            )
            
            embeddings.append(latent_features.cpu().numpy())
    
    return np.vstack(embeddings) if embeddings else np.array([])


def compute_and_save_umap(model, desc_train, maccs_train, smiles_train, graphs_train, y_train,
                          labeled_idx, pool_idx, query_idx, strategy_name, round_num, 
                          output_dir, device):
    """Compute UMAP embeddings and save visualization."""
    try:
        # Extract embeddings from current model
        full_dataset = MolecularDataset(desc_train, maccs_train, smiles_train, graphs_train, y_train)
        embeddings = extract_embeddings(model, full_dataset, device)
        
        # Compute UMAP reduction
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot and save
        plot_umap_sampling(embeddings_2d, labeled_idx, pool_idx, query_idx,
                          strategy_name, round_num, output_dir,
                          smiles_train, y_train)
        print(f"    ✅ Saved UMAP plot for round {round_num}")
    except Exception as e:
        print(f"    ⚠️  Error computing UMAP: {e}")


def plot_umap_sampling(
    all_embeddings_2d, labeled_idx, pool_idx, query_idx,
    strategy_name, round_num, output_dir,
    smiles_train, y_train):
    # --- Plotting ---
    labeled_emb = all_embeddings_2d[labeled_idx]
    pool_emb = all_embeddings_2d[pool_idx]
    query_emb = all_embeddings_2d[query_idx]

    plt.figure(figsize=(3, 3))
    
    # Plot pool data (grey)
    plt.scatter(pool_emb[:, 0], pool_emb[:, 1], c='dimgray', alpha=0.7, label='Pool data')
    
    # Plot labeled data (blue)
    plt.scatter(labeled_emb[:, 0], labeled_emb[:, 1], c='royalblue', alpha=0.7, label='Training data')
    
    # Plot queried data (yellow)
    plt.scatter(query_emb[:, 0], query_emb[:, 1], c='goldenrod', edgecolor='black', linewidth=1, label='Queried data')
    
    plt.title(f'UMAP of Latent Space - Round {round_num}, Strategy: {strategy_name.capitalize()}', fontsize=12, fontweight='bold', style='italic')
    plt.xlabel('UMAP 1', fontsize=12, fontweight='bold', style='italic')
    plt.ylabel('UMAP 2', fontsize=12, fontweight='bold', style='italic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Create a dedicated folder for UMAP plots
    umap_dir = os.path.join(output_dir, "umap_plots")
    os.makedirs(umap_dir, exist_ok=True)
    
    output_file = os.path.join(umap_dir, f"round_{round_num}_{strategy_name}.svg")
    plt.savefig(output_file, dpi=300, format='svg', bbox_inches='tight')
    plt.close()

    # --- Save Coordinates to CSV ---
    status = np.full(len(all_embeddings_2d), 'pool', dtype=object)
    status[labeled_idx] = 'labeled'
    status[query_idx] = 'queried'

    df = pd.DataFrame({
        'UMAP_1': all_embeddings_2d[:, 0],
        'UMAP_2': all_embeddings_2d[:, 1],
        'SMILES': smiles_train,
        'Label': y_train,
        'Status': status
    })

    # Create a dedicated folder for UMAP data
    umap_data_dir = os.path.join(output_dir, "umap_data")
    os.makedirs(umap_data_dir, exist_ok=True)
    
    csv_output_file = os.path.join(umap_data_dir, f"round_{round_num}_{strategy_name}_coords.csv")
    df.to_csv(csv_output_file, index=False)

# === 1.6. Metrics Calculation ===
def calculate_classification_metrics(y_true, y_prob_positive_class, k=100):
    y_pred = (y_prob_positive_class > 0.5).astype(int)
    # Ensure there are both classes in y_true to avoid errors in metric calculation
    if len(np.unique(y_true)) < 2:
        return {
            'auroc': 0.5, 'auprc': 0.0, 'f1': 0.0, 'balanced_accuracy': 0.0,
            'recall': 0.0, 'specificity': 0.0, 'precision': 0.0, 
            'mcc': 0.0, f'hit_rate_at_{k}': 0.0
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'auroc': roc_auc_score(y_true, y_prob_positive_class),
        'auprc': average_precision_score(y_true, y_prob_positive_class),
        'f1': f1_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': sensitivity,
        'specificity': specificity,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        f'hit_rate_at_{k}': calculate_hit_rate_at_k(y_true, y_prob_positive_class, k)
    }
    return metrics

def calculate_hit_rate_at_k(y_true, y_prob_positive_class, k=100): 
    """
    Calculates the hit rate (precision) at k.
    This is the fraction of true positives among the top k predicted positives.
    """
    # Sort predictions by probability in descending order
    sorted_indices = np.argsort(y_prob_positive_class)[::-1]
    top_k_indices = sorted_indices[:k]
    
    # Calculate hit rate
    hits = np.sum(y_true[top_k_indices])
    hit_rate = hits / k
    return hit_rate

# === Calculate Enrichment Hits ===
def calculate_enrichment_hits(y_true, y_prob_positive_class, fractions=[0.01, 0.05, 0.1]):
    """
    Calculates the number of hits and Enrichment Factor (EF) at various fractions of the top-ranked list.
    
    Args:
        y_true (np.ndarray): True binary labels.
        y_prob_positive_class (np.ndarray): Predicted probabilities for the positive class.
        fractions (List[float]): List of fractions (e.g., [0.01, 0.05]) to evaluate.
        
    Returns:
        dict: A dictionary containing 'hits_at_X%' and 'ef_at_X%' for each fraction.
    """
    n_total = len(y_true)
    n_positives = np.sum(y_true)
    
    # Sort by predicted probability in descending order
    sorted_indices = np.argsort(y_prob_positive_class)[::-1]
    sorted_y_true = y_true[sorted_indices]
    
    results = {}
    for frac in fractions:
        k = int(n_total * frac)
        if k == 0:
            k = 1  # Ensure at least 1 sample is considered
        
        # Hits in the top k
        hits_k = np.sum(sorted_y_true[:k])
        
        # Enrichment Factor: (Hits / k) / (Total Positives / Total Samples)
        # If no positives exist, EF is 0. If k=0, it's handled above.
        precision_k = hits_k / k
        prior = n_positives / n_total
        ef_k = precision_k / prior if prior > 0 else 0.0
        
        results[f'hits_at_{int(frac*100)}%'] = hits_k
        results[f'ef_at_{int(frac*100)}%'] = ef_k
        
    return results

# === 2. CNN Module ===
class CNN_Module(nn.Module):
    """CNN for feature extraction from fingerprints"""
    def __init__(self, input_dim, output_dim=128):
        super(CNN_Module, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (batch, 128)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, output_dim)
        return x

# === 3. Transformer Module ===
class TransformerModule(nn.Module):
    """Transformer for feature extraction"""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, output_dim=64):
        super(TransformerModule, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.embedding(x)  # (batch, 1, d_model)
        x = self.transformer(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, output_dim)
        return x

# === 3.5 GNN Module ===
class GNN_Module(nn.Module):
    """Enhanced GNN with more layers and skip connections"""
    def __init__(self, input_dim=8, feature_dim=256):  # Increase capacity
        super(GNN_Module, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = GCNConv(256, 256)  # Add third layer
        self.bn3 = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)
        
        # Layer 2
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2)
        
        # Layer 3 with skip connection
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        x3 = x3 + x2  # Residual connection
        x3 = self.dropout(x3)
        
        # Global pooling
        x = global_mean_pool(x3, batch)
        x = self.fc(x)
        
        return x
    
class SmilesDecoder(nn.Module):
    """Transformer Decoder for SMILES reconstruction"""
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, latent_dim):
        super(SmilesDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, MAX_SMILES_LEN, embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.latent_to_memory = nn.Linear(latent_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tgt_tokens, memory):
        # tgt_tokens shape: (batch, seq_len)
        # memory shape: (batch, latent_dim)
        
        tgt_embed = self.embedding(tgt_tokens) + self.pos_encoder[:, :tgt_tokens.size(1), :]
        
        # Project latent vector to match decoder dimension and repeat for each token
        memory_proj = self.latent_to_memory(memory).unsqueeze(1).repeat(1, tgt_tokens.size(1), 1)
        
        # Generate a mask to prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)
        
        output = self.transformer_decoder(tgt_embed, memory_proj, tgt_mask=tgt_mask)
        return self.fc_out(output)
# === 4. Combined Model ===
class Multimodal(nn.Module):
    """Enhanced Multimodal with better architecture"""
    def __init__(self, desc_dim, maccs_dim, feature_dim=64):  # Removed rdkit_dim
        super(Multimodal, self).__init__()
        
        # --- Encoder Part with Residual Connections ---
        self.cnn_desc  = CNN_Module(desc_dim, feature_dim)
        self.cnn_maccs = CNN_Module(maccs_dim, feature_dim)
        
        # Transformer with more capacity
        self.trans_desc  = TransformerModule(desc_dim, d_model=128, nhead=4, num_layers=2, output_dim=feature_dim)
        self.trans_maccs = TransformerModule(maccs_dim, d_model=128, nhead=4, num_layers=2, output_dim=feature_dim)

        # Enhanced GNN
        self.gnn = GNN_Module(input_dim=8, feature_dim=feature_dim)
        
        # 🔥 IMPROVED: Deeper fusion with residual connections and attention
        fusion_dim = feature_dim * 5  # Reduced from 7
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 🔥 ADD: Attention mechanism for feature importance
        self.feature_attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 5),  # 5 feature groups
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

        # Decoder
        self.smiles_decoder = SmilesDecoder(
            vocab_size=SMILES_VOCAB_SIZE,
            embed_dim=64,
            nhead=4,
            num_layers=3,
            latent_dim=64
        )
        
    def forward(self, desc, maccs, graph_data, smiles_tokens):
        # --- Encoder ---
        cnn_desc_feat = self.cnn_desc(desc)
        cnn_maccs_feat = self.cnn_maccs(maccs)
        
        trans_desc_feat = self.trans_desc(desc)
        trans_maccs_feat = self.trans_maccs(maccs)

        gnn_feat = self.gnn(graph_data)
        
        # 🔥 Concatenate with attention weighting
        combined = torch.cat([
            cnn_desc_feat, cnn_maccs_feat,
            trans_desc_feat, trans_maccs_feat,
            gnn_feat
        ], dim=1)
        
        # 🔥 Apply feature attention
        attention_weights = self.feature_attention(combined).unsqueeze(2)  # (batch, 5, 1)
        feature_groups = combined.view(combined.size(0), 5, -1)  # (batch, 5, feature_dim)
        weighted_features = (feature_groups * attention_weights).view(combined.size(0), -1)
        
        # Fusion
        latent_features = self.fusion(weighted_features)
        
        # Classification
        logits = self.classifier(latent_features)
        
        # Decoder
        decoder_input = smiles_tokens[:, :-1]
        reconstruction_logits = self.smiles_decoder(decoder_input, latent_features)
        
        return logits, latent_features, reconstruction_logits
    
    def predict_proba(self, desc, maccs, graph_data):
        """Get probability predictions. Ignores decoder for prediction."""
        with torch.no_grad():
            # Create dummy tokens for forward pass during prediction
            batch_size = desc.size(0)
            dummy_tokens = torch.zeros((batch_size, 2), dtype=torch.long).to(desc.device)
            logits, _, _ = self.forward(desc, maccs, graph_data, dummy_tokens)
            probs = F.softmax(logits, dim=1)
        return np.array(probs.detach().cpu().tolist())


# === 4.5. Focal Loss ===
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are the logits from the model (batch_size, C)
        # targets are the ground truth labels (batch_size)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# === 5. Training Function ===
def train_multimodal(model, train_loader, criterion_cls, criterion_recon, optimizer, device, recon_weight=0.2):
    """Train Multimodal for one epoch with combined loss"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        if batch is None: continue
        desc = batch['desc'].to(device)
        maccs = batch['maccs'].to(device)
        graph_data = batch['graph_data'].to(device)
        labels = batch['label'].to(device)
        smiles_tokens = batch['smiles_tokens'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        cls_logits, _, recon_logits = model(desc, maccs, graph_data, smiles_tokens)
        
        # --- Calculate Losses ---
        # 1. Classification Loss
        loss_cls = criterion_cls(cls_logits, labels)
        
        # 2. Reconstruction Loss
        # Target is tokens shifted by one, excluding the first one (<sos>)
        recon_target = smiles_tokens[:, 1:]
        # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize) and (Batch * SeqLen)
        loss_recon = criterion_recon(
            recon_logits.reshape(-1, SMILES_VOCAB_SIZE),
            recon_target.reshape(-1)
        )
        
        # Combined Loss
        loss = loss_cls + recon_weight * loss_recon
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = cls_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


# === 5.5. Train with Validation ===
def train_with_validation(model, full_train_dataset, epochs, criterion_cls, criterion_recon, optimizer, scheduler, device):
    """
    Trains a model, using a validation split to find the best model state.
    Returns the best model state dict and training history.
    """
    if len(full_train_dataset) < 5: # Cannot split if too small
        train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        best_model_state = None
        history = {'loss': [], 'acc': [], 'val_auprc': []}
        for epoch in range(epochs):
            loss, acc = train_multimodal(model, train_loader, criterion_cls, criterion_recon, optimizer, device)
            history['loss'].append(loss)
            history['acc'].append(acc)
            history['val_auprc'].append(0.0) # No validation
        best_model_state = model.state_dict()
        return best_model_state, history

    # Split the full training data set nto sub-train and sub-validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    sub_train_dataset, sub_val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    sub_train_loader = DataLoader(sub_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    sub_val_loader = DataLoader(sub_val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    best_val_auprc = 0.0
    best_model_state = None
    history = {'loss': [], 'acc': [], 'val_auprc': []}

    for epoch in range(epochs):
        loss, acc = train_multimodal(model, sub_train_loader, criterion_cls, criterion_recon, optimizer, device)
        history['loss'].append(loss)
        history['acc'].append(acc)

        # Evaluate on the sub-validation set
        val_probs, val_labels = evaluate_multimodal(model, sub_val_loader, device)
        if len(np.unique(val_labels)) < 2:
            val_auprc = 0.0
        else:
            precision, recall, _ = precision_recall_curve(val_labels, val_probs[:, 1])
            val_auprc = auc(recall, precision)
        history['val_auprc'].append(val_auprc)

        scheduler.step(val_auprc)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_model_state = model.state_dict()
            
    # If no improvement was seen, save the last state
    if best_model_state is None:
        best_model_state = model.state_dict()

    return best_model_state, history


# === 6. Evaluation Function ===
def evaluate_multimodal(model, data_loader, device):
    """Evaluate Multimodal and return predictions"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None: continue
            desc = batch['desc'].to(device)
            maccs = batch['maccs'].to(device)
            # rdkit removed
            graph_data = batch['graph_data'].to(device)
            
            # Create dummy tokens for evaluation pass
            batch_size = desc.size(0)
            dummy_tokens = torch.zeros((batch_size, 2), dtype=torch.long).to(device)
            
            logits, _, _ = model(desc, maccs, graph_data, dummy_tokens)
            probs = F.softmax(logits, dim=1)
            
            # Use tolist() to avoid "Numpy is not available" error
            all_probs.extend(probs.detach().cpu().tolist())
            if 'label' in batch:
                all_labels.extend(batch['label'].detach().cpu().tolist())
    
    all_probs = np.array(all_probs)
    if all_labels:
        all_labels = np.array(all_labels)
        return all_probs, all_labels
    return all_probs


# === 7. Active Learning Sampling Strategies ===

def get_latent_embeddings(model, dataset, device):
    """Helper function to get latent space embeddings for a dataset."""
    model.eval()
    embeddings = []
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            desc = batch['desc'].to(device)
            maccs = batch['maccs'].to(device)
            graph_data = batch['graph_data'].to(device)
            
            # Encoder pass
            cnn_desc_feat = model.cnn_desc(desc)
            cnn_maccs_feat = model.cnn_maccs(maccs)
            
            trans_desc_feat = model.trans_desc(desc)
            trans_maccs_feat = model.trans_maccs(maccs)
            
            gnn_feat = model.gnn(graph_data)
            
            combined = torch.cat([
                cnn_desc_feat, cnn_maccs_feat,
                trans_desc_feat, trans_maccs_feat,
                gnn_feat
            ], dim=1)

            # 🔥 Apply feature attention
            attention_weights = model.feature_attention(combined).unsqueeze(2)
            feature_groups = combined.view(combined.size(0), 5, -1)
            weighted_features = (feature_groups * attention_weights).view(combined.size(0), -1)
            
            latent_features = model.fusion(weighted_features)
            embeddings.extend(latent_features.detach().cpu().tolist())
            
    return np.array(embeddings)


def uncertainty_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select samples with highest uncertainty (closest to 0.5 probability)"""
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    probs, _ = evaluate_multimodal(model, pool_loader, device)
    
    # For binary classification, uncertainty is highest when probability is close to 0.5
    uncertainty = np.abs(probs[:, 1] - 0.5)  # Distance from 0.5
    
    # Select samples with the smallest distance to 0.5 (highest uncertainty)
    # We use argsort which sorts in ascending order, so we want the smallest values.
    selected_idx = np.argsort(uncertainty)[:n_samples]
    return selected_idx


def confidence_sampling_toxic(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Selects samples with the highest predicted probability for the toxic class (exploitation)."""
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    probs, _ = evaluate_multimodal(model, pool_loader, device)
    
    # Assuming class 1 is the "toxic" class
    toxic_class_prob = probs[:, 1]
    
    # Get the indices of the samples with the highest probability for the toxic class
    selected_idx = np.argsort(toxic_class_prob)[-n_samples:]
    return selected_idx


def random_sampling(pool_size: int, n_samples: int) -> np.ndarray:
    """Random sampling"""
    return np.random.choice(pool_size, n_samples, replace=False)


def diversity_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select diverse samples using k-means++ like approach in the latent space."""
    pool_embeddings = get_latent_embeddings(model, pool_dataset, device)
    
    selected_idx = []
    # Select the first point randomly
    first_idx = np.random.randint(0, len(pool_embeddings))
    selected_idx.append(first_idx)
    
    for _ in range(n_samples - 1):
        selected_features = pool_embeddings[selected_idx]
        # Calculate distance from all points to the already selected points
        distances = pairwise_distances(pool_embeddings, selected_features, metric='euclidean')
        # Find the minimum distance for each point to any of the selected points
        min_distances = distances.min(axis=1)
        
        # Avoid re-selecting already chosen samples
        min_distances[selected_idx] = -1
        # Select the point that is furthest from any already selected point
        next_idx = np.argmax(min_distances)
        selected_idx.append(next_idx)
    
    return np.array(selected_idx)


# === 8.6. Generate Reconstructions ===
def generate_reconstructions(model, test_loader, idx_to_char, device, output_dir, n_samples_to_show=20):
    """
    Generates reconstructed SMILES from the test set using the trained autoencoder.
    """
    model.eval()
    original_smiles, reconstructed_smiles, predictions, true_labels = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Generating Reconstructions")):
            if batch is None: continue
            desc = batch['desc'].to(device)
            maccs = batch['maccs'].to(device)
            graph_data = batch['graph_data'].to(device)
            labels = batch['label']
            
            # --- 1. Encoder Pass to get latent features ---
            cnn_desc_feat = model.cnn_desc(desc)
            cnn_maccs_feat = model.cnn_maccs(maccs)
            
            trans_desc_feat = model.trans_desc(desc)
            trans_maccs_feat = model.trans_maccs(maccs)
            
            gnn_feat = model.gnn(graph_data)
            
            combined = torch.cat([
                cnn_desc_feat, cnn_maccs_feat,
                trans_desc_feat, trans_maccs_feat,
                gnn_feat
            ], dim=1)

            # 🔥 Apply feature attention
            attention_weights = model.feature_attention(combined).unsqueeze(2)
            feature_groups = combined.view(combined.size(0), 5, -1)
            weighted_features = (feature_groups * attention_weights).view(combined.size(0), -1)
            
            latent_features = model.fusion(weighted_features)
            # --- Get classification prediction ---
            logits = model.classifier(latent_features)
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            
            # --- 2. Autoregressive Decoding ---
            batch_size = latent_features.size(0)
            # Start with the <sos> token for each sequence in the batch
            decoder_input = torch.full((batch_size, 1), char_to_idx['<sos>'], dtype=torch.long, device=device)
            
            # Project the latent features once to create the memory for the decoder
            memory = model.smiles_decoder.latent_to_memory(latent_features)
            
            # Generate sequence step-by-step
            for _ in range(MAX_SMILES_LEN - 1):
                # The memory shape for the decoder should be (seq_len, batch, embed_dim)
                # but since we generate one token at a time, we can adapt.
                # Let's make memory (batch, 1, embed_dim) for simplicity with batch_first=True
                memory_for_step = memory.unsqueeze(1)

                tgt_embed = model.smiles_decoder.embedding(decoder_input) + model.smiles_decoder.pos_encoder[:, :decoder_input.size(1), :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)

                # The memory needs to be repeated for each token in the target sequence
                memory_proj = memory.unsqueeze(1).repeat(1, decoder_input.size(1), 1)

                output = model.smiles_decoder.transformer_decoder(tgt_embed, memory_proj, tgt_mask=tgt_mask)
                
                # Get the prediction for the very last token
                last_token_logits = model.smiles_decoder.fc_out(output[:, -1, :])
                next_token = torch.argmax(last_token_logits, dim=1).unsqueeze(1)
                
                # Append the predicted token to the input for the next iteration
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # --- 3. Convert tokens to SMILES strings ---
            for j in range(batch_size):
                original_smiles.append(batch['smiles'][j])
                true_labels.append(labels[j].item())
                predictions.append(probs[j])
                
                # Convert sequence of indices to string
                seq = ""
                for token_idx in decoder_input[j, :]:
                    char = idx_to_char.get(token_idx.item())
                    if char == '<eos>': break
                    if char not in ['<sos>', '<pad>']:
                        seq += char
                reconstructed_smiles.append(seq)

    # Create and save DataFrame
    df = pd.DataFrame({
        'Original_SMILES': original_smiles,
        'Reconstructed_SMILES': reconstructed_smiles,
        'True_Label': true_labels,
        'Predicted_Proba': predictions
    })
    
    output_path = os.path.join(output_dir, "baseline_reconstructions.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} SMILES reconstructions to {output_path}")
    
    # Print a few examples
    print("\n🔍 Example Reconstructions:")
    # print(df.head(n_samples_to_show).to_string()) # this caused error
    
    return df


# === 8.5. Baseline: Train with initial data and evaluate ===
def run_initial_model_evaluation(
    desc_train, maccs_train, smiles_train, graphs_train, y_train,
    desc_test, maccs_test, smiles_test, graphs_test, y_test,
    initial_idx, pool_idx,
    output_dir,
    epochs=10,
    n_acquire=100  # Number of samples to acquire from the pool
):
    """
    Trains a model on the initial samples.
    """
    print(f"\n{'='*80}")
    print(f"🎯 Running Baseline Evaluation on Initial {len(initial_idx)} Samples")
    print(f"{'='*80}\n")

    # Create datasets
    initial_train_dataset = MolecularDataset(
        desc_train.iloc[initial_idx], maccs_train.iloc[initial_idx],
        smiles_train[initial_idx], [graphs_train[i] for i in initial_idx], y_train[initial_idx]
    )
    pool_dataset = MolecularDataset(
        desc_train.iloc[pool_idx], maccs_train.iloc[pool_idx],
        smiles_train[pool_idx], [graphs_train[i] for i in pool_idx], y_train[pool_idx]
    )
    test_dataset = MolecularDataset(
        desc_test, maccs_test, smiles_test, graphs_test, y_test
    )

    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    # Create and train model
    model = Multimodal(
        desc_dim=desc_train.shape[1], maccs_dim=maccs_train.shape[1]
    ).to(device)

    criterion_cls = FocalLoss()
    criterion_recon = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    print(f"📊 Training on {len(initial_idx)} initial samples for {epochs} epochs...")
    best_model_state, history = train_with_validation(
        model, initial_train_dataset, epochs, criterion_cls, criterion_recon, optimizer, scheduler, device
    )
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, os.path.join(output_dir, 'baseline_initial_one_shot_model.pth'))

    # 1. Evaluate on the unseen test set
    test_probs, test_labels = evaluate_multimodal(model, test_loader, device)
    test_metrics = calculate_classification_metrics(test_labels, test_probs[:, 1], k=100)
    test_auprc = test_metrics['auprc']
    test_hit_rate = test_metrics['hit_rate_at_100']
    print(f"✅ Initial Model Test AUPRC: {test_auprc:.4f}")
    print(f"✅ Initial Model Test Hit Rate @ 100: {test_hit_rate:.0f}")

    # 2. Predict on the pool to simulate screening and calculate "Hit 100"
    print(f"\n🔍 Simulating screening on the pool of {len(pool_idx)} samples to find top {n_acquire}...")
    pool_probs, pool_labels = evaluate_multimodal(model, pool_loader, device)
    
    # Get indices of top n_acquire predicted toxic samples from the pool
    top_acquire_indices_local = np.argsort(pool_probs[:, 1])[-n_acquire:]
    top_acquire_indices_global = pool_idx[top_acquire_indices_local]

    # Save predictions for the acquired samples
    acquired_df = pd.DataFrame({
        'SMILES': smiles_train[top_acquire_indices_global],
        'True_Label': y_train[top_acquire_indices_global],
        'Predicted_Proba_Toxic': pool_probs[top_acquire_indices_local, 1],
        'Source': 'Acquired_Baseline'
    })

    # Combine with initial data for the full set
    initial_df = pd.DataFrame({
        'SMILES': smiles_train[initial_idx],
        'True_Label': y_train[initial_idx],
        'Predicted_Proba_Toxic': -1,  # No prediction, as it was in training
        'Source': 'Initial_Random'
    })
    
    hit_df = pd.concat([initial_df, acquired_df], ignore_index=True)
    total_samples_count = len(hit_df)
    hit_csv_path = os.path.join(output_dir, f"baseline_hit_{total_samples_count}_samples.csv")
    hit_df.to_csv(hit_csv_path, index=False)
    print(f"✅ Saved baseline's {total_samples_count} selected samples to {hit_csv_path}")

    # Calculate Cumulative Hits
    total_hits = hit_df['True_Label'].sum()
    print(f"✅ Baseline Total Hits: Found {total_hits:.0f} toxic compounds in the combined {total_samples_count} samples.")

    # Save summary results
    baseline_results = {
        'initial_samples': len(initial_idx),
        'n_acquire': n_acquire,
        'test_auprc': test_auprc, 
        'test_hit_rate_100': test_hit_rate,
        'baseline_total_hits': total_hits,
        'total_samples': total_samples_count
    }

    summary_df = pd.DataFrame([baseline_results])
    summary_csv_path = os.path.join(output_dir, "initial_model_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved initial model summary to {summary_csv_path}")

    return baseline_results


# === 9. Active Learning Experiment ===
def run_al_experiment_for_strategy(
    strategy_name,
    desc_train, maccs_train, smiles_train, graphs_train, y_train,
    test_loader, smiles_test, # External Test
    y_test, 
    internal_test_loader, # Internal Test Loader
    initial_idx, pool_idx,
    output_dir,
    n_queries, n_instances, epochs_per_round,
    device
):
    """Runs a full active learning loop for a single strategy."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*30} Running Strategy: {strategy_name.capitalize()} {'='*30}")

    # Initialize indices for this strategy
    labeled_idx = initial_idx.copy()
    pool_idx = pool_idx.copy()

    # Performance tracking for this strategy
    test_metrics_history = []  # Store full dict of metrics (External)
    internal_test_metrics_history = [] # Store AUPRC for internal
    
    cumulative_hits_history = [y_train[initial_idx].sum()]
    enrichment_history = []
    acquired_samples_list = []
    
    # Initial dataset and model
    initial_dataset = MolecularDataset(
        desc_train.iloc[initial_idx],
        maccs_train.iloc[initial_idx],
        smiles_train[initial_idx],
        [graphs_train[i] for i in initial_idx],
        y_train[initial_idx]
    )
    
    model = Multimodal(
        desc_dim=desc_train.shape[1], 
        maccs_dim=maccs_train.shape[1]
    ).to(device)
    
    criterion_cls = FocalLoss()
    criterion_recon = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    # Initial training and evaluation
    print(f"  Round 0: Training on initial {len(initial_idx)} samples...")
    best_initial_state, _ = train_with_validation(
        model, initial_dataset, epochs_per_round, criterion_cls, criterion_recon, optimizer, scheduler, device
    )
    model.load_state_dict(best_initial_state)
    
    # External Test Eval
    probs, labels = evaluate_multimodal(model, test_loader, device)
    metrics = calculate_classification_metrics(labels, probs[:, 1], k=100)
    test_metrics_history.append(metrics)
    
    # Internal Test Eval
    int_probs, int_labels = evaluate_multimodal(model, internal_test_loader, device)
    int_metrics = calculate_classification_metrics(int_labels, int_probs[:, 1], k=100)
    internal_test_metrics_history.append(int_metrics)
    
    # Calculate enrichment metrics
    enrichment_metrics = calculate_enrichment_hits(labels, probs[:, 1], fractions=[0.01, 0.05, 0.1])
    enrichment_history.append(enrichment_metrics)
    
    print(f"    - Initial Test AUPRC (Ext): {metrics['auprc']:.4f}, (Int): {int_metrics['auprc']:.4f}, Initial Hits: {cumulative_hits_history[0]}")
    
    # Compute and save UMAP plot for initial round
    try:
        compute_and_save_umap(model, desc_train, maccs_train, smiles_train, graphs_train, y_train,
                             labeled_idx, pool_idx, np.array([], dtype=int), strategy_name, 0, 
                             output_dir, device)
    except Exception as e:
        print(f"    ⚠️  Could not save initial UMAP plot: {e}")

    # Active learning loop
    for i in range(n_queries):
        print(f"\n  Query round {i+1}/{n_queries}")

        if len(pool_idx) == 0:
            print("    ⚠️  No more samples in pool. Stopping.")
            break
        
        n_instances_round = min(n_instances, len(pool_idx))


        # Create current datasets
        labeled_dataset = MolecularDataset(
            desc_train.iloc[labeled_idx], 
            maccs_train.iloc[labeled_idx],
            smiles_train[labeled_idx],
            [graphs_train[i] for i in labeled_idx],
            y_train[labeled_idx]
        )
        pool_dataset = MolecularDataset(
            desc_train.iloc[pool_idx],
            maccs_train.iloc[pool_idx],
            smiles_train[pool_idx],
            [graphs_train[i] for i in pool_idx],
            y_train[pool_idx]
        )

        # Train model on current labeled set to guide sampling
        best_state, _ = train_with_validation(
            model, labeled_dataset, epochs_per_round, criterion_cls, criterion_recon, optimizer, scheduler, device
        )
        model.load_state_dict(best_state)

        # Select samples
        pool_loader_for_sampling = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
        pool_probs, _ = evaluate_multimodal(model, pool_loader_for_sampling, device)

        if strategy_name == 'random':
            query_idx_local = random_sampling(len(pool_idx), n_instances_round)
        elif strategy_name == 'uncertainty':
            query_idx_local = uncertainty_sampling(model, pool_dataset, n_instances_round, device)
        elif strategy_name == 'diversity':
            query_idx_local = diversity_sampling(model, pool_dataset, n_instances_round, device)
        else: # confidence_toxic (greedy)
            query_idx_local = confidence_sampling_toxic(model, pool_dataset, n_instances_round, device)

        # Get absolute indices and update sets
        query_idx_abs = pool_idx[query_idx_local]
        
        # --- 1. Save Acquired Samples (Sampling SMILES Hit/Non-Hits) ---
        acquired_this_round_df = pd.DataFrame({
            'SMILES': smiles_train[query_idx_abs],
            'True_Label': y_train[query_idx_abs],
            'Predicted_Proba_Toxic': pool_probs[query_idx_local, 1],
            'Strategy': strategy_name,
            'Round': i + 1
        })
        acquired_samples_list.append(acquired_this_round_df)
        
        acquired_file_path = os.path.join(output_dir, f"al_{strategy_name}_round_{i+1}_acquired_samples.csv")
        acquired_this_round_df.to_csv(acquired_file_path, index=False)

        # --- Update Cumulative Hits ---
        new_hits_mask = y_train[query_idx_abs] == 1
        new_hits_count = new_hits_mask.sum()
        cumulative_hits_history.append(cumulative_hits_history[-1] + new_hits_count)
        
        # Calculate "Pool Hit Rate" (Enrichment in the acquired batch)
        # This tells us: Of the 100 compounds we just bought, what % are toxic?
        pool_hit_rate = new_hits_count / len(query_idx_abs) if len(query_idx_abs) > 0 else 0
        print(f"    🎣 Pool Acquisition: Bought {len(query_idx_abs)} samples, Found {new_hits_count} Toxic (Hit Rate: {pool_hit_rate:.2%})")

        # Update indices for the next round
        labeled_idx = np.concatenate([labeled_idx, query_idx_abs])
        pool_idx = np.setdiff1d(pool_idx, query_idx_abs)
        
        # Create updated dataset for final evaluation this round
        updated_labeled_dataset = MolecularDataset(
            desc_train.iloc[labeled_idx],
            maccs_train.iloc[labeled_idx],
            smiles_train[labeled_idx],
            [graphs_train[i] for i in labeled_idx],
            y_train[labeled_idx]
        )
        
        # Retrain on the newly expanded set
        print(f"    Retraining on {len(labeled_idx)} samples...")
        best_final_state, _ = train_with_validation(
            model, updated_labeled_dataset, epochs_per_round, criterion_cls, criterion_recon, optimizer, scheduler, device
        )
        model.load_state_dict(best_final_state)

        # --- 1 & 2. Evaluate performance on test set and Save Test Predictions ---
        final_probs, final_labels = evaluate_multimodal(model, test_loader, device)
        
        # Save Test Predictions
        test_df = pd.DataFrame({
            'SMILES': smiles_test,
            'True_Label': y_test,
            'Predicted_Prob': final_probs[:, 1]
        })
        test_preds_path = os.path.join(output_dir, f"al_{strategy_name}_round_{i+1}_test_predictions.csv")
        test_df.to_csv(test_preds_path, index=False)
        
        # Compute Metrics (External)
        metrics = calculate_classification_metrics(final_labels, final_probs[:, 1], k=100)
        test_metrics_history.append(metrics)
        
        # Compute Metrics (Internal)
        int_probs, int_labels = evaluate_multimodal(model, internal_test_loader, device)
        int_metrics = calculate_classification_metrics(int_labels, int_probs[:, 1], k=100)
        internal_test_metrics_history.append(int_metrics)
        
        # Calculate enrichment metrics ON EXTERNAL TEST SET
        enrichment_metrics = calculate_enrichment_hits(final_labels, final_probs[:, 1], fractions=[0.01, 0.05, 0.1])
        enrichment_history.append(enrichment_metrics)
        
        print(f"    - Ext AUPRC: {metrics['auprc']:.4f}, Int AUPRC: {int_metrics['auprc']:.4f}, Hits: {cumulative_hits_history[-1]}")
        
        # Compute and save UMAP plot for this round
        try:
            compute_and_save_umap(model, desc_train, maccs_train, smiles_train, graphs_train, y_train,
                                 labeled_idx, pool_idx, query_idx_abs, strategy_name, i+1, 
                                 output_dir, device)
        except Exception as e:
            print(f"    ⚠️  Could not save UMAP plot: {e}")

    # Save the final model after the last round
    torch.save(model.state_dict(), os.path.join(output_dir, f"al_{strategy_name}_last_round.pth"))

    return test_metrics_history, cumulative_hits_history, pd.concat(acquired_samples_list, ignore_index=True), enrichment_history, internal_test_metrics_history


def active_learning_multimodal(
    desc_train, maccs_train, smiles_train, graphs_train, y_train,
    desc_test, maccs_test, smiles_test, graphs_test, y_test,
    desc_internal, maccs_internal, smiles_internal, graphs_internal, y_internal, # Added Internal Test
    initial_idx, pool_idx,
    output_dir,
    n_queries=7,
    n_instances=100,
    epochs_per_round=20
):
    """Run active learning experiment with Multimodal, starting from a pre-defined split."""
    
    strategies = [
        'random', 'uncertainty', 'diversity', 'confidence_toxic'
    ]
    
    # Track performance and hits across all strategies
    test_performance_auprc = {s: [] for s in strategies} # External
    internal_test_performance_auprc = {s: [] for s in strategies} # Internal
    
    test_metrics_full = {s: [] for s in strategies} 
    cumulative_hits = {s: [] for s in strategies}
    enrichment_performance = {s: [] for s in strategies}
    
    # Dataframe to store all acquired samples for all strategies
    all_acquired_samples_df = pd.DataFrame()

    print(f"\n📊 Initial training set size: {len(initial_idx)} samples")
    print(f"📊 Initial pool size: {len(pool_idx)} samples")
    print(f"📊 Initial hits: {y_train[initial_idx].sum()}")
    
    # Store initial samples (once)
    initial_samples_df = pd.DataFrame({
        'SMILES': smiles_train[initial_idx],
        'True_Label': y_train[initial_idx],
        'Predicted_Proba_Toxic': -1,
        'Strategy': 'Initial',
        'Round': 0
    })
    all_acquired_samples_df = pd.concat([all_acquired_samples_df, initial_samples_df], ignore_index=True)

    # Define test datasets
    test_dataset = MolecularDataset(desc_test, maccs_test, smiles_test, graphs_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    internal_dataset = MolecularDataset(desc_internal, maccs_internal, smiles_internal, graphs_internal, y_internal)
    internal_loader = DataLoader(internal_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    # Loop through each strategy and run the full experiment
    for strategy in strategies:
        print(f"\n🚀 Running Active Learning Strategy: {strategy}")
        test_metrics_hist, hits_hist, acquired_samples, enrichment_hist, internal_metrics_hist = run_al_experiment_for_strategy(
            strategy,
            desc_train, maccs_train, smiles_train, graphs_train, y_train,
            test_loader, smiles_test, y_test,
            internal_loader, # Pass internal loader
            initial_idx, pool_idx,
            os.path.join(output_dir, strategy),
            n_queries, n_instances, epochs_per_round,
            device
        )
        
        test_metrics_full[strategy] = test_metrics_hist
        # Extract AUPRC for plotting compatibility
        test_performance_auprc[strategy] = [m['auprc'] for m in test_metrics_hist]
        internal_test_performance_auprc[strategy] = [m['auprc'] for m in internal_metrics_hist]
        
        cumulative_hits[strategy] = hits_hist
        enrichment_performance[strategy] = enrichment_hist
        all_acquired_samples_df = pd.concat([all_acquired_samples_df, acquired_samples], ignore_index=True)

    # Calculate sample sizes for plotting
    sample_sizes = [len(initial_idx) + i * n_instances for i in range(n_queries + 1)]

    # Save all acquired samples to a single CSV
    acquired_samples_csv_path = os.path.join(output_dir, "active_learning_acquired_samples.csv")
    all_acquired_samples_df.to_csv(acquired_samples_csv_path, index=False)
    print(f"\n✅ Saved all acquired samples across all strategies to {acquired_samples_csv_path}")

    return test_performance_auprc, test_metrics_full, cumulative_hits, enrichment_performance, sample_sizes, internal_test_performance_auprc


# === 10. Plotting Function ===
def plot_learning_curves_separate(test_perf, hit_rate_perf, sample_sizes, total_train_size, output_dir):
    """Plots separate learning curves for test AUPRC and Cumulative Hits."""
    os.makedirs(output_dir, exist_ok=True)
    
    percent_of_train = [100.0 * s / total_train_size for s in sample_sizes]
    
    # Plot 1: Test AUPRC
    plt.figure(figsize=(6, 4))
    for strategy, scores in test_perf.items():
        # Ensure scores list has the same length as percent_of_train
        if len(scores) == len(percent_of_train):
            plt.plot(percent_of_train, scores, marker='o', linestyle='-', label=strategy.capitalize())
    plt.xlabel('Percentage of Training Data Used (%)', fontsize=12, fontweight='bold', style='italic')
    plt.ylabel('Test AUPRC', fontsize=12, fontweight='bold', style='italic')
    plt.title('Active Learning: Test AUPRC', fontsize=12, fontweight='bold', style='italic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "al_test_auprc_curves.svg"), format='svg', dpi=300)
    plt.close()

    # Plot 2: Cumulative Hits
    plt.figure(figsize=(6, 4))
    for strategy, scores in hit_rate_perf.items():
        # Ensure scores list has the same length as percent_of_train
        if len(scores) == len(percent_of_train):
            plt.plot(percent_of_train, scores, marker='o', linestyle='-', label=strategy.capitalize())
    plt.xlabel('Percentage of Training Data Used (%)', fontsize=12, fontweight='bold', style='italic')
    plt.ylabel('Cumulative Toxic Hits Found', fontsize=12, fontweight='bold', style='italic')
    plt.title('Active Learning: Cumulative Hits', fontsize=12, fontweight='bold', style='italic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "al_cumulative_hits_curves.svg"), format='svg', dpi=300)
    plt.close()
    
    print(f"✅ Saved active learning plots to {output_dir}")


# === 11. Main Function ===
def main(random_seed=0, folder="parp"):
    print(f"\n{'='*80}")
    print(f"🚀 Starting Active Learning with Multimodal")
    print(f"SEED: {random_seed}")
    print(f"FOLDER: {folder}")
    print(f"{'='*80}\n")
    
    # Load data
    print("📂 Loading data...")
    output_dir = os.path.join(folder, f"al_{random_seed}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: '{output_dir}'")
    train = pd.read_csv(os.path.join(folder, 'train.csv'))
    test  = pd.read_csv(os.path.join(folder, 'test.csv'))
    y_train = train['Class'].values
    y_test  =  test['Class'].values
    smiles_train = train['canonical_smiles'].values
    smiles_test  = test['canonical_smiles'].values
    print(f"✅ Loaded labels and SMILES: {len(y_train)} train, {len(y_test)} test")

    # === Pre-computed Feature Loading ===
    train_prefix = os.path.join(folder, 'train')
    test_prefix = os.path.join(folder, 'test')
    
    use_precomputed = True
    required_files = [
        f"{train_prefix}_maccs.csv", f"{test_prefix}_maccs.csv",
        f"{train_prefix}_desc.csv", f"{test_prefix}_desc.csv",
        f"{train_prefix}_graphs.pt", f"{test_prefix}_graphs.pt"
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"⚠️ Pre-computed features not found: {f}")
            use_precomputed = False
            break
            
    if use_precomputed:
        print("🚀 Loading pre-computed features from CSV/PT files...")
        try:
            train_maccs = pd.read_csv(f"{train_prefix}_maccs.csv")
            test_maccs = pd.read_csv(f"{test_prefix}_maccs.csv")
            train_desc = pd.read_csv(f"{train_prefix}_desc.csv")
            test_desc = pd.read_csv(f"{test_prefix}_desc.csv")
            
            # Load graphs
            graphs_train = torch.load(f"{train_prefix}_graphs.pt")
            graphs_test = torch.load(f"{test_prefix}_graphs.pt")
            
            print("✅ Successfully loaded all pre-computed features.")
        except Exception as e:
            print(f"❌ Error loading files: {e}. Falling back to calculation.")
            use_precomputed = False

    #if not use_precomputed:
    #    # Calculate features
    #    print("⚙️  Calculating molecular features (This may take a while)...")
    #    train_maccs = calculate_maccs(train, smiles_col="canonical_smiles")
    #    test_maccs  = calculate_maccs(test, smiles_col="canonical_smiles")
    #    train_desc  = calculate_descriptors(train, smiles_col="canonical_smiles")
    #    test_desc   = calculate_descriptors(test, smiles_col="canonical_smiles")
    #    
    #    # Calculate graphs
    #    print("⚙️  Calculating graph features...")
    #    graphs_train = [mol_to_graph(s, l) for s, l in zip(tqdm(smiles_train, desc="Train Graphs"), y_train)]
    #    graphs_test = [mol_to_graph(s, l) for s, l in zip(tqdm(smiles_test, desc="Test Graphs"), y_test)]
    #
    print(f"✅ Features ready.")

    # Handle potential NaN values from feature calculation


    desc_train, maccs_train = train_desc, train_maccs
    desc_test, maccs_test = test_desc, test_maccs
    
    print(f"\n📊 Original training samples: {len(y_train)}")
    print(f"📊 External Test samples: {len(y_test)}")
    
    # === Split original train into new train and internal test ===
    # For active learning, we typically want a large pool.
    # We will take a small portion of the training data as "Internal Test"
    test_size = 0.2
    
    # We will use indices to split so we can maintain alignment across all features
    full_indices = np.arange(len(y_train))
    
    # Stratified split to maintain class balance in internal test
    train_indices, internal_test_indices = train_test_split(
        full_indices, 
        test_size=test_size, 
        random_state=random_seed, 
        stratify=y_train
    )
    
    # Create Feature Subsets for training (pool + initial)
    desc_train_split = desc_train.iloc[train_indices].reset_index(drop=True)
    maccs_train_split = maccs_train.iloc[train_indices].reset_index(drop=True)
    smiles_train_split = smiles_train[train_indices]
    graphs_train_split = [graphs_train[i] for i in train_indices]
    y_train_split = y_train[train_indices]
    
    # Create Feature Subsets for Internal Test
    desc_internal_test = desc_train.iloc[internal_test_indices].reset_index(drop=True)
    maccs_internal_test = maccs_train.iloc[internal_test_indices].reset_index(drop=True)
    smiles_internal_test = smiles_train[internal_test_indices]
    graphs_internal_test = [graphs_train[i] for i in internal_test_indices]
    y_internal_test = y_train[internal_test_indices]
    
    print(f"📊 New Training Pool size: {len(y_train_split)}")
    print(f"📊 Internal Test size: {len(y_internal_test)}")

    # === Create the initial 300-sample split from the REDUCED training set ===
    n_initial = 300
    np.random.seed(random_seed)
    
    # We select from the NEW indices (0 to len(train_indices)-1)
    initial_idx_local = np.random.choice(len(desc_train_split), n_initial, replace=False)
    pool_idx_local = np.setdiff1d(np.arange(len(desc_train_split)), initial_idx_local)
    
    # === BASELINE: Train with initial 300 data and evaluate ===
    baseline_results = run_initial_model_evaluation(
        desc_train_split, maccs_train_split, smiles_train_split, graphs_train_split, y_train_split,
        desc_test, maccs_test, smiles_test, graphs_test, y_test,
        initial_idx_local, pool_idx_local,
        output_dir=output_dir,
        epochs=20,
        n_acquire=700
    )
    
    # Evaluate Baseline on Internal Test
    print(f"\n🔍 Evaluating Baseline on Internal Test Set...")
    # Re-load the best initial model
    baseline_model = Multimodal(desc_dim=desc_train_split.shape[1], maccs_dim=maccs_train_split.shape[1]).to(device)
    baseline_model.load_state_dict(torch.load(os.path.join(output_dir, 'baseline_initial_one_shot_model.pth')))
    
    internal_test_dataset = MolecularDataset(
        desc_internal_test, maccs_internal_test, smiles_internal_test, graphs_internal_test, y_internal_test
    )
    internal_test_loader = DataLoader(internal_test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    int_test_probs, int_test_labels = evaluate_multimodal(baseline_model, internal_test_loader, device)
    int_test_metrics = calculate_classification_metrics(int_test_labels, int_test_probs[:, 1], k=100)
    print(f"✅ Baseline Internal Test AUPRC: {int_test_metrics['auprc']:.4f}")
    
    # Add to baseline results
    baseline_results['internal_test_auprc'] = int_test_metrics['auprc']

    
    # === ACTIVE LEARNING ===
    print(f"\n{'='*80}")
    print(f"🎯 Starting Active Learning Experiments")
    print(f"{'='*80}")
    
    # Run active learning - PASS BOTH TEST SETS
    # We need to modify active_learning_multimodal to handle two test sets or run twice?
    # Modifying the full pipeline is best to track both curves
    
    test_perf, test_metrics_full, cumulative_hits, enrichment_perf, sample_sizes, internal_test_perf = active_learning_multimodal(
        desc_train_split, maccs_train_split, smiles_train_split, graphs_train_split, y_train_split,
        desc_test, maccs_test, smiles_test, graphs_test, y_test,
        desc_internal_test, maccs_internal_test, smiles_internal_test, graphs_internal_test, y_internal_test, # Pass internal test
        initial_idx_local, pool_idx_local,
        output_dir=output_dir,
        n_queries=7,
        n_instances=100,
        epochs_per_round=20
    )
    
    # Save active learning results
    df = pd.DataFrame({"n_samples": sample_sizes})
    total_train_size = len(y_train)
    df["percent_of_full_train"] = [100.0 * s / total_train_size for s in sample_sizes]
    
    # Save AUPRC (External)
    for strategy, scores in test_perf.items():
        if len(scores) == len(sample_sizes):
            df[f"{strategy}_test_auprc_diff_split_internal"] = internal_test_perf[strategy]
            df[f"{strategy}_test_auprc_external"] = scores
            
    # Save other detailed metrics (MCC, Accuracy, etc.)
    for strategy, metric_hist in test_metrics_full.items():
        if len(metric_hist) == len(sample_sizes):
            # Keys in the first dict
            metric_keys = metric_hist[0].keys()
            for key in metric_keys:
                if key == 'auprc': continue # Already added
                values = [m[key] for m in metric_hist]
                df[f"{strategy}_{key}"] = values
            
    for strategy, hits in cumulative_hits.items():
        if len(hits) == len(sample_sizes):
            df[f"{strategy}_cumulative_hits"] = hits
            
    # Add enrichment metrics
    for strategy, enrich_history in enrichment_perf.items():
        if len(enrich_history) == len(sample_sizes):
            # Keys in the first dict of the history list
            metric_keys = enrich_history[0].keys() 
            for key in metric_keys:
                # Extract this specific metric for all rounds
                values = [round_dict[key] for round_dict in enrich_history]
                df[f"{strategy}_{key}"] = values
                
    csv_path = os.path.join(output_dir, "performance_active_learning.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved active learning performance to {csv_path}")
    
    # Plot learning curves
    plot_learning_curves_separate(test_perf, cumulative_hits, sample_sizes, total_train_size, output_dir)
    # Plot Internal Test Curves
    plot_learning_curves_separate(internal_test_perf, cumulative_hits, sample_sizes, total_train_size, os.path.join(output_dir, "internal_test"))

    # === COMPARISON SUMMARY ===
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON: Baseline vs Active Learning (External Test)")
    print(f"{'='*80}\n")
    
    # ... (baseline_results print same as before, but note it is external test)
    
    print(f"\nACTIVE LEARNING (Final External Test Results):")
    for strategy in test_perf:
        if len(test_perf[strategy]) == len(sample_sizes):
            test_final_ext = test_perf[strategy][-1]
            test_final_int = internal_test_perf[strategy][-1]
            hits_final = cumulative_hits[strategy][-1]
            print(f"  {strategy.capitalize():<14s}: Ext AUPRC = {test_final_ext:.4f}, Int AUPRC = {test_final_int:.4f}, Hits = {hits_final:<2.0f}")
    
    # Create comparison table
    baseline_n_acquire = baseline_results.get('n_acquire', 700)
    baseline_total_samples = baseline_results.get('total_samples', n_initial + baseline_n_acquire)
    
    baseline_row = {
        'Method': 'Baseline',
        'Samples': baseline_total_samples,
        'Percent': 100.0 * baseline_total_samples / len(y_train),
        'Test_AUPRC_External': baseline_results.get('test_auprc_retrained', baseline_results['test_auprc']),
        'Test_AUPRC_Internal': baseline_results.get('internal_test_auprc', 0.0), # Add internal baseline
        'Cumulative_Hits': baseline_results['baseline_total_hits'],
    }
    # Add baseline enrichment specific keys
    for k, v in baseline_results.items():
        if 'hits_at_' in k or 'ef_at_' in k:
            baseline_row[k] = v
            
    comparison_data = [baseline_row]
    
    for strategy in test_perf:
        if len(test_perf[strategy]) == len(sample_sizes):
            row = {
                'Method': f'AL_{strategy.capitalize()}',
                'Samples': sample_sizes[-1],
                'Percent': 100.0 * sample_sizes[-1] / len(y_train),
                'Test_AUPRC_External': test_perf[strategy][-1],
                'Test_AUPRC_Internal': internal_test_perf[strategy][-1],
                'Cumulative_Hits': cumulative_hits[strategy][-1],
            }
            # Add final enrichment metrics
            final_enrich_metrics = enrichment_perf[strategy][-1]
            row.update(final_enrich_metrics)
            
            # Add final detailed metrics
            final_detailed = test_metrics_full[strategy][-1]
            row.update(final_detailed)
            
            comparison_data.append(row)
            
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv_path = os.path.join(output_dir, "comparison_baseline_vs_al.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\n✅ Saved comparison to {comparison_csv_path}")
    print("✅ Multimodal Experiments Completed!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    for seed in [1,2,3,4]:
        main(random_seed=seed)
