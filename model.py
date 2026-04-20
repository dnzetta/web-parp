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
