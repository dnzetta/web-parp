from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np

def compute_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)

from rdkit.Chem import Descriptors

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(Descriptors._descList))
    
    desc = []
    for name, func in Descriptors._descList:
        try:
            desc.append(func(mol))
        except:
            desc.append(0)
    
    return np.array(desc)

from torch_geometric.data import Data
import torch

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
            atom.GetMass(),
            atom.GetChiralTag()
        ])

    x = torch.tensor(node_features, dtype=torch.float)

    # Edges
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)