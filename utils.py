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
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    descriptor_functions = [
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        rdMolDescriptors.CalcTPSA,
        Descriptors.NumRotatableBonds,
        Descriptors.NumAromaticRings,
        rdMolDescriptors.CalcNumAromaticCarbocycles,
        rdMolDescriptors.CalcNumAromaticHeterocycles,
        rdMolDescriptors.CalcNumSaturatedRings,
        rdMolDescriptors.CalcNumHeteroatoms,
        rdMolDescriptors.CalcNumRings,
        rdMolDescriptors.CalcNumHeavyAtoms,
    ]

    return [fn(mol) for fn in descriptor_functions]

def mol_to_graph(smiles):
    from rdkit import Chem
    import torch
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # --- Node features (ORDER MUST MATCH TRAINING) ---
    x = torch.stack([
        torch.tensor([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetChiralTag()),
            atom.GetTotalNumHs(),
            int(atom.GetHybridization()),
            atom.GetIsAromatic(),
            atom.GetMass(),
        ], dtype=torch.float)
        for atom in mol.GetAtoms()
    ])

    # --- Edge features ---
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        edge_index += [[i, j], [j, i]]

        feat = torch.tensor([
            float(bond.GetBondTypeAsDouble()),
            bond.IsInRing(),
            int(bond.GetStereo()),
            bond.GetIsConjugated(),
        ], dtype=torch.float)

        edge_attr += [feat, feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)