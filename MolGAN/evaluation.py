from rdkit import Chem
from MolGAN.molecular_metrics import MolecularMetrics
import numpy as np

def evaluate_generated_molecules(mols, dataset, verbose=True):
    """
    Evaluates generated molecules using validity, uniqueness, and novelty.
    Args:
        mols: List of generated RDKit molecules
        dataset: The dataset object (must have .smiles attribute for novelty)
        verbose: If True, prints metrics
    Returns:
        dict with keys: validity, uniqueness, novelty
    """
    # Valid
    validity_mask = MolecularMetrics.valid_scores(mols)
    valid_mols = [m for m, v in zip(mols, validity_mask) if v]
    validity = np.mean(validity_mask)

    # Unique
    uniqueness = MolecularMetrics.unique_total_score(valid_mols)

    # Novelty
    novelty = MolecularMetrics.novel_total_score(valid_mols, dataset)

    if verbose:
        print(f"✅ Validity:   {validity*100:.2f}%")
        print(f"✅ Uniqueness: {uniqueness*100:.2f}%")
        print(f"✅ Novelty:    {novelty*100:.2f}%")

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty
    }





