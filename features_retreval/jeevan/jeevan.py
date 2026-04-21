import numpy as np
from pymatgen.core import Composition


def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None


def person2_features(comp):
    comp = Composition(comp)
    elements = comp.elements
    fractions = [comp.get_atomic_fraction(el) for el in elements]

    features = {}

    # -------------------------------
    # BLOCK FRACTIONS (4)
    # -------------------------------
    for block in ["s", "p", "d", "f"]:
        features[f"{block}_block_fraction"] = sum(
            f for el, f in zip(elements, fractions)
            if safe_get(el, "block") == block
        )

    # -------------------------------
    # VALENCE (5) → using group safely
    # -------------------------------
    valence = []
    for el in elements:
        val = safe_get(el, "group")
        if val is not None:
            valence.append(val)

    if valence:
        features["avg_valence_electrons"] = np.mean(valence)
        features["max_valence_electrons"] = max(valence)
        features["min_valence_electrons"] = min(valence)
        features["valence_range"] = max(valence) - min(valence)
        features["valence_std"] = np.std(valence)
    else:
        features.update({
            "avg_valence_electrons": 0,
            "max_valence_electrons": 0,
            "min_valence_electrons": 0,
            "valence_range": 0,
            "valence_std": 0
        })

    # -------------------------------
    # IONIZATION ENERGY (5)
    # -------------------------------
    ion = []
    for el in elements:
        val = safe_get(el, "ionization_energy")
        if val is not None:
            ion.append(val)

    if ion:
        features["avg_ionization_energy"] = np.mean(ion)
        features["max_ionization_energy"] = max(ion)
        features["min_ionization_energy"] = min(ion)
        features["ionization_range"] = max(ion) - min(ion)
        features["ionization_std"] = np.std(ion)
    else:
        features.update({
            "avg_ionization_energy": 0,
            "max_ionization_energy": 0,
            "min_ionization_energy": 0,
            "ionization_range": 0,
            "ionization_std": 0
        })

    # -------------------------------
    # ELECTRONEGATIVITY VARIANCE (1)
    # -------------------------------
    X = []
    for el in elements:
        val = safe_get(el, "X")
        if val is not None:
            X.append(val)

    features["electronegativity_variance"] = np.var(X) if X else 0

    return features
