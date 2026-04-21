import numpy as np
from pymatgen.core import Composition

def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None


def person1_features(comp):
    comp = Composition(comp)
    elements = comp.elements
    fractions = [comp.get_atomic_fraction(el) for el in elements]

    def weighted_avg(prop):
        vals = []
        weights = []
        for el, f in zip(elements, fractions):
            val = safe_get(el, prop)
            if val is not None:
                vals.append(val * f)
                weights.append(f)
        return sum(vals) / sum(weights) if weights else 0

    def values(prop):
        vals = []
        for el in elements:
            val = safe_get(el, prop)
            if val is not None:
                vals.append(val)
        return vals if vals else [0.0]

    features = {}

    for prop in ["Z", "atomic_mass", "X"]:
        vals = values(prop)

        features[f"avg_{prop}"] = weighted_avg(prop)
        features[f"max_{prop}"] = max(vals)
        features[f"min_{prop}"] = min(vals)
        features[f"range_{prop}"] = max(vals) - min(vals)
        features[f"std_{prop}"] = np.std(vals)

    return features
