import numpy as np
from pymatgen.core import Composition

def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None


def person3_features(comp):
    comp = Composition(comp)

    elements = comp.elements
    fractions = np.array([comp.get_atomic_fraction(el) for el in elements])

    features = {}

    # -------------------------------
    # COMPOSITION (5)
    # -------------------------------
    features["num_elements"] = len(elements)

    # avoid log(0)
    safe_frac = fractions + 1e-12
    features["mixing_entropy"] = -np.sum(safe_frac * np.log(safe_frac))

    features["max_fraction"] = np.max(fractions)
    features["min_fraction"] = np.min(fractions)
    features["fraction_std"] = np.std(fractions)

    # -------------------------------
    # DIFFERENCES (5)
    # -------------------------------
    X = []
    mass = []
    radius = []
    Z = []

    for el in elements:
        val_X = safe_get(el, "X")
        val_mass = safe_get(el, "atomic_mass")
        val_radius = safe_get(el, "atomic_radius")
        val_Z = safe_get(el, "Z")

        if val_X is not None:
            X.append(val_X)
        if val_mass is not None:
            mass.append(val_mass)
        if val_radius is not None:
            radius.append(val_radius)
        if val_Z is not None:
            Z.append(val_Z)

    features["electronegativity_diff"] = max(X) - min(X) if X else 0
    features["atomic_mass_diff"] = max(mass) - min(mass) if mass else 0
    features["atomic_radius_diff"] = max(radius) - min(radius) if radius else 0
    features["Z_diff"] = max(Z) - min(Z) if Z else 0
    features["radius_std"] = np.std(radius) if radius else 0

    # -------------------------------
    # RATIOS (5)
    # -------------------------------
    metal_frac = sum(f for el, f in zip(elements, fractions) if safe_get(el, "is_metal"))
    heavy_frac = sum(f for el, f in zip(elements, fractions) if safe_get(el, "Z") and safe_get(el, "Z") > 50)
    transition_frac = sum(f for el, f in zip(elements, fractions) if safe_get(el, "is_transition_metal"))

    features["metal_fraction"] = metal_frac
    features["nonmetal_fraction"] = 1 - metal_frac
    features["heavy_element_fraction"] = heavy_frac
    features["light_element_fraction"] = 1 - heavy_frac
    features["transition_metal_fraction"] = transition_frac

    return features
