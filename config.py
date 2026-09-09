# config.py
# Centralised constants, feature lists, physics limits, and directory paths.

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Output"
MODEL_DIR = BASE_DIR / "Models"

for d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(exist_ok=True)

# Metallurgical features (input columns)
metallurgy_features = [
    'C_Mean', 'Si_Mean', 'Mn_Mean', 'Cr_Mean', 'Ni_Mean', 'Mo_Mean',
    'Yield_strength', 'UTS_Min', 'Hardness_HB', 'Temp_HT', 'Time_HT'
]

# Geological features (input columns)
geology_features = [
    'Na2O_Pct', 'MgO_Pct', 'Al2O3_Pct', 'SiO2_Pct', 'P2O5_Pct',
    'SO3_Pct', 'Cl_Pct', 'K2O_Pct', 'CaO_Pct', 'TiO2_Pct',
    'Cr2O3_Pct', 'MnO_Pct', 'Fe2O3_Pct', 'ZnO_Pct', 'Rb2O_Pct',
    'SrO_Pct', 'ZrO2_Pct', 'BaO_Pct'
]

# Physics limits for validation
PHYSICS_LIMITS = {
    "min_hardness": 200,
    "max_hardness": 700,
    "min_yield": 200,
    "max_yield": 1000
}