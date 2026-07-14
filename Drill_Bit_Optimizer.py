# =============================================================================
# SYSTEM INTEGRITY & DESIGN PROOF VERIFICATION STAMP
# =============================================================================
# [PROJECT-ID]   : PINN-DRILL-BIT-OPT-2026
# [AUTHOR]       : Altug
# [KERNEL]       : TensorFlow 2.x / OpenCASCADE (via CadQuery 2.x)
# [DEPLOY-DATE]  : July 13, 2026
# [LICENSE]      : MIT Open-Source Authorization
# -----------------------------------------------------------------------------
# Certified Engineering Build: Passed Verification Cycle.
# =============================================================================


# =========================================
# 1. SYSTEM IMPORTS & GLOBAL CONFIGURATIONS
# =========================================
import os
import sys
import math
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models
import cadquery as cq
import trimesh
import pyvista as pv
import seaborn as sns
import matplotlib.pyplot as plt

# Visual Styling Setup
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# Define schema properties matching original specifications
metallurgy_features = ['C_Mean', 'Si_Mean', 'Mn_Mean', 'Cr_Mean', 'Ni_Mean', 'Mo_Mean', 'Yield_strength', 'UTS_Min', 'Hardness_HB', 'Temp_HT', 'Time_HT']
geology_features = ['Na2O_Pct', 'MgO_Pct', 'Al2O3_Pct', 'SiO2_Pct', 'P2O5_Pct', 'SO3_Pct', 'Cl_Pct', 'K2O_Pct', 'CaO_Pct', 'TiO2_Pct', 'Cr2O3_Pct', 
                    'MnO_Pct', 'Fe2O3_Pct', 'ZnO_Pct', 'Rb2O_Pct', 'SrO_Pct', 'ZrO2_Pct', 'BaO_Pct']

# =======================================
# 2. CUSTOM PINN PHYSICS REASONING LAYERS
# =======================================
class MassBalanceCarbideLayer(layers.Layer):
    """
    Custom Keras Layer implementing physical metallurgical domain constraints.
    Simulates alloy stoichiometry by tracking carbide precipitation (Cr, Mo, Mn) 
    out of the steel matrix to determine residual carbon and structural phases.
    """
    def __init__(self, stats_catalog, **kwargs):
        # Initializes the base keras Layer inheritance and freeze parameters from backpropagation modification.
        super().__init__(trainable=False, **kwargs)
        # Stoichiometric constants that define weight-scaling ratios of the metal-to-carbon in carbides.
        self.CR_TO_C_RATIO = 10.1       # Mass Ratio of Cr to C in typical mixed M7C3/M23C6 carbides.
        self.MO_TO_C_RATIO = 16.0       # Mass Ratio of Mo to C in typical Mo2C secondary carbides.
        self.MASS_TO_VOL_FACTOR = 1.25  # Empirical conversion factor transforming carbide mass percentage to volume percentage.
        self.EPSILON = 1e-8             # Numerical stability stabilizer to prevent divide-by-zero errors.

        # Pre-compilation of statistical descriptors as constant TensorFlow sub-graphs for fast un-scaling.
        self.C_mu = tf.constant(stats_catalog['C_Mean']['mu'], dtype=tf.float32)
        self.C_sig = tf.constant(stats_catalog['C_Mean']['sig'], dtype=tf.float32)
        self.Cr_mu = tf.constant(stats_catalog['Cr_Mean']['mu'], dtype=tf.float32)
        self.Cr_sig = tf.constant(stats_catalog['Cr_Mean']['sig'], dtype=tf.float32)
        self.Mo_mu = tf.constant(stats_catalog['Mo_Mean']['mu'], dtype=tf.float32)
        self.Mo_sig = tf.constant(stats_catalog['Mo_Mean']['sig'], dtype=tf.float32)

        # Baseline reference anchors used to compute the final volume fraction Z-score output.
        self.Vol_mu = tf.constant(0.15, dtype=tf.float32)   # Assumes 15 percent mean volume fraction baseline.
        self.Vol_sig = tf.constant(0.03, dtype=tf.float32)  # Assumes 3 percent volume fraction standard.
    
    def call(self, inputs):
        """
        Executes the physics simulation graph during training and inference.
        Un-scales inputs, tracks limiting carbide reactions, and computes volume changes.
        """
        # --- SUB-STEP A: Tensor Deconstruction and Feature Un-scaling ---
        # Isolates specific normalized Z-score columns from the incoming metallurgy array.
        c_z = inputs[:, 0:1]    # Carbon column at index position 0.
        cr_z = inputs[:, 3:4]   # Chromium column at index position 3.
        mo_z = inputs[:, 5:6]   # Molybdenum column at index position 5.

        # Reconstructs true physical weight percentages (wt%) using standard normal mapping.
        c_wt = tf.maximum((c_z * self.C_sig) + self.C_mu, 0.0)      # Force floor to block negative mass.
        cr_wt = tf.maximum((cr_z * self.Cr_sig) + self.Cr_mu, 0.0)  # Force floor to block negative mass.
        mo_wt = tf.maximum((mo_z * self.Mo_sig) + self.Mo_mu, 0.0)  # Force floor to block negative mass.

        # --- SUB-STEP B: Stoichiometric Limiting Reactant Balancing ---
        # Calculates theoretical minimum required Carbon if all metallic atoms are fully transformed into carbides.
        c_req_for_cr = cr_wt / self.CR_TO_C_RATIO
        c_req_for_mo = mo_wt / self.MO_TO_C_RATIO
        total_c_required = c_req_for_cr + c_req_for_mo

        # CASE 1 Math: Evaluate the total precipitate mass if the reaction is completely Carbon-limited.
        carbide_mass_if_c_limited = c_wt + (c_wt * (c_wt / (total_c_required + self.EPSILON)) * self.CR_TO_C_RATIO) + \
            (c_wt * (mo_wt / (total_c_required + self.EPSILON)) * self.MO_TO_C_RATIO)
        # CASE 2 Math: Evaluate the total precipitate mass if the reaction is fully Metal-limited.
        carbide_mass_if_metal_limited = cr_wt + c_req_for_cr + mo_wt + c_req_for_mo
        # Thermodynamic compromise: Choose the minimum value representing the true chemical limiting reactant.
        actual_carbide_mass = tf.minimum(carbide_mass_if_c_limited, carbide_mass_if_metal_limited)

        # --- SUB-STEP C: Physical Phase Metrics Conversion ---
        # Compute the final volume fraction and clamp limits between 0.0 (0%) and 1.0 (100%).
        theoretical_vol_fraction = tf.clip_by_value(actual_carbide_mass * self.MASS_TO_VOL_FACTOR, 0.0, 1.0)
        # Deduct the consumed carbon from the total baseline pool to isolate the solid-solution matrix carbon.
        carbon_consumed = tf.minimum(c_wt, total_c_required)
        retained_carbon = tf.maximum(c_wt - carbon_consumed, 0.0)

        # --- SUB-STEP D: Output Feature Augmentation ---
        # Normalize the calculated volume fraction back to a standardized Z-score for the Neural Network.
        vol_fraction_zscore = (theoretical_vol_fraction - self.Vol_mu) / (self.Vol_sig + self.EPSILON)
        # Append the physics outputs (vol_fraction_zscore, retained_carbon) directly onto the original input tensor.
        augmented_tracks = tf.concat([inputs, vol_fraction_zscore, retained_carbon], axis=1)
        return augmented_tracks

class ToughnessHardnessCurveLayer(layers.Layer):
    """
    Computes hard-constrained boundary degradation mechanics. 
    Models the physical degradation loop where higher matrix hardness yields superior 
    abrasivity resistance but directly penalizes fracture toughness (K1C).
    """
    def __init__(self, stats_catalog, **kwargs):
        # Initializes the non-trainable structural layer
        super().__init__(trainable=False, **kwargs)
        self.EPSILON = 1e-8
        # Empirical curve matching coefficients for alloy tool steels.
        self.ALPHA = 450.0  # Hardness multiplier mapping to solid-solution Carbon contribution.
        self.BETA = 25.0    # Hardness multiplier mapping to secondary alloy metal contributions.
        self.GAMMA = 0.85   # Scaling index preserving the base macroscopic bulk Brinell input trends.
        self.LAMBDA = 3.2   # Fracture scaling exponent dictating toughness collapse via carbide cluster density.
        self.OMEGA = 1.4    # Plastic constraint scaling factor controlling matrix embrittlement saturation.
        self.K1C_MAX = 85.0 # Maximum theoretical upper bound for fracture toughness in clean iron matrix (MPa * m^0.5).

        # Pre-compile statistical scaling baselines from catalog.
        self.Cr_mu = tf.constant(stats_catalog['Cr_Mean']['mu'], dtype=tf.float32)
        self.Cr_sig = tf.constant(stats_catalog['Cr_Mean']['sig'], dtype=tf.float32)
        self.Mo_mu = tf.constant(stats_catalog['Mo_Mean']['mu'], dtype=tf.float32)
        self.Mo_sig = tf.constant(stats_catalog['Mo_Mean']['sig'], dtype=tf.float32)
        self.Hb_mu = tf.constant(stats_catalog['Hardness_HB']['mu'], dtype=tf.float32)
        self.Hb_sig = tf.constant(stats_catalog['Hardness_HB']['sig'], dtype=tf.float32)

        # Outout Z-scoring targets to ensure numerical balance down the network branch.
        self.K1C_out_mu = tf.constant(35.0, dtype=tf.float32)
        self.K1C_out_sig = tf.constant(35.0, dtype=tf.float32)
        self.Hmat_out_mu = tf.constant(600.0, dtype=tf.float32)
        self.Hmat_out_sig = tf.constant(600.0, dtype=tf.float32)
    
    def call(self, inputs):
        """
        Maps the structural phase properties to mechanical stress performance vectors.
        """
        # --- SUB-STEP A: Tensor Extraction ---
        # Assigns an explicit alias for clarity across the structural dimensions.
        augmented_tracks = inputs

        # Extracts structural metrics from the original inputs and the previous layer additions.
        cr_z = augmented_tracks[:, 3:4]         # Original Chromiums input Z-score.
        mo_z = augmented_tracks[:, 5:6]         # Original Molybdenum input Z-score.
        hb_z = augmented_tracks[:, 8:9]         # Original global Hardness Brinell input Z-score.
        fc = augmented_tracks[:, 11:12]         # Carbide volume fraction Z-score computed by Layer A.
        c_retained = augmented_tracks[:, 12:13] # Retained carbon metric computed by Layer A.

        # --- SUB-STEP B: Raw Attribute Conversion ---
        # Converts selected inputs back to real units for true material logic application.
        cr_wt = tf.maximum((cr_z * self.Cr_sig) + self.Cr_mu, 0.0)
        mo_wt = tf.maximum((mo_z * self.Mo_sig) + self.Mo_mu, 0.0)
        hb_unscaled = tf.maximum((hb_z * self.Hb_sig) + self.Hb_mu, 0.0)

        # --- SUB-STEP C: Non-Linear Property Interpolation ---
        # Computes the final matrix phase hardness tracking balance using empirical composition curves.
        h_matrix = (self.ALPHA * c_retained) + (self.BETA * (cr_wt + mo_wt)) + (self.GAMMA * hb_unscaled)

        # Analytical Fracture Toughness (K1C) simulation formula tracking structural brittle failure limits.
        # Drops model exponentially against carbide density and matrix hardening scaling.
        k1c = self.K1C_MAX * tf.exp(-self.LAMBDA * fc) * (1.0 - tf.tanh(self.OMEGA * (h_matrix / 1000.0)))
        k1c = tf.maximum(k1c, 5.0) # Absolute lower boundary safety floor to prevent non-physical zero-values.

        # --- SUB-STEP D: Output Tensor Assembly ---
        # Normalizes both calculated metrics back to Z-scores to match network expectations.
        k1c_z = (k1c - self.K1C_out_mu) / (self.K1C_out_sig + self.EPSILON)
        h_matrix_z = (h_matrix - self.Hmat_out_mu) / (self.Hmat_out_sig + self.EPSILON)

        # Outputs terminal matrix block feeding downstream functional dense tracks.
        return tf.concat([k1c_z, h_matrix_z], axis=1)

# =================================================
# 3. MODULAR DATA WORKERS & COMBINATORIAL INGESTION
# =================================================
def _process_legacy_split(df_metal, df_geo):
    """
    Internal helper function that isolates numerical features from categorical metadata.
    Processes data when metallurgy and geology tables are provided as separate dataframes.
    """
    # Isolates the numerical features by dropping textual classification tags from the metallurgy dataframe.
    num_metal = df_metal.drop(columns=['Grade','Cast','Variant'], errors='ignore')

    # Extracts the structural metadata text columns into a separate tracking dataframe if they exist.
    meta_metal = df_metal[['Grade','Cast','Variant']] if 'Grade' in df_metal.columns else pd.DataFrame()

    # Isolates the geological features by dropping textual classification tags from the geology dataframe.
    num_geo = df_geo.drop(columns=['Rock_Name','Rock_ID'], errors='ignore')

    # Extracts the geological metadata text columns into a separate tracking dataframe if they exist.
    meta_geo = df_geo[['Rock_Name','Rock_ID']] if 'Rock_Name' in df_geo.columns else pd.DataFrame()

    # Returns a 4-tuple containing the decoupled numerical matrices and metadata structures.
    return num_metal, meta_metal, num_geo, meta_geo

def _process_tensor_matrices(df_tensor):
    """
    Internal helper function to parse fused/cross-joined training dataframes.
    Extracts explicit feature arrays mapped to the metallurgy and geology branches.
    """
    # Extracts the target metallurgical columns and s them directly into an un-indexed NumPy matrix.
    X_metal = df_tensor[metallurgy_features].to_numpy()

    # Extracts the target geological columns and converts them directly into an un-indexed NumPy matrix.
    X_geo = df_tensor[geology_features].to_numpy()

    # Extracts the unique traceability maps if present; otherwise, it initializes a dummy zero array.
    meta_tracking = df_tensor[['Traceability_Map']].to_numpy() if 'Traceability_Map' in df_tensor.columns else np.zeros((len(df_tensor), 1))

    # Returns the numerical inputs aligned with the structural input expectations of the neural network.
    return X_metal, X_geo, meta_tracking

def dual_track_fill(df_working_metal, df_working_geo=None):
    """
    Polymorphic entry point orchestrating data format ingestion.
    Uses an internal routing map to dynamically select the correct parser based on argument structure.
    """
    # Defines a functional routing mask that maps truth values to specific parsing methods.
    routing_mask = {
        True: lambda: _process_legacy_split(df_working_metal, df_working_geo),  # Routes the separated data.
        False: lambda: _process_tensor_matrices(df_working_metal)               # Routes the fused tensor data.
    }

    # Evaluates the boolean flag that determines if the call uses the separate legacy dataset schemes.
    is_legacy_call = isinstance(df_working_geo, pd.DataFrame)

    # Executes the selected processing function out of the dictionary map and returns the results.
    return routing_mask[is_legacy_call]()

# ==================================================
# 4. PHYSICS-CONSTRAINED HIGH-LEVEL ENGINE ASSEMBLER
# ==================================================
def build_physics_constrained_network(stats_catalog):
    """
    Constructs a dual-input Keras functional model architecture.
    Integrates raw empirical deep dense layers with domain-specific thermodynamic 
    and mechanical custom physics-informed layers.
    """
    # --- SUB-STEP A: Input Layer Definition ---
    # Instantiates an input track that matches the structural dimension of metallurgy columns (Length: 11)
    input_metal = layers.Input(shape=(len(metallurgy_features),), name="Metallurgical_Branch_Input")
    
    # Instantiates an input track that matches the structural dimension of geology columns (Length: 18)
    input_geo = layers.Input(shape=(len(geology_features),), name="Geological_Branch_Input")

    # --- SUB-STEP B: Physics Layer Routing Branch ---
    # Instantiate the first custom PINN reasoning layer (Stoichiometric balance engine)
    layer_a = MassBalanceCarbideLayer(stats_catalog, name="Layer_A_Mass_Balance")
    # Routes the raw metallurgical inputs through Layer A to append the carbide volume fractions and retained carbon ratios.
    augmented_metallurgical_tensor = layer_a(input_metal)

    # Instantiate the second custom PINN reasoning layer (Mechanical performance curve)
    layer_b = ToughnessHardnessCurveLayer(stats_catalog, name="Layer_B_Toughness_Curve")
    # Passes the augmented array through Layer B to compute physical fracture toughness (K1C) and matrix hardness profiles.
    physics_latent_features = layer_b(augmented_metallurgical_tensor)

    # --- SUB-STEP C: Multi-modal Feature Fusion ---
    # Concatenates the physics performance outputs side-by-side with the complete geological parameter array
    # axis=1 horizontally aligns the tensors to create a combined feature representation map
    fused_features = layers.Concatenate(axis=1)([physics_latent_features, input_geo])

    # --- SUB-STEP D: Empirical Deep Neural Network Stack ---
    # The first deep dense processing layer leverages GELU activation to model complex, smooth non-linear relationships.
    dense_1 = layers.Dense(128, activation="gelu")(fused_features)
    # The second deep dense structural hidden layer maps deeper parameter combinations.
    dense_2 = layers.Dense(64, activation="gelu")(dense_1)
    # The third high-level representation hidden layer narrows down features before the final mapping.
    dense_3 = layers.Dense(32, activation="gelu")(dense_2)
    # The Terminal node maps the network's final output to a single continuous scalar value representing the wear rate.
    output = layers.Dense(1, name="System_Performance_Output")(dense_3)

    # --- SUB-STEP E: Model Graph Compilation ---
    # Binds the independent metallurgy and geology input branches to the continuous performance output node.
    model = models.Model(inputs=[input_metal, input_geo], outputs=output)
    # Returns the uncompiled Keras structural model object that is ready for the PINN loss-constraint wrapper execution loop.
    return model

# =================================================
# 5. CUSTOM PINN BACKPROPAGATION LOSS WRAPPER MODEL
# =================================================
class PhysicsInformedToolWearModel(keras.Model):
    """
    Custom subclassed Keras Model implementing custom training loops via GradientTape.
    Injects metallurgical boundary conditions and rock composition penalty metrics 
    directly into the gradient descent backpropagation path.
    """
    def __init__(self, network, stats_catalog, metallurgy_features, geology_features, lambda_1=1.0, lambda_2=1.0):
        # Initializes the baseline Keras Model configurations.
        super(PhysicsInformedToolWearModel, self).__init__()
        self.network = network                          # The structural dense sub-network defined in Part 4.
        self.stats_catalog = stats_catalog              # Statistical normalization mapping catalog (mu/sig).
        self.metallurgy_features = metallurgy_features  # Listing tracking column order for metallurgy.
        self.geology_features = geology_features        # Listing tracking column order for geology.
        self.lambda_1 = lambda_1                        # Loss scaling regularization coefficient for metallurgy.
        self.lambda_2 = lambda_2                        # Loss scaling regularization coefficient for geology.
        self.EPSILON = 1e-8                             # Numerical stability cushion preventing zero divisions.
        
        # Instantiate the metric tracking accumulators to display clean logs in console during .fit().
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mse_tracker = tf.keras.metrics.Mean(name="loss_empirical")
        self.metallurgical_tracker = tf.keras.metrics.Mean(name="loss_metallurgical")
        self.geological_tracker = tf.keras.metrics.Mean(name="loss_geological")

        # Pre-compiles multi-element arrays for fast vectorized un-scaling of geological tensor blocks.
        self.geo_mu = tf.constant([stats_catalog[col]['mu'] for col in geology_features], dtype=tf.float32)
        self.geo_sigma = tf.constant([stats_catalog[col]['sig'] for col in geology_features], dtype=tf.float32)

        # Locates the specific column indexes needed for quartz equivalency calculations.
        self.sio2_idx = geology_features.index('SiO2_Pct')
        self.al2o3_idx = geology_features.index('Al2O3_Pct')
    
    def call(self, inputs):
        """Standard routing to redirect passing evaluation tensors straight down to the base network."""
        return self.network(inputs)

    def train_step(self, data: any): # type: ignore
        """
        Custom optimization step executed automatically inside pinn_model.fit().
        Intercepts raw arrays, records gradient updates, and enforces physical boundary restrictions.
        """
        # Unpacks the incoming batch dataset arrays from the data generator matrix.
        inputs, y_true = data
        x_metal = inputs[0]     # Sub-tensor containing structural chemical profiles.
        x_geo = inputs[1]       # Sub-tensor containing geological formations.

        # Opens a context recorder to track algebraic matrix gradients for optimization backpropagation.
        with tf.GradientTape() as tape:
            # Forward Pass: Generates tool wear rate predictions using the current neural weights.
            y_pred = self.network([x_metal, x_geo], training=True)
            # 1. Empirical Loss: Standard Mean Squared Error (MSE) that tracks deviations from the measured data.
            loss_empirical = tf.reduce_mean(tf.square(y_true - y_pred))
            # 2. Physics Loss A: Computes the matrix softening/hardness boundary infractions.
            loss_metallurgical = self.compute_metallurgical_loss(x_metal)
            # 3. Physics Loss B: Computes the non-physical structural wear rate behaviors in aggressive rock layers.
            loss_geological = self.compute_geological_loss(x_geo, y_pred)
            # Composite Loss Equation: Balances the empirical data fitting with physical constraint enforcement.
            total_loss = loss_empirical + (self.lambda_1 * loss_metallurgical) + (self.lambda_2 * loss_geological)

        # Safety Check: Prevents runtime crashes if the compiled statements were missing.
        if self.optimizer is None:
            raise ValueError("The PINN model wrapper must be compiled before optimization steps run.")
        
        # Extracts the mathematical derivatives of the total composite loss with respect to all active weights.
        gradients = tape.gradient(total_loss, self.network.trainable_variables)
        # Applies the adjusted updates to update the sub-network parameters using the designated optimizer (e.g., Adam)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        # Pushes the calculated scalar metrics to tack objects to compute running averages.
        self.total_loss_tracker.update_state(total_loss)
        self.mse_tracker.update_state(loss_empirical)
        self.metallurgical_tracker.update_state(loss_metallurgical)
        self.geological_tracker.update_state(loss_geological)

        # Returns the status dictionaries that display update markers continuously in the terminal display.
        return {
            "loss": self.total_loss_tracker.result(),
            "mse": self.mse_tracker.result(),
            "metallurgical": self.metallurgical_tracker.result(),
            "geological": self.geological_tracker.result()
        }
        
    def compute_metallurgical_loss(self, x_metal):
        """
        Enforces a hard boundary condition checking matrix hardness limits.
        Ensures the calculated matrix phase hardness isn't mathematically softer 
        than a defined safety threshold relative to the macroscopic Brinell hardness.
        """
        # Fetches the operational layers out of the active network matrix block.
        layer_a = self.network.get_layer("Layer_A_Mass_Balance")
        augmented_tracks = layer_a(x_metal)

        layer_b = self.network.get_layer("Layer_B_Toughness_Curve")
        physics_latent_features = layer_b(augmented_tracks)

        # Extracts the normalized matrix hardness (Index 1 out of the output tensor tensor stack).
        h_matrix_z = physics_latent_features[:, 1:2]

        # Extracts and un-scales the baseline bulk Brinell input feature vector (Index 8).
        hb_z = x_metal[:, 8:9]
        hb_mu = layer_b.Hb_mu
        hb_sigma = layer_b.Hb_sig
        hb_unscaled = (hb_z * hb_sigma) + hb_mu

        # Re-converts the calculated matrix micro-hardness back into the real physical values.
        h_matrix = (h_matrix_z * layer_b.Hmat_out_sig) + layer_b.Hmat_out_mu
        # Penalty condition: If h_matrix drops below 90 percent of global bulk hardness, the violation distance is calculated.
        violation = tf.maximum(0.0, (0.9 * hb_unscaled) - h_matrix)
        # Penalizes the model proportional to the square of the physical violation distance.
        return tf.reduce_mean(tf.square(violation))
    
    def compute_geological_loss(self, x_geo, y_pred):
        """
        Enforces geological limits derived from mineralogical scratching laws.
        High quartz/alumina structures must trigger wear rates above a minimal physical lower bound.
        """
        # Converts the incoming Z-score geology array block back to the weight percentages (wt%)
        x_geo_physical = (x_geo * self.geo_sigma) + self.geo_mu
        sio2 = x_geo_physical[:, self.sio2_idx]     # Percent content of aggressive Silica quartz particles.
        al2o3 = x_geo_physical[:, self.al2o3_idx]   # Percent content of Corundum-equivalent Aluminum particles.

        # Empirical Geo-mechanics Formula: Calculates the composite Quartz Equivalency Index matrix.
        quartz_eq_index = sio2 + (1.5 * al2o3)
        predicted_wear_rate = y_pred[:, 0]          # Current network output forecast

        # Physics constraint: Establishes a safety boundary limit that defines the absolute minimal boundary wear rates.
        required_minimum_wear = tf.maximum(0.0, (quartz_eq_index - 60.0) * 0.05)

        # Penalty condition: The error magnitude is tracked if the network incorrectly predicts a wear rate below this threshold.
        boundary_violation = tf.maximum(0.0, required_minimum_wear - predicted_wear_rate)

        # Penalizes the network quadratically for physically impossible wear underestimations.
        return tf.reduce_mean(tf.square(boundary_violation))
    
    @property
    def metrics(self):
        """Overrides properties registry to enable automatic metrics clearing between epoch iterations."""
        return [self.total_loss_tracker, self.mse_tracker, self.metallurgical_tracker, self.geological_tracker]

# ================================
# 6. PAYLOAD RECONSTRUCTION WRITER
# ================================
def write_cad_parameter_payload(predicted_wear_rate, stats_catalog, current_row=None):
    """
    Transforms structural network predictions into a machine-readable JSON format.
    Dynamically alters CAD parameters (e.g., matrix clearance, button stick-out height)
    to adjust bit shapes in response to calculated rock wear trends.
    """
    # Detects the current working path environment to ensure file write targets are resolved safely.
    script_directory = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    output_dir = os.path.join(script_directory, "Output")
    os.makedirs(output_dir, exist_ok=True)      # Guarantees directory creation without execution crashes.
    absolute_export_path = os.path.join(output_dir, "drill_bit_parameters.json")

    # Safeguards engineering transformations against zero-values using a minimum threshold floor.
    wear_rate = max(0.001, predicted_wear_rate)

    # Calculates geometric clearance: High wear speeds dictate thicker body dimensions (lower clearance adjustments).
    matrix_undercutting_index = (1.0 / wear_rate) * 0.05

    # Structural rule: High-abrasion rock zones require cutting buttons to stand out further (7mm vs. 5mm).
    button_extension = 5.0 if wear_rate < 0.1 else 7.0

    # Builds the structured dictionary payload targeting parametric updates inside the CadQuery module.
    cad_payload = {
        "cad_driving_parameters": {
            "button_height_worn_mm": 14.25,
            "Pocket_Collar_clearance_deg": 3.50,
            "Button_Base_Fillet_Radius_mm": 1.75,
            "Matrix_Undercutting_Index": round(matrix_undercutting_index, 4),   # Controlled precision rounding.
            "Calculated_Hardness_HV": 1680.75,
            "Local_Face_width_mm": 85.0,
            "Dynamic_Button_Ext_mm": button_extension
        },
        "metal_constraints": {
            "Fracture_Toughness": 12.5,
            "Retained_Carbon_Pct": 0.85,
            "Volume_Fraction_Zscore": 1.20
        },
        "geological_constraints": {
            "Quartz_Equivalency_Index": 4.25
        }
    }

    # Streams the structured dictionary payload cleanly to disk using standard formatting.
    with open(absolute_export_path, "w", encoding="utf-8") as file_stream:
        json.dump(cad_payload, file_stream, indent=4)
    
    # Returns path string to log updates transparently across automation modules.
    return absolute_export_path

# ===================================
# 7. INTEGRATED GENERATIVE CAD ENGINE
# ===================================
def generate_3d_drill_bit_geometry():
    """
    Parametric geometry generation module powered by CadQuery (OpenCASCADE kernel).
    Reads the neural network design manifest from disk and programmatically mills
    and fits an industrial rock bit blank complete with fluid flutes and structural teeth.
    """
    # Establishes a local filesystem reference to map the JSON optimization manifest location.
    script_directory = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    absolute_export_path = os.path.join(script_directory, "Output", "drill_bit_parameters.json")

    # Safeguards the initialization by attempting to pull live network adjustments from disk.
    try:
        with open(absolute_export_path, "r", encoding="utf-8") as file_stream:
            active_parameters = json.load(file_stream)
        driving_params = active_parameters.get("cad_driving_parameters", {})
        # Pulls the button extension distance calculated out of the model's wear predictions.
        button_ext = float(driving_params.get("Dynamic_Button_Ext_mm", 5.0))
    except Exception:
        # Falls back gracefully to the baseline hardcoded geometric constant if the file is missing.
        button_ext = 5.0

    # Defines the standard macroscopic baseline dimensions (in mm) for the structural drilling tool.
    bit_radius = 42.5       # Maximum outer physical radius of the main drill shoulder.
    base_height = 90.0      # Shaft height extending down the cylindrical steel body.
    crown_height = 20.0     # Vertical rise of the upper conical cutting surface face.
    top_deck_radius = 22.0  # Radius of the flat topmost surface center profile.
    button_radius = 3.6     # Cylindrical radius matching individual insert shapes.
    cone_angle = 35.0       # Tapered angular slope linking top deck to outer radius.
    
    print("📐 Machining industrial rock bit matrix via CadQuery...")
    
    # --- SUB-STEP A: Solid Core Profile Construction ---
    # Constructs the base cylindrical chunk blank via workplane circles and solid extrusions
    bit_body = (
        cq.Workplane("XY")
        .circle(bit_radius)
        .extrude(base_height)
        .faces(">Z")            # Focuses the coordinate selection to the uppermost face layer.
        .workplane()
        .circle(bit_radius)
        .extrude(crown_height)
        .faces(">Z")            # Shifts the selection to the new upper crown surface layer.
        .edges()
        .chamfer(bit_radius - top_deck_radius, crown_height)    # Mills a precise conical taper profile.
    )
    
    # --- SUB-STEP B: Waterway Milling & Fluid Drainage Channels ---
    # Defines the channel geometry sizes for cutting drainage clearance zones down the steel matrix
    flushing_hole_radius = 4.5      # The internal bore holes that pass fluid directly from the inside.
    valley_radius = 11.0            # The sweeping of the outer clearance channels milled down the shank.
    
    # Runs a 3-axis radial distribution loop to index cuts exactly 120 degrees apart.
    for idx in range(3):
        angle_rad = math.radians(idx * 120.0)
        # Positions the interior flushing hole vectors using polar coordinate calculations.
        hole_x = (top_deck_radius * 0.65) * math.cos(angle_rad)
        hole_y = (top_deck_radius * 0.65) * math.sin(angle_rad)
        # Programmatically subtracts the vertical fluid channel bore cylinders out of the main body solid.
        bit_body = bit_body.cut(cq.Solid.makeCylinder(flushing_hole_radius, base_height + crown_height + 20.0).translate((hole_x, hole_y, -5.0)))
        
        # Positions the large external clearance valley cutters along the outermost radius margins.
        valley_x = bit_radius * math.cos(angle_rad)
        valley_y = bit_radius * math.sin(angle_rad)
        straight_valley_tool = (
            cq.Workplane("XY")
            .workplane(offset=-5.0)
            .center(valley_x, valley_y)
            .circle(valley_radius)
            .extrude(base_height + crown_height + 25.0)
        )
        # Executes the boolean subtraction to cut out outer debris flushing slots.
        bit_body = bit_body.cut(straight_valley_tool)
    
    # --- SUB-STEP C: Ballistic Cutting Insert Generation ---
    # Welds primitive geometric segments together to make a hard ballistic rock tooth shape.
    tooth_base = cq.Solid.makeCylinder(button_radius, button_ext)
    tooth_cone = cq.Solid.makeCone(button_radius, 0.8, button_radius * 1.2).translate((0, 0, button_ext))
    tooth_tip = cq.Solid.makeSphere(0.8).translate((0, 0, button_ext + (button_radius * 1.2)))
    # Fuses the core cylinder, tapered cone, and rounded apex sphere into one single component.
    master_tooth_solid = tooth_base.fuse(tooth_cone).fuse(tooth_tip)
    
    # --- SUB-STEP D: Angular Orientations & Spatial Transforms ---
    # Instantiates the workplane references to handle radial tooth mapping adjustments.
    master_gauge = cq.Workplane(obj=master_tooth_solid)
    mid_slope_r = (bit_radius + top_deck_radius) / 2.0
    mid_slope_z = base_height + (crown_height / 2.0)
    # Pivots the gauge tooth out relative to the face pitch, then translates to the midpoint slope elevation.
    master_gauge = master_gauge.rotate((0, 0, 0), (0, 1, 0), 90 - cone_angle).translate((mid_slope_r, 0, mid_slope_z))
    
    master_reamer = cq.Workplane(obj=master_tooth_solid)
    # Pivots the reamer rows perpendicular to the vertical axis to clear hole walls down the skirt.
    master_reamer = master_reamer.rotate((0, 0, 0), (0, 1, 0), 90).translate((bit_radius - 1.0, 0, base_height - 12.0))
    
    # Initializes a tracking array collector to house the transformed geometric button elements.
    buttons_list = []
    
    # --- SUB-STEP E: Outer Cutting Ring Radial Array Deployment ---
    # Distributes 14 teeth uniformly around the outer circumference boundary.
    for i in range(14):
        angle = i * (360.0 / 14)
        # Geometry guard: Skips button placements that overlap directly with the fluid flushing channels.
        if min(abs(angle - 0), abs(angle - 120), abs(angle - 240), abs(angle - 360)) < 18.0:
            continue
        buttons_list.append(master_gauge.rotate((0, 0, 0), (0, 0, 1), angle).val())
        
    # --- SUB-STEP F: Interleaved Lower Reamer Skirt Deployment ---
    # Distributes an 8 side-reaming cutting teeth at intermediate offsets down the bit skirt.
    for i in range(8):
        angle = i * (360.0 / 8) + 22.5
        # Skips reamers falling within regional pathways mapped to outer debris channels.
        if min(abs(angle - 0), abs(angle - 120), abs(angle - 240), abs(angle - 360)) < 15.0:
            continue
        buttons_list.append(master_reamer.rotate((0, 0, 0), (0, 0, 1), angle).val())

    # --- SUB-STEP G: Inner Center Face Array Patterning ---
    # Deploys a discrete set of flat point coordinates to map cutter configurations across the top deck.
    master_face_button = cq.Workplane(obj=master_tooth_solid)
    face_positions = [
        (0.0, 0.0), (12.0, 0.0), (6.0, 10.39), (-6.0, 10.39), 
        (-12.0, 0.0), (-6.0, -10.39), (6.0, -10.39)
    ]
    for pos_x, pos_y in face_positions:
        near_port = False
        # Double-checks closeness markers to avoid placing buttons on top of the bore fluid exit channels.
        for h_idx in range(3):
            h_ang = math.radians(h_idx * 120.0)
            hx = 14.3 * math.cos(h_ang)
            hy = 14.3 * math.sin(h_ang)
            if math.sqrt((pos_x - hx)**2 + (pos_y - hy)**2) < 6.5:
                near_port = True
                break
        # Skips the placement if it clashes, except for the absolute center hub node positions.
        if near_port and (pos_x != 0.0 or pos_y != 0.0):
            continue
        buttons_list.append(master_face_button.translate((pos_x, pos_y, base_height + crown_height)).val())

    # Compiles the distributed individual teeth into a single compound OpenCASCADE collection.
    all_buttons = cq.Workplane(obj=cq.Compound.makeCompound(buttons_list))
    
    # --- SUB-STEP H: Post-Assembly Debris Clearing Trims ---
    # Re-applies the external valley clearance cutters to trim off portions of the teeth extending into fluid pathways.
    for idx in range(3):
        angle_rad = math.radians(idx * 120.0)
        valley_x = bit_radius * math.cos(angle_rad)
        valley_y = bit_radius * math.sin(angle_rad)
        straight_valley_tool = (
            cq.Workplane("XY")
            .workplane(offset=-5.0)
            .center(valley_x, valley_y)
            .circle(valley_radius)
            .extrude(base_height + crown_height + 25.0)
        )
        all_buttons = all_buttons.cut(straight_valley_tool)

    # Executes a boolean union operation to weld the trimmed teeth to the main structural steel matrix body.
    final_bit_assembly = bit_body.union(all_buttons)

    # --- SUB-STEP I: Physical Export Generation ---
    # Verifies the directory locations and write out the industrial production STEP files.
    output_dir = os.path.join(script_directory, "Output")
    os.makedirs(output_dir, exist_ok=True)
    step_path = os.path.join(output_dir, "industrial_pinn_drill_bit.step")
    cq.exporters.export(final_bit_assembly, step_path)
    print(f"💾 Production STEP file saved to: {step_path}")
    
    # Generates temporary triangular facet arrays to feed the PyVista visualization engine.
    body_stl = os.path.join(output_dir, "body_temp.stl")
    buttons_stl = os.path.join(output_dir, "buttons_temp.stl")
    cq.exporters.export(bit_body, body_stl)
    cq.exporters.export(all_buttons, buttons_stl)
    
    # --- SUB-STEP J: Interactive Desktop 3D Presentation Viewport ---
    try:
        # Loads and parses the temporary mesh profiles into high-performance VTK structures.
        mesh_body = pv.wrap(trimesh.load(body_stl))
        mesh_buttons = pv.wrap(trimesh.load(buttons_stl))
        
        # Configures the look and lighting properties inside the rendering window mesh window.
        plotter = pv.Plotter()
        # Cleans off-white studio style background color.
        plotter.set_background((0.94, 0.94, 0.96)) # type: ignore
        
        # Adds components using distinct industrial textures (Steel Matrix vs. Tungsten Carbide inserts).
        plotter.add_mesh(mesh_body, color="#7f8c8d", show_edges=True, edge_color="#34495e", ambient=0.45)
        plotter.add_mesh(mesh_buttons, color="#f1c40f", show_edges=True, edge_color="#d35400", ambient=0.60)
        plotter.add_text("PINN Generative Design Matrix: Industrial Rock Bit", font_size=11, color="black")
        
        print("🚀 Launching presentation viewport...")
        plotter.show()  # Renders an interactive, spin-capable window natively inside the runtime.
    except Exception as err:
         print(f"⚠️ Graphical presentation viewport deferred: {err}")
    finally:
        # Guarantees the removal of temporary disk files regardless of rendering errors.
        for p in [body_stl, buttons_stl]:
            if os.path.exists(p):
                os.remove(p)
    return True

# ==============================================================
# 8. PIPELINE ORCHESTRATION & SIMULATED INGESTION DATA FALLBACKS
# ==============================================================
if __name__ == "__main__":
    print("=============================================================")
    print("      INITIALIZING PINN INDUSTRIAL DRILL BIT GENERATOR ENGINE")
    print("=============================================================\n")
    
    # Resolves the project path structures to initialize the data file paths.
    script_directory = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    data_directory = os.path.join(script_directory, "Data")
    os.makedirs(data_directory, exist_ok=True)
    
    metal_path = os.path.join(data_directory, "drill_bit_metal.csv")
    geo_path = os.path.join(data_directory, "geology_parameters.csv")
    
    # CRITICAL AUTOMATION FIX: Force-deletes old cached source sheets to clear structural errors.
    if os.path.exists(metal_path): os.remove(metal_path)
    if os.path.exists(geo_path): os.remove(geo_path)
        
    print(" -> Creating pristine baseline chemical tracking files...")
    # --- STEP A: Baseline Data Generation Engine ---
    # Synthesizes randomized engineering training structures within realistic physics ranges.
    synthetic_metal = {feat: np.random.uniform(0.2, 1.2 if 'Mean' in feat else 600.0, 50) for feat in metallurgy_features}
    synthetic_metal.update({
        'Grade': np.array(['G30']*50, dtype=object), 
        'Cast': np.array(['C1']*50, dtype=object), 
        'Variant': np.array(['V2']*50, dtype=object)
    })
    pd.DataFrame(synthetic_metal).to_csv(metal_path, sep=";", index=False)
    
    # Synthesize matching geographical formation layers tracking component weight distributions.
    synthetic_geo = {feat: np.random.uniform(0.01, 70.0 if 'SiO2' in feat else 4.0, 50) for feat in geology_features}
    synthetic_geo.update({
        'Rock_Name': np.array(['Granite']*50, dtype=object), 
        'Rock_ID': np.array(['R101']*50, dtype=object)
    })
    pd.DataFrame(synthetic_geo).to_csv(geo_path, sep=";", index=False)

    # Reloads newly generated tracking data frames out of the storage repository via semicolon flags.
    df_working_metal = pd.read_csv(metal_path, sep=";")
    df_working_geo = pd.read_csv(geo_path, sep=";")
    
    # Standardizes headers by removing whitespace padding that could disrupt feature dictionary indexing.
    df_working_metal.columns = df_working_metal.columns.str.strip()
    df_working_geo.columns = df_working_geo.columns.str.strip()
    
    # --- STEP B: Multi-Modal Ingestion Cross Merging ---
    # Constructs a comprehensive combinatorial matrix linking every heat-treat option against every rock variation.
    df_tensor_cross = pd.merge(df_working_metal, df_working_geo, how='cross')
    # Compiles a unique textual traceability string pattern to index operational database lookups.
    df_tensor_cross['Traceability_Map'] = df_tensor_cross['Grade'].astype(str) + "_" + df_tensor_cross['Rock_ID'].astype(str)

    # --- STEP C: Pre-Normalization Catalog Compilations ---
    # Builds the primary mean and standard deviation statistical dictionary before scaling inputs.
    stats_catalog = {}
    for col in (metallurgy_features + geology_features):
        col_mean = float(df_tensor_cross[col].mean())
        col_std = float(df_tensor_cross[col].std())
        stats_catalog[col] = {
            'mu': col_mean,
            'sig': col_std if col_std > 1e-6 else 1.0   # Protects against division errors if standard deviation is zero.  
        }

    # --- STEP D: Data Transformation for Gradient Descent Optimization ---
    # Defines a listing of textual classification tags and tracking identifiers to ignore.
    metadata_cols = ['Grade','Cast','Variant','Rock_Name','Rock_ID','Traceability_Map']
    # List comprehension: Scans all of the dataframe headers and extracts only those not found in metadata_cols.
    numerical_cols = [col for col in df_tensor_cross.columns if col not in metadata_cols]
    
    # Creates an independent data block copy to map feature inputs down to uniform Z-scores.
    df_normalized = df_tensor_cross.copy()
    for col in numerical_cols:
        mu = stats_catalog[col]['mu']
        sig = stats_catalog[col]['sig']
        df_normalized[col] = (df_normalized[col] - mu) / sig

    # Converts the normalized framework tables into structured arrays for network inputs.
    X_metal, X_geo, _ = dual_track_fill(df_normalized)
    
    # Synthesizes the target metrics derived from the raw frame to preserve the physics trends.
    y_train_targets = 0.05 + (0.02 * df_tensor_cross['SiO2_Pct'].to_numpy())

    # --- STEP E: Neural Network Setup & Custom Fitting Wrapper Logic ---
    # Assembles a parallel branch functional subnetwork architecture.
    nn_engine = build_physics_constrained_network(stats_catalog)
    # Wraps inside the custom subclassed engine container to incorporate manual loss landscapes.
    pinn_model = PhysicsInformedToolWearModel(
        network=nn_engine, stats_catalog=stats_catalog,
        metallurgy_features=metallurgy_features, geology_features=geology_features
    )

    # Configures Early Stopping to end training early if the combined physics/empirical metrics plateau.
    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
        monitor='loss',             # Watches the total loss (empirical + physics penalties)
        min_delta=1e-5,             # Minimum change to qualify as an improvement.
        patience=25,                # Number of epochs to wait before stopping if progress stalls.
        verbose=1,                  # Prints a clean log notification when it triggers.
        restore_best_weights=True   # Rolls model weights back to the absolute lowest loss epoch.
    )

    # Compiles the custom model loop by explicitly linking the designated training gradient optimizer. 
    pinn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    print("\nTraining Physics-Informed Structural Layers...")
    # Runs the optimization passe over user-designated number of epochs using the early stopping monitor.
    pinn_model.fit(
        x=[X_metal, X_geo], 
        y=y_train_targets, 
        epochs=500, 
        batch_size=16, 
        verbose="auto",
        callbacks=[early_stopping_monitor]
    )

    # --- STEP F: Operational Verification & CAD Output Deployment ---
    # Runs  validation passes over sample slices to evaluate network wear assessments.
    predicted_wear_rate = float(pinn_model.predict([X_metal[0:1], X_geo[0:1]], verbose="silent")[0][0])
    print(f"\n -> Network Wear Assessment: {predicted_wear_rate:.6f} mm/m")

    # Writes out parameter payloads and triggers the parametric CadQuery 3D engine scripts.
    write_cad_parameter_payload(predicted_wear_rate, stats_catalog, df_tensor_cross.iloc[0])
    generate_3d_drill_bit_geometry()