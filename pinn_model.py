# Physics-Informed Neural Network (PINN) for wear rate prediction.

import tensorflow as tf
from keras import layers, models
import numpy as np
from config import metallurgy_features, geology_features

class MassBalanceCarbideLayer(layers.Layer):
    """Stoichiometric carbide prepicipation model (Cr, Mo)."""
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

        # Unscale to weight percentages (wt%).
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
    """Matrix hardness and fracture toughness (K1C)."""
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
        """Maps the structural phase properties to mechanical stress performance vectors."""
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

def build_pinn(stats_catalog):
    """Build the dual-input Keras model with physics layers."""
    input_metal = layers.Input(shape=(len(metallurgy_features),), name="metallurgy")
    input_geo = layers.Input(shape=(len(geology_features),), name="geology")

    layer_a = MassBalanceCarbideLayer(stats_catalog, name="Layer_A_Mass_Balance")
    augmented = layer_a(input_metal)
    layer_b = ToughnessHardnessCurveLayer(stats_catalog, name="Layer_B_Toughness_Curve")
    physics = layer_b(augmented)

    fused = layers.Concatenate(axis=1)([physics, input_geo])
    x = layers.Dense(128, activation="gelu")(fused)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dense(32, activation="gelu")(x)
    output = layers.Dense(1, name="wear_rate")(x)

    model = models.Model(inputs=[input_metal, input_geo], outputs=output)
    return model

class PhysicsInformedToolWearModel(tf.keras.Model):
    """Custom model wrapper that injects physics losses during training."""
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
        x_metal, x_geo = inputs[0], inputs[1]     # Sub-tensor containing structural chemical profiles and geological formations.

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
            raise ValueError("Model must be compiled first.")
        
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
        """Enforce matrix hardness ≥ 90% of bulk hardness."""
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

def train_pinn(X_metal, X_geo, y, stats_catalog, epochs=500, batch_size=16):
    """Train the PINN and return the trained model."""
    network = build_pinn(stats_catalog)
    model = PhysicsInformedToolWearModel(network, stats_catalog, metallurgy_features, geology_features)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)
    model.fit([X_metal, X_geo], y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])

    return model

def predict_wear(model, metal_row, geo_row, stats_catalog):
    """Predict wear rate for a single metallurgy+geology sample."""
    # Normalize inputs using stats_catalog.
    metal_norm = [(metal_row[0, idx] - stats_catalog[col]['mu']) / stats_catalog[col]['sig']
    for idx, col in enumerate(metallurgy_features)]
    geo_norm = [(geo_row[0, idx] - stats_catalog[col]['mu']) / stats_catalog[col]['sig']
    for idx, col in enumerate(geology_features)]

    X_metal = np.array(metal_norm).reshape(1, -1)
    X_geo = np.array(geo_norm).reshape(1, -1)

    pred = model.predict([X_metal, X_geo], verbose = 0)
    return float(pred[0, 0])