# main.py
# Entry point: command‑line interface or GUI.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
import argparse
import tensorflow as tf

from config import OUTPUT_DIR, MODEL_DIR, metallurgy_features, geology_features
from data import load_and_prepare_data, compute_stats, normalize_data, dual_track_fill
from pinn_model import train_pinn, predict_wear, build_pinn, PhysicsInformedToolWearModel
from cad_generator import generate_pdc_components
from visualization import visualize_components, export_to_fea
from iadc_mapper import decode_iadc

def run_design(iadc_params, visualize=True, export_fea=False):
    """Core design pipeline: data → PINN → CAD → export."""
    # Load and normalise data
    cross_df, _, _ = load_and_prepare_data()
    all_features = metallurgy_features + geology_features
    stats = compute_stats(cross_df, all_features)
    df_norm = normalize_data(cross_df, stats, all_features)

    X_metal, X_geo, _ = dual_track_fill(df_norm)
    y = 0.05 + 0.02 * cross_df['SiO2_Pct'].to_numpy()

    # Train or load PINN
    model_path = MODEL_DIR / "drill_pinn.weights.h5"
    if model_path.exists():
        network = build_pinn(stats)
        network.load_weights(str(model_path))
        model = PhysicsInformedToolWearModel(network, stats, metallurgy_features, geology_features)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001))
        print("✅ Loaded existing PINN model.")
    else:
        model = train_pinn(X_metal, X_geo, y, stats, epochs=100)
        model.network.save_weights(str(model_path))
        print("✅ Trained and saved new PINN model.")

    # Predict wear rate
    metal_row = X_metal[0:1]
    geo_row = X_geo[0:1]
    wear_rate = predict_wear(model, metal_row, geo_row, stats)
    wear_rate = max(0.001, wear_rate)

    # Generate components
    components = generate_pdc_components(iadc_params, wear_rate)

    if visualize:
        visualize_components(components, show=True)

    if export_fea:
        out_path_stl = OUTPUT_DIR / "pdc_bit_surface.stl"
        export_to_fea(components, str(out_path_stl))

    return components

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iadc", type=str, help="IADC code (e.g., M431)", default="M431")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--fea", action="store_true", help="Export surface STL")
    args = parser.parse_args()

    if args.gui:
        from gui import run_gui
        run_gui()
    else:
        params = decode_iadc(args.iadc)
        print("IADC Parameters:", params)
        run_design(params, visualize=True, export_fea=args.fea)

if __name__ == "__main__":
    main()