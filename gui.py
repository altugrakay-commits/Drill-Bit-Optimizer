# gui.py
# tkinter GUI for PDC Drill Bit Designer using IADC dropdowns.

import tkinter as tk
from tkinter import ttk, messagebox
from iadc_mapper import decode_iadc, IADC_BODY, IADC_FORMATION, IADC_CUTTER, IADC_PROFILE
from cad_generator import generate_pdc_components
from visualization import visualize_components
from main import run_design

def run_gui():
    root = tk.Tk()
    root.title("PDC Drill Bit Designer - IADC Classification")

    body_var = tk.StringVar(value="M")
    formation_var = tk.StringVar(value="4")
    cutter_var = tk.StringVar(value="3")
    profile_var = tk.StringVar(value="2")

    # Dropdowns
    ttk.Label(root, text="Body Type:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    body_combo = ttk.Combobox(root, textvariable=body_var,
                              values=[f"{k} - {v['name']}" for k, v in IADC_BODY.items()])
    body_combo.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="Formation:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    formation_combo = ttk.Combobox(root, textvariable=formation_var,
                                   values=[f"{k} - {v['name']}" for k, v in IADC_FORMATION.items()])
    formation_combo.grid(row=1, column=1, padx=5, pady=5)

    ttk.Label(root, text="Cutter Size:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
    cutter_combo = ttk.Combobox(root, textvariable=cutter_var,
                                values=[f"{k} - {v['label']} ({v['size_mm']}mm)" for k, v in IADC_CUTTER.items()])
    cutter_combo.grid(row=2, column=1, padx=5, pady=5)

    ttk.Label(root, text="Bit Profile:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
    profile_combo = ttk.Combobox(root, textvariable=profile_var,
                                 values=[f"{k} - {v['name']}" for k, v in IADC_PROFILE.items()])
    profile_combo.grid(row=3, column=1, padx=5, pady=5)

    # Code display
    code_label = ttk.Label(root, text="IADC Code: M431")
    code_label.grid(row=4, column=0, columnspan=2, pady=10)

    def update_code(*args):
        code = f"{body_var.get()[0]}{formation_var.get()[0]}{cutter_var.get()[0]}{profile_var.get()[0]}"
        code_label.config(text=f"IADC Code: {code}")

    body_var.trace_add('write', update_code)
    formation_var.trace_add('write', update_code)
    cutter_var.trace_add('write', update_code)
    profile_var.trace_add('write', update_code)

    def on_preview():
        code = code_label.cget("text").split()[-1]
        try:
            params = decode_iadc(code)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        components = generate_pdc_components(params, wear_rate=0.001)
        visualize_components(components, show=True)

    def on_export():
        code = code_label.cget("text").split()[-1]
        try:
            params = decode_iadc(code)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        success = run_design(params, visualize=False, export_fea=True)
        if success:
            messagebox.showinfo("Success", "Bit mesh exported.")
        else:
            messagebox.showerror("Error", "Export failed.")

    ttk.Button(root, text="Preview Bit", command=on_preview).grid(row=6, column=0, pady=5)
    ttk.Button(root, text="Generate PDC Bit", command=on_export).grid(row=6, column=1, pady=5)

    root.mainloop()