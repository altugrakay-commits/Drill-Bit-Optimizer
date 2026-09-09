# Technical Brief: Physics-Informed PDC Drill Bit Design

## 1. Introduction

The Drill‑Bit Optimizer is a physics‑informed engineering tool that automates the design of PDC (Polycrystalline Diamond Compact) drill bits. It integrates:

- A **Physics‑Informed Neural Network (PINN)** that predicts wear rate from metallurgical and geological inputs, constrained by physical laws.
- The **IADC Fixed Cutter Classification System**, which translates a 4‑character code into design parameters.
- **Parametric CAD generation** using CadQuery, producing a realistic 3D bit.

This document details the constitutive models, loss functions, and geometry generation routines.

## 2. Physics-Informed Neural Network (PINN)

### 2.1 Input Features:

**Metallurgy (11 features):**

* Chemical composition: C, Si, Mn, Cr; Ni, Mo (mean wt%)
* Mechanical properties: Yield strength, UTS, Hardness (HB)
* Heat treatment: Temperature, Time

**Geology (18 features):**

* Oxide compositions (SiO₂, Al₂O₃, Fe₂O₃, etc.)

### 2.2 Physics Layers

#### MassBalanceCarbideLayer

Simulates carbide precipitation (Cr, Mo) using stoichiometric constraints:

C + Cr → Cr₇C₃ / Cr₂₃C₆

C + Mo → Mo₂C

* Computes carbide volume fraction and retained carbon.
* Enforces mass balance and prevents unphysical negative values.

#### ToughnessHardnessCurveLayer

Models matrix hardness and fracture toughness:

* **Matrix hardness:** $H_{\text{matrix}} = \alpha \cdot C_{\text{retained}} + \beta \cdot (Cr + Mo) + \gamma \cdot HB$

* **Fracture toughness:** $K_{1C} = K_{1C,\max} \cdot \exp(-\lambda \cdot V_f) \cdot \left(1 - \tanh(\omega \cdot H_{\text{matrix}} / 1000)\right)$

* **Carbide Volume fraction:** $V_{f}$

* **Empirical Constants:** $\omega$ and $\lambda$

### 2.3 Physics‑Informed Loss

The composite loss function penalises:

1. **Yield > UTS**: $\sigma_y > 3.45 \times HB$ (violation penalised quadratically).
2. **Contact stress > 3×Yield**: $\sigma_c > 3\sigma_y$
3. **Negative properties**: Any predicted property below zero.
4. **Matrix hardness < 90% of bulk hardness**: $H_{\text{matrix}} < 0.9 \times HB$.

The total loss is: $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_1 \mathcal{L}_{\text{met}} + \lambda_2 \mathcal{L}_{\text{geo}}$

where $\mathcal{L}_{\text{MSE}}$ is the data‑driven mean squared error, and $mathcal{L}_{\text{met}}$ and $\mathcal{L}_{\text{geo}}$ are the metallurgical and geological penalty terms.

## 3. IADC Classification Mapping

The 4‑character IADC code is decoded as follows:

| Position | Key | Mapping |
|----------|-----|---------|
| 1 | Body | `M` (Matrix), `S` (Steel), `D` (Diamond)
| 2 | Formation | `1`–`7` with UCS ranges and wear factors
| 3 | Cutter | `1`–`4` with size (mm) and count range
| 4 | Profile | `1`–`4` with cone angle and stability

These parameters directly influence CAD dimensions (e.g., slot width/depth scale with profile digit).

## 4. CAD Generation

The geometry is built from primitive operations in CadQuery:

1. **Shank** – cylindrical base with chamfer.
2. **Body** – lofted cone with flat top.
3. **Blades** – three radial extrusions on the top face.
4. **Cutters** – domed cylinders placed along blades (count from IADC).
5. **Gauge pads** – segmented blocks behind each blade.
6. **Nozzles** – central and three side holes with tall protrusions for visibility.
7. **Junk slots** – triangular grooves between blades, depth/width scaled by profile.

All dimensions are scaled by a factor $1 + 0.1 \times \text{wear\_rate}$, so higher wear produces a larger, more robust bit.

## 5. Export

- **STL** – triangular surface mesh for 3D printing or FEA (shell elements).
- **Streamlit** – web interface with download button.

## 6. Limitations & Future Work

- Volume mesh generation (tetrahedral) is not implemented due to complexity; users can generate meshes from the STL in their preferred FEA software.
- The current cutter placement does not avoid nozzle overlap; a warning is printed if overlap occurs.
- Future improvements could include:
  - Cutter back‑rake angle.
  - Spiral blades (instead of straight radial).
  - API pin connection at the bottom.

## 7. References

1. IADC (International Association of Drilling Contractors) – Fixed Cutter Bit Classification.
2. Raizer, V. (2022). *Physics‑Informed Neural Networks for Engineering Applications*.
3. Budynas, R. G., & Nisbett, J. K. (2015). *Shigley's Mechanical Engineering Design*.
4. CadQuery Documentation: [https://cadquery.readthedocs.io/](https://cadquery.readthedocs.io/)
5. PyVista Documentation: [https://docs.pyvista.org/](https://docs.pyvista.org/)