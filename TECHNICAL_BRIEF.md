# Technical Brief: Physics-Informed Neural Network (PINN) for Generative Rock Bit Design

This brief outlines the mathematical formulations, physical constraints, and geological loss functions integrated directly into the neural network architecture to optimize the structural design of industrial rock bits.

---

## 1. Core Paradigm: Hard Physics Integration

Unlike standard empirical networks, this architecture places physical laws directly into the neural network's loss and feedback loop. It ensures that predicted wear speeds and material properties conform strictly to metallurgy and mineral abrasivity physics, penalizing mathematically invalid solutions.

---

## 2. Mathematical Foundations & Custom Layer Formulations

The system achieves physical grounding by routing feature tensors through two distinct non-trainable functional layers.

### A. Metallurgical Stoichiometry Engine (`MassBalanceCarbideLayer`)

Traditional models assume an arbitrary linear relationship between bulk chemical inputs. This layer isolates raw features for Carbon, Chromium, and Molybdenum, reconstructing their actual weight percentages ($wt\%$) to simulate competitive carbide precipitation reactions:

$$C_{\text{free}} = \max(C_{\text{raw}} - C_{\text{required}}, 0.0)$$

Given the characteristic mass ratios for chromium mixed carbides ($M_7C_3$ / $M_{23}C_6 \approx 10.1$) and molybdenum secondary carbides ($Mo_2C \approx 16.0$), the layer computes the total theoretical carbon demand:

$$C_{\text{required}} = \frac{Cr_{wt\%}}{10.1} + \frac{Mo_{wt\%}}{16.0}$$

To determine true phase distributions, the layer assesses the chemical limiting reactants:

* **Carbon-Limited Regime:** $Mass_{\text{carbide}} \propto C_{wt\%}$
* **Metal-Limited Regime:** $Mass_{\text{carbide}} \propto \left(\frac{Cr_{wt\%}}{10.1} + \frac{Mo_{wt\%}}{16.0}\right)$

The actual calculated carbide mass is bound by the definitive thermodynamic minimum:

$$\text{Mass}_{\text{carbide}} = \min(\text{Mass}_{\text{C-lim}}, \text{Mass}_{\text{M-lim}})$$

This mass is scaled by an empirical conversion factor ($1.25$) to isolate the carbide volume fraction ($f_c$), which is clamped between $[0.0, 1.0]$ and appended to the tracking tensor.

### B. Material Degradation Curves (`ToughnessHardnessCurveLayer`)

To model tool brittleness limits, this layer captures the structural trade-off where increasing matrix hardness suppresses abrasive wear but accelerates catastrophic fracture failure. The microstructural matrix phase hardness ($H_{\text{matrix}}$) is derived via empirical solution curves:

$$H_{\text{matrix}} = (450.0 \cdot C_{\text{retained}}) + (25.0 \cdot [Cr_{wt\%} + Mo_{wt\%}]) + (0.85 \cdot HB_{\text{unscaled}})$$

The plane-strain fracture toughness ($K_{1C}$) is evaluated through an exponential decay mapping tracking carbide cluster density and plastic constraint saturation:

$$K_{1C} = 85.0 \cdot e^{-3.2 \cdot f_c} \cdot \left(1.0 - \tanh\left(1.4 \cdot \frac{H_{\text{matrix}}}{1000.0}\right)\right)$$

A strict boundary safety floor guarantees $K_{1C} \ge 5.0 \text{ MPa} \cdot \text{m}^{0.5}$ to block non-physical, zero-toughness computational singular spaces.

---

## 3. Custom Physics-Informed Multi-Objective Loss

The network is constrained by a compound loss function balancing empirical data fit, metallurgical limits, and geological boundaries:

$$\mathcal{L}_{\text{total}} = w_1 \mathcal{L}_{\text{MSE}} + w_2 \mathcal{L}_{\text{metallurgical}} + w_3 \mathcal{L}_{\text{geological}}$$

### Geological Scratching Penalty ($\mathcal{L}_{\text{geological}}$)

Enforces the physical law that cutting high-quartz or corundum-bearing rock mass formations must generate a baseline wear speed. It translates geological composition weight percentages into a customized Quartz Equivalency Index ($QEI$):

$$QEI = SiO_{2, wt\%} + (1.5 \cdot Al_2O_{3, wt\%})$$

$$\text{Wear}_{\text{min}} = \max(0.0, [QEI - 60.0] \cdot 0.05)$$
$$\text{Violation}_{\text{geo}} = \max(0.0, \text{Wear}_{\text{min}} - y_{\text{pred}})$$

$$\mathcal{L}_{\text{geological}} = \frac{1}{N} \sum_{i=1}^{N} (\text{Violation}_{\text{geo}})^2$$

---

## 4. References & Domain Citations

1. **Ovako AB Metallurgy Database:** Baseline structural heat-treatment, carbon distribution envelopes, and bulk hardness properties for tool-grade matrix alloys.
2. **CERCHAR Abrasivity Database:** Rock mineral composition parameters and experimental quartz scratching metrics.