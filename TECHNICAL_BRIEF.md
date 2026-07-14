# Technical Brief: Physics-Informed Design Matrix for Rock Bit Optimization

---

## 1. Mathematical Foundations & Custom Layer Formulations

The system achieves physical grounding by routing feature tensors through two distinct non-trainable functional layers.

### A. Metallurgical Stoichiometry Engine (`MassBalanceCarbideLayer`)

Traditional models assume an arbitrary linear relationship between bulk chemical inputs. This layer isolates raw features for Carbon, Chromium, and Molybdenum, reconstructing their actual weight percentages to simulate competitive carbide precipitation reactions:

$$C_{\text{free}} = \max(C_{\text{raw}} - C_{\text{required}}, 0.0)$$

Given the characteristic mass ratios for chromium mixed carbides and molybdenum secondary carbides, the layer computes the total theoretical carbon demand:

$$C_{\text{required}} = \frac{\text{Cr}}{10.1} + \frac{\text{Mo}}{16.0}$$

To determine true phase distributions, the layer assesses the chemical limiting reactants:

* **Carbon-Limited Regime:** $\text{Mass}_{\text{carbide}} \propto C$
* **Metal-Limited Regime:** $\text{Mass}_{\text{carbide}} \propto \left( \frac{\text{Cr}}{10.1} + \frac{\text{Mo}}{16.0} \right)$

The actual calculated carbide mass is bound by the definitive thermodynamic minimum:

$$\text{Mass}_{\text{carbide}} = \min(\text{Mass}_{\text{C-lim}}, \text{Mass}_{\text{M-lim}})$$

This mass is scaled by an empirical conversion factor ($1.25$) to isolate the carbide volume fraction ($f_c$), which is clamped between $[0.0, 1.0]$.

### B. Material Degradation Curves (`ToughnessHardnessCurveLayer`)

To model tool brittleness limits, this layer captures the structural trade-off where increasing matrix hardness suppresses abrasive wear but accelerates catastrophic fracture failure. The microstructural matrix phase hardness ($H_{\text{matrix}}$) is derived via empirical solution curves:

$$H_{\text{matrix}} = (450.0 \cdot C_{\text{retained}}) + (25.0 \cdot [\text{Cr} + \text{Mo}]) + (0.85 \cdot \text{HB})$$

The plane-strain fracture toughness ($K_{\text{1C}}$) is evaluated through an exponential decay mapping:

$$K_{\text{1C}} = 85.0 \cdot e^{-3.2 \cdot f_c} \cdot \left(1.0 - \tanh\left(1.4 \cdot \frac{H_{\text{matrix}}}{1000.0}\right)\right)$$

A strict boundary safety floor guarantees $K_{\text{1C}} \ge 5.0$ to block non-physical, zero-toughness computational singular spaces.

---

## 2. Custom Physics-Informed Multi-Objective Loss

The network is constrained by a compound loss function balancing empirical data fit, metallurgical limits, and geological boundaries:

$$\mathcal{L}_{\text{total}} = w_1 \mathcal{L}_{\text{MSE}} + w_2 \mathcal{L}_{\text{metallurgical}} + w_3 \mathcal{L}_{\text{geological}}$$

### Geological Scratching Penalty ($\mathcal{L}_{\text{geological}}$)

It translates geological compositions into a customized Quartz Equivalency Index ($\text{QEI}$):

$$\text{QEI} = \text{SiO}_2 + (1.5 \cdot \text{Al}_2\text{O}_3)$$

$$\text{Wear}_{\text{min}} = \max(0.0, [\text{QEI} - 60.0] \cdot 0.05)$$


$$\text{Violation}_{\text{geo}} = \max(0.0, \text{Wear}_{\text{min}} - y_{\text{pred}})$$

$$\mathcal{L}_{\text{geological}} = \frac{1}{N} \sum_{i=1}^{N} (\text{Violation}_{\text{geo}})^2$$







