# Miami Hotel Chiller Energy Prediction

Machine-learning analysis of chiller energy consumption using EnergyPlus simulation data for a hotel in Miami.

## Project overview

The project studies relationships between operating conditions, weather, cooling demand, and chiller energy use. Phase 1 focuses on predicting energy consumption. Phase 2 will explore chilled-water supply temperature setpoint optimization.

## Phase 1: Energy prediction

- Nine-feature prediction model
- Linear regression and neural network approaches
- Examination of solar radiation as an input
- Separate evaluation of low-load operation

### Reported model results

| Model | Reported evaluation result |
| --- | --- |
| Linear regression | R² ≈ 0.93 |
| Neural network | R² ≈ 0.91 |
| Low-load neural network | MAE ≈ 8 kWh |

These preliminary figures summarize previous project analysis. Detailed evaluation methods and results belong in the project report.

## Status

- **Phase 1:** Chiller energy consumption prediction and model evaluation
- **Phase 2 (not yet complete):** Chilled-water temperature setpoint optimization

## Repository organization

Planned contents include Python code, a detailed report, model figures, and shareable simulation data or a sample dataset. Large CSV files may need a separate distribution method.

## Important note

This is a simulation-based analysis. Predictive accuracy alone does not demonstrate real-world energy savings from changing chiller settings.
