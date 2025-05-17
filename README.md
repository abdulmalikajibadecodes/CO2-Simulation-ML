# Sleipner CO2 Storage Migration Monitoring 🌊💾

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![pandas](https://img.shields.io/badge/pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-green.svg)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)

![Sleipner Platform](https://www.tgs.com/hs-fs/hubfs/Sleipner%20Subsurface%20Image%20with%20Reservoir%20-%20Web%20Size.jpg?width=1340&height=803&name=Sleipner%20Subsurface%20Image%20with%20Reservoir%20-%20Web%20Size.jpg)

## Project Overview

This project provides a comprehensive framework for analyzing and simulating **CO₂ storage** in the **Sleipner field**, one of the world's first and longest-running carbon capture and storage (CCS) projects located in the North Sea. The simulation tools enable:

- **3D visualization** of CO₂ plume development across different geological layers
- **Quantitative analysis** of plume volume, distribution, and migration patterns
- **Velocity-based saturation modeling** from time-lapse seismic data
- **Machine learning predictions** of future plume behavior and evolution

---

## Background

The **Sleipner CO₂ storage project**, operated by Equinor (formerly Statoil) in the Norwegian North Sea, began injecting CO₂ in 1996. CO₂ is captured from natural gas processing on the Sleipner platform and injected into the **Utsira Formation**, a saline aquifer approximately 1000 meters below the seabed. This pioneering project demonstrates the feasibility, safety, and effectiveness of industrial-scale CO₂ storage.

The geological structure of the Utsira Formation includes:
- Nine sandstone layers (*L1-L9*)
- Thin shale barriers between layers acting as partial flow barriers
- A top seal (caprock) preventing upward migration of CO₂

More than **20 million tonnes of CO₂** have been successfully stored at Sleipner, making it one of the most significant and well-documented CCS reference projects worldwide.

---

## Data & Tools

### Data Sources

The simulation uses data from the [CO2 Data Share consortium](https://co2datashare.org/), including:

- **Plume boundaries** - Outlines of CO₂ presence in each layer derived from seismic monitoring
- **Seismic data** - Pre-injection and post-injection velocity maps showing changes due to CO₂
- **Well logs** - Data from wells 15/9-A-16 (injection well) and 15/9-13 (exploration well)
- **Grid information** - Geological grid specifications for the simulation model (GRDECL format)
- **Reference model** - Benchmark simulation model released by Equinor

### Key Dependencies

```
numpy >= 1.20.0
scipy >= 1.7.0
pandas >= 1.3.0
matplotlib >= 3.4.0
scikit-learn >= 1.0.0
seaborn >= 0.11.0
```

---

## Core Functionality

The `SleipnerProcessor` class provides the following capabilities:

### Data Loading

```python
# Initialize processor
processor = SleipnerProcessor(data_dir="data/")

# Load grid, plume boundaries, and well data
processor.load_grid_data("Sleipner_Reference_Model.grdecl")
processor.load_plume_boundary("L9_plume_boundaries.pdf", "L9")
processor.load_well_data("15-9-A16_las_data.pdf", well_name="15/9-A-16", data_type="log")
processor.load_well_data("15-9-A16_position_data.pdf", well_name="15/9-A-16", data_type="position")

# Load velocity maps
processor.load_velocity_map("1994_Top_Base_Utsira_Fm_Trend.pdf", 
                           "pre_TopUtsira_BaseUtsira", "pre-injection")
processor.load_velocity_map("Ext_TopUtsiraFm-Ref7.pdf", 
                           "post_TopUtsira_Ref7", "post-injection")
```

### Analysis Tools

```python
# Create 3D binary representation of CO₂ presence
processor.create_binary_plume_grid()

# Calculate velocity differences between pre and post-injection
processor.calculate_velocity_difference("pre_TopUtsira_BaseUtsira", "post_TopUtsira_Ref7")

# Convert velocity differences to CO₂ saturation estimates
processor.convert_velocity_to_saturation(method="linear")

# Calculate plume metrics
metrics = processor.calculate_plume_metrics()
print(f"CO₂ volume: {metrics['co2_volume_m3'] / 1e6:.2f} million m³")
print(f"CO₂ mass: {metrics['co2_mass_tonnes'] / 1e6:.2f} million tonnes")

# Analyze specific layer
processor.analyze_layer_velocities("L9", "pre_TopUtsira_BaseUtsira", "post_TopUtsira_Ref7")
```

### Machine Learning

```python
# Prepare features for ML
X_features, y_targets, feature_names = processor.prepare_ml_features()

# Train saturation model
model, metrics, importance = processor.train_saturation_model(model_type="regression_forest")

# Predict future saturation (1.5x current time)
future_saturation = processor.predict_saturation(time_factor=1.5)
```

### Visualization

```python
# Create 2D layer visualization
processor.visualize_layer(layer_idx=8, data_type="saturation", 
                         save_path="figures/layer_9_saturation.png")

# Create 3D visualization of the entire CO₂ plume
processor.visualize_3d_saturation(data_type="saturation", 
                                 save_path="figures/3d_saturation.png")

# Visualize plume metrics
processor.visualize_metrics(metrics=metrics, 
                          save_path="figures/plume_metrics.png")

# Specialized Layer 9 visualization
processor.visualize_layer9_comparison(save_path="figures/layer9_comparison.png")
```

---

## Workflow

The typical workflow follows these steps:

1. **Data Loading** - Import all necessary data sources
2. **Create Binary Grid** - Generate 3D representation of the CO₂ plume
3. **Velocity Analysis** - Calculate and normalize velocity differences
4. **Convert to Saturation** - Transform velocity data to CO₂ saturation estimates
5. **Layer-Specific Analysis** - Perform detailed analysis of individual layers
6. **Machine Learning** - Train models and predict future states
7. **Visualization & Reporting** - Generate visualizations and compile metrics

Execute the entire workflow at once:

```python
# Run full simulation with output to specified directory
results = processor.run_simulation(output_dir="results/")
```

---

## Example Results

### Layer Visualizations

Visualization of CO₂ saturation in Layer 9:

![Layer Visualization Example](https://raw.githubusercontent.com/username/sleipner-co2/main/examples/layer_9_saturation.png)

### 3D Plume Visualization

Three-dimensional representation of the entire CO₂ plume across all layers:

![3D Plume Example](https://raw.githubusercontent.com/username/sleipner-co2/main/examples/3d_saturation.png)

### Metrics Visualization

Distribution of CO₂ across different geological layers and overall metrics:

![Metrics Example](https://raw.githubusercontent.com/username/sleipner-co2/main/examples/plume_metrics.png)

---

## 🔍 Scientific Insights

This project enables several key scientific investigations:

- **Plume Migration Patterns**: Track how CO₂ moves laterally and vertically through the storage formation
- **Layer Communication**: Identify where CO₂ migrates through shale barriers between layers
- **Saturation Development**: Quantify CO₂ concentration in different parts of the reservoir
- **Long-term Forecasting**: Predict future plume development based on historical trends

---

## Getting Started

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sleipner-co2-simulation.git
   cd sleipner-co2-simulation
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation

```python
from sleipner_processor import SleipnerProcessor

# Initialize the processor
processor = SleipnerProcessor("path/to/data")

# Run the simulation
processor.run_simulation(output_dir="results")
```

---

## License & Acknowledgments

This project is available under the [MIT License](LICENSE).

This work uses data from the [CO2 Data Share consortium](https://co2datashare.org/), which provides access to Sleipner CO₂ storage monitoring data. We gratefully acknowledge Equinor and the CO₂ Data Share partners for making this valuable data available to the research community.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## References

1. **Chadwick, R. A., et al. (2010)**. *Quantitative analysis of time-lapse seismic monitoring data at the Sleipner CO2 storage operation*. The Leading Edge, 29(2), 170-177.

2. **Singh, V. P., et al. (2010)**. *Reservoir modeling of CO2 plume behavior calibrated against monitoring data from Sleipner, Norway*. SPE Annual Technical Conference and Exhibition.

3. **Furre, A. K., et al. (2017)**. *20 Years of Monitoring CO2 Injection at Sleipner*. Energy Procedia, 114, 3916-3926.

4. **Eiken, O., et al. (2011)**. *Lessons learned from 14 years of CCS operations: Sleipner, In Salah and Snøhvit*. Energy Procedia, 4, 5541-5548.

5. **White, J. C., et al. (2018)**. *Sleipner CO2 securely stored deep beneath the seabed, in spite of unexpected Sleipner platform settlement*. International Journal of Greenhouse Gas Control, 79, 117-124.
