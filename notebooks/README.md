# 📓 Jupyter Notebooks

This directory contains interactive Jupyter notebooks demonstrating various features and use cases of the AI Space Exploration platform.

## 📚 Available Notebooks

### 1. Satellite Image Classification
**File:** `01_satellite_classification.ipynb`

Learn how to:
- Load and preprocess satellite imagery
- Train a CNN for terrain classification
- Evaluate model performance
- Visualize predictions

**Difficulty:** Beginner  
**Time:** 30 minutes

---

### 2. Exoplanet Detection
**File:** `02_exoplanet_detection.ipynb`

Discover how to:
- Process Kepler light curve data
- Build a 1D CNN + LSTM model
- Detect transit signals
- Analyze results

**Difficulty:** Intermediate  
**Time:** 45 minutes

---

### 3. Trajectory Optimization
**File:** `03_trajectory_optimization.ipynb`

Master:
- Reinforcement learning basics
- Spacecraft dynamics simulation
- DQN implementation
- Trajectory visualization

**Difficulty:** Advanced  
**Time:** 60 minutes

---

### 4. Galaxy Morphology Classification
**File:** `04_galaxy_morphology.ipynb`

Explore:
- Galaxy image processing
- Vision Transformer architecture
- Transfer learning techniques
- Morphology analysis

**Difficulty:** Intermediate  
**Time:** 40 minutes

---

### 5. Asteroid Tracking
**File:** `05_asteroid_tracking.ipynb`

Learn:
- Object tracking algorithms
- Trajectory prediction
- Risk assessment
- Real-time visualization

**Difficulty:** Intermediate  
**Time:** 35 minutes

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install jupyter notebook jupyterlab
```

### Launch Jupyter

```bash
# From project root
jupyter notebook notebooks/

# Or with JupyterLab
jupyter lab notebooks/
```

### Required Datasets

Some notebooks require datasets. Download them using:

```bash
python scripts/download_datasets.py
```

## 📊 Notebook Structure

Each notebook follows this structure:

1. **Introduction**: Overview and learning objectives
2. **Setup**: Imports and configuration
3. **Data Loading**: Load and explore data
4. **Preprocessing**: Data preparation
5. **Model Building**: Architecture definition
6. **Training**: Model training with visualization
7. **Evaluation**: Performance analysis
8. **Inference**: Making predictions
9. **Visualization**: Results visualization
10. **Conclusion**: Summary and next steps

## 💡 Tips

- **GPU Acceleration**: Some notebooks benefit from GPU. Use Google Colab if you don't have a GPU.
- **Memory**: Close unused notebooks to free memory.
- **Save Progress**: Notebooks auto-save, but manually save important work.
- **Restart Kernel**: If experiencing issues, try restarting the kernel.

## 🔗 Additional Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Google Colab](https://colab.research.google.com/)
- [Kaggle Kernels](https://www.kaggle.com/kernels)

## 🤝 Contributing

Have an idea for a new notebook? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

**Happy Learning! 🚀**
