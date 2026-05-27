# Guided Flow Matching for CellOT

This repository contains the implementation of **Guided Flow Matching** for **Cell Optimal Transport (CellOT)**. It integrates Continuous-Time Normalizing Flows (specifically, Flow Matching) with guidance mechanisms (such as Contrastive Energy Guidance, Guidance Matching, and Class-Conditional Guidance) to model and predict single-cell states and drug perturbations.

The repository is structured into two main components:
1. **Biological Data Exploration (`src/`)**: Tools for loading and exploring real single-cell perturbation datasets (like the 4i multiplexed imaging dataset).
2. **Synthetic Toy Experiments (`synthetic/`)**: A complete framework (`guided_flow` package) for training, testing, and visualizing flow matching with different target/source distributions (e.g., Circle, S-curve, Moon, 8-Gaussian, etc.) and guidance approaches.

---

## 🛠 Prerequisites

* **Conda** (Anaconda or Miniconda)
* **Python 3.11+ / 3.12+**
* **CUDA-enabled GPU** (recommended for training, though MPS also works)

---

## ⚙️ Installation & Environments

The project contains two separate Conda environments depending on the context:

### 1. Main Bio-ML Environment (Root)
For working with single-cell biological datasets (e.g., using `anndata` for 4i).
```bash
# Create the environment from the root directory
conda env create -f environment.yml

# Activate the environment
conda activate bio_ml
```

### 2. Synthetic Guided Flow Environment
For running 2D synthetic flow matching and guidance toy experiments.
```bash
# Create the environment using the synthetic environment.yml
conda env create -f synthetic/environment.yml

# Activate the environment
conda activate guided_flow_synthetic_data
```

Once the synthetic environment is activated, you can install the `guided_flow` package in editable mode:
```bash
cd synthetic
pip install -e .
```

---

## 📂 Project Structure

```diff
.
  ├── environment.yml                  # Main conda environment file (bio_ml)
  ├── environment_with_versions.yml    # Main conda environment file with pinned dependency versions
  ├── LICENSE                          # License file
  ├── README.md                        # You are here!
  ├── src/                             # Biological data and model scripts
  │   ├── data/
  │   │   └── 4i/                      # 4i single-cell multiplexed imaging data
  │   │       ├── 8h.h5ad              # H5AD dataset containing ~97,000 cells
  │   │       ├── drugs.txt            # List of drugs in the dataset
  │   │       └── features.txt         # List of single-cell features (markers)
  │   ├── read_data.py                 # Helper to load and inspect the 4i dataset
  │   └── model.py                     # Main model wrapper placeholder
  └── synthetic/                       # Synthetic flow matching framework
      ├── environment.yml              # Synthetic conda environment file (guided_flow_synthetic_data)
      ├── setup.py                     # Setup script for guided_flow package
      ├── guided_flow/                 # Core python package
      │   ├── backbone/                # MLP and Transformer wrapper models
+     │   │   ├── mlp.py               # [MODIFIED] Multi-Layer Perceptron architecture
+     │   │   ├── wrapper.py           # [MODIFIED] Model wrapper for flows and guidance
      │   │   └── transformer.py       # Transformer backbone model
      │   ├── config/                  # Configuration dataclasses for training & sampling (Tyro)
      │   │   ├── training.py          # Training configurations
      │   │   ├── sampling.py          # Sampling configurations
      │   │   └── guidance_training.py # Guidance training configurations
      │   ├── dataset/                 # Dataset utilities
      │   │   └── objective.py         # Training loss objectives
      │   ├── distributions/           # Custom synthetic samplers (Circle, 8-Gaussian, Moon, S-curve, etc.)
+     │   │   ├── gaussian.py          # [MODIFIED] Gaussian distribution sampler with J potential
      │   │   ├── base.py              # Base distribution class
      │   │   ├── circle.py            # Circle distribution sampler
      │   │   ├── moon.py              # Moon distribution sampler
      │   │   └── s_curve.py           # S-curve distribution sampler
      │   ├── flow/                    # Conditional Flow Matching (CFM) & Optimal Transport (OT-CFM)
      │   │   ├── conditional_flow_matching.py # CFM vector fields and loss
      │   │   └── optimal_transport.py # Optimal Transport sampling and paths
      │   ├── guidance/                # Contrastive Energy Guidance & Gradient Guidance
      │   │   ├── contrastive_energy.py  # Contrastive Energy Guidance (CEG)
      │   │   └── gradient_guidance.py   # Classifier/Gradient Guidance
      │   ├── train/                   # Pipelines for training CFM, CCFM, CEG, and Guidance Matching
      │   │   ├── cfm.py               # Main CFM training script
+     │   │   ├── ccfm.py              # [CREATED] Class-Conditional Flow Matching training script
      │   │   ├── value_ceg.py         # Contrastive Energy Guidance training script
      │   │   └── value_guidance_matching.py # Guidance Matching training script
      │   └── utils/                   # Evaluation metrics, KL divergence, and visualization scripts
      │       ├── kl_divergence.py     # KL divergence calculations
      │       ├── metrics.py           # Wasserstein and other metrics
      │       └── visualize.py         # Trajectory and vector field plotting utilities
      ├── logs/                        # Directory where model checkpoints and logs are saved
      ├── notebooks/                   # Jupyter notebooks for evaluating models and plotting flow trajectories
+     │   ├── fig_compare_to_class-cond_CFM.ipynb # [MODIFIED] Visualizes class-conditional comparison
      │   ├── fig.ipynb                # Notebook for generating figures
      │   ├── test_distributions.ipynb # Notebook for testing distributions
      │   └── test_unconditional.ipynb # Notebook for testing unconditional flow
      └── script/                      # Bash scripts for training the models
          ├── train_cfm.sh             # Training script for CFM
          ├── train_ceg.sh             # Training script for CEG
          └── train_guidance_matching.sh # Training script for Guidance Matching
```

---

## 🚀 Running the Code

### 1. Exploring Biological Data
To inspect the single-cell data properties, ensure you have the `bio_ml` environment activated:
```bash
python src/read_data.py
```

### 2. Synthetic Experiments
Navigate to the `synthetic/` directory, ensure the `guided_flow_synthetic_data` environment is activated, and run the training scripts:

* **Train Conditional Flow Matching (CFM):**
  ```bash
  bash script/train_cfm.sh
  ```
  *(This trains unconditional/conditional flow matching and guidance matching on several distribution pairs such as circle to s-curve, 8-gaussian to moon, and uniform to 8-gaussian)*.

* **Train Contrastive Energy Guidance (CEG):**
  ```bash
  bash script/train_ceg.sh
  ```

* **Train Guidance Matching:**
  ```bash
  bash script/train_guidance_matching.sh
  ```

### 3. Notebooks & Visualizations
Inside `synthetic/notebooks/`, you will find several Jupyter notebooks:
* [fig.ipynb](file:///Users/LS/Documents/Studium/8-SS26/Deep%20Learning%20Seminar/own%20Project/Guided-Flow-Matching-for-CellOT/synthetic/notebooks/fig.ipynb): Generates plots/figures comparing different guidance methods and baselines.
* [fig_compare_to_class-cond_CFM.ipynb](file:///Users/LS/Documents/Studium/8-SS26/Deep%20Learning%20Seminar/own%20Project/Guided-Flow-Matching-for-CellOT/synthetic/notebooks/fig_compare_to_class-cond_CFM.ipynb): Visualizes and compares flow matching variants against class-conditional baselines.
* [mc.ipynb](file:///Users/LS/Documents/Studium/8-SS26/Deep%20Learning%20Seminar/own%20Project/Guided-Flow-Matching-for-CellOT/synthetic/notebooks/mc.ipynb): Explores Monte Carlo/Markov Chain guidance compared to Contrastive Energy Guidance (CEG) and Guidance Matching.
* [test_distributions.ipynb](file:///Users/LS/Documents/Studium/8-SS26/Deep%20Learning%20Seminar/own%20Project/Guided-Flow-Matching-for-CellOT/synthetic/notebooks/test_distributions.ipynb): Sanity check of the synthetic source/target distribution samplers.
* [test_grad.ipynb](file:///Users/LS/Documents/Studium/8-SS26/Deep%20Learning%20Seminar/own%20Project/Guided-Flow-Matching-for-CellOT/synthetic/notebooks/test_grad.ipynb): Tests gradient calculations and guidance vector fields.
* [test_guidance_matching.ipynb](file:///Users/LS/Documents/Studium/8-SS26/Deep%20Learning%20Seminar/own%20Project/Guided-Flow-Matching-for-CellOT/synthetic/notebooks/test_guidance_matching.ipynb): Tests training and sampling with Guidance Matching.
* [test_mc.ipynb](file:///Users/LS/Documents/Studium/8-SS26/Deep%20Learning%20Seminar/own%20Project/Guided-Flow-Matching-for-CellOT/synthetic/notebooks/test_mc.ipynb): Evaluates Monte Carlo sampling and validation of guidance formulas.
* [test_unconditional.ipynb](file:///Users/LS/Documents/Studium/8-SS26/Deep%20Learning%20Seminar/own%20Project/Guided-Flow-Matching-for-CellOT/synthetic/notebooks/test_unconditional.ipynb): Evaluation of unconditional flow trajectories.

---

## 📝 Broad Topic: Guided Flow Matching for CellOT
In single-cell biology, predicting how cells respond to perturbations (like drugs or disease states) is framed as **Optimal Transport** (mapping control cells to treated cells). 

**Flow Matching** models continuous normalizing flows by regressing vector fields directly. This project implements **Guided Flow Matching**, which extends flow matching by adding a guidance term (e.g., derived from classifier gradients or contrastive energy functions) to control the trajectory of the matching flow, facilitating targeted cell state transitions.
