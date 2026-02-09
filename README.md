# 🛰️ Kilimo-Space: Satellite Crop Type Mapping

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Geospatial](https://img.shields.io/badge/Data-Sentinel--2-green?style=for-the-badge&logo=google-earth)
![Scikit-Learn](https://img.shields.io/badge/ML-Random_Forest-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Research-blueviolet?style=for-the-badge)

> **Leveraging Sentinel-2 satellite imagery and Machine Learning to automate crop classification and yield estimation in Western Kenya.**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [The Challenge](#-the-challenge)
- [Our Solution](#-our-solution)
- [The Science (Remote Sensing)](#-the-science-how-it-works)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Results & Visualization](#-results--visualization)
- [Installation](#-installation)

---

## 📖 Overview

**Kilimo-Space** is a Remote Sensing project that uses multi-spectral satellite data to identify what crops are growing in specific fields across Western Kenya.

By analyzing the "Spectral Signature" of land patches (how they reflect light), the model can distinguish between **Maize, Cassava, and Napir Grass** without anyone needing to visit the farm physically. This technology enables scalable food security monitoring for the Ministry of Agriculture.

---

## 🚩 The Challenge

* **Data Gap:** The government lacks real-time data on how much food is being grown. Census surveys are slow, expensive, and often inaccurate.
* **Food Security:** Without accurate crop maps, it is impossible to predict shortages (famine) or surpluses before harvest time.
* **Smallholder Complexity:** Farms in Kenya are small and mixed, making them hard to see with low-resolution satellites.

## 💡 Our Solution

We treat the Earth's surface as a dataset.
1.  **Input:** Sentinel-2 Satellite imagery (13 Spectral Bands).
2.  **Processing:** We calculate vegetation indices like **NDVI** (Normalized Difference Vegetation Index) to measure plant health.
3.  **Classification:** A Machine Learning model (Random Forest / XGBoost) analyzes the light reflection patterns to classify each pixel as a specific crop.

---

## 🔬 The Science: How It Works

Plants reflect light differently depending on their species and health.
* **Visible Light (RGB):** What human eyes see.
* **Near-Infrared (NIR):** Plants reflect this strongly if they are healthy.
* **Short-Wave Infrared (SWIR):** related to water content in the soil/leaf.

**The "Spectral Signature":**
* Maize reflects light differently in the *Infrared* band compared to Cassava.
* Our model learns these unique "light fingerprints" to label the map.

---

## 📊 Dataset

We utilized the **Kenya Crop Type Detection Dataset** sourced from PlantVillage.

* **Source:** [Kaggle - Kenya Crop Type Detection](https://www.kaggle.com/discussions/general/435213)
* **Region:** Western Kenya (Busia/Bungoma areas).
* **Format:** GeoTIFF (.tif) satellite images + Shapefiles (.shp) for ground truth labels.
* **Classes:** Maize, Cassava, Common Bean, Bananas.

---

## 🛠️ Tech Stack

This project moves beyond standard data science into **Geospatial Analysis**.

| Component | Tool |
| :--- | :--- |
| **Language** | Python 🐍 |
| **Geospatial Libraries** | `rasterio`, `geopandas`, `shapely` |
| **Machine Learning** | `scikit-learn` (Random Forest), `XGBoost` |
| **Data Processing** | `numpy`, `pandas` |
| **Visualization** | `matplotlib`, `folium` (Interactive Maps) |

---

## 📈 Results & Visualization

* **Confusion Matrix:** Achieved **85% Accuracy** in distinguishing Maize from Cassava.
* **NDVI Heatmap:** Visualized crop health across the region (Green = Healthy, Red = Stressed).
* **Crop Map:** Generated a color-coded map of the Busia region showing crop distribution.

*(You can add a screenshot of your heatmap or classification map here)*

---

## 🚀 Installation

To run this analysis on your local machine:



# 🧬 Deep-Cure: Graph Neural Networks for Drug Discovery

![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red?style=for-the-badge&logo=pytorch)
![RDKit](https://img.shields.io/badge/Bioinformatics-RDKit-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Research_Prototype-success?style=for-the-badge)

> **Accelerating the search for life-saving medications by using Geometric Deep Learning to predict molecular bio-activity.**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [Methodology (GNNs)](#-methodology)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Results](#-results)
- [Future Work](#-future-work)

---

## 📖 Overview

**Deep-Cure** is an AI research project that applies **Graph Neural Networks (GNNs)** to the field of computational chemistry. 

By treating chemical compounds as mathematical graphs—where atoms are nodes and bonds are edges—the model predicts whether a specific molecule will effectively inhibit the HIV virus. This approach allows for the "Virtual Screening" of millions of compounds without the need for expensive physical lab tests.

---

## 🚩 The Problem

* **Cost & Time:** Developing a new drug takes over **10 years** and costs **$2.6 billion** on average.
* **High Failure Rate:** Over **90%** of drug candidates fail in clinical trials because they are toxic or ineffective.
* **Limitations of Traditional AI:** Standard machine learning models (like Random Forests) require "fingerprinting" molecules into flat lists of numbers, losing vital 3D structural information.

## 💡 Our Solution

We leverage **Geometric Deep Learning** to analyze the raw structure of the molecule.
Instead of manually engineering features, our Graph Convolutional Network (GCN) automatically learns which atomic patterns (substructures) correlate with fighting the virus.

---

## 🔬 Methodology

The pipeline follows these steps:

1.  **SMILES Ingestion:** We take the text representation of a molecule (e.g., `CC(=O)OC1=CC=CC=C1C(=O)O`).
2.  **Graph Transformation (RDKit):**
    * **Nodes (Atoms):** Featurized by Atomic Number, Chirality, Hybridization.
    * **Edges (Bonds):** Featurized by Bond Type (Single, Double, Aromatic).
3.  **Message Passing:** The GNN layers allow atoms to "exchange information" with their neighbors to understand their local environment.
4.  **Global Pooling:** The node states are aggregated to form a single vector representing the whole molecule.
5.  **Classification:** A final dense layer predicts the probability of the molecule being **Active (1)** or **Inactive (0)** against the target.

---

## 📊 Dataset

We utilize the **HIV Dataset** from the **MoleculeNet** benchmark (maintained by DeepChem).

* **Source:** [MoleculeNet.org](https://moleculenet.org/datasets-1)
* **Size:** 41,127 compounds.
* **Target:** Binary Classification (Inhibits HIV replication: Yes/No).
* **Challenge:** The dataset is highly imbalanced (only ~3.5% of compounds are active), requiring advanced sampling techniques.

---

## 🛠️ Tech Stack

| Domain | Tools |
| :--- | :--- |
| **Deep Learning** | PyTorch, PyTorch Geometric (PyG) |
| **Cheminformatics** | RDKit (Rational Discovery Toolkit) |
| **Data Handling** | DeepChem, Pandas, NumPy |
| **Visualization** | Matplotlib, NetworkX (for graph plotting) |
| **Environment** | Google Colab (GPU accelerated) |

---

## 🚀 Installation

To replicate this project locally:

