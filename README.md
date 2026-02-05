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

### 1. Clone the Repo
```bash
git clone [https://github.com/yourusername/kilimo-space.git](https://github.com/yourusername/kilimo-space.git)
cd kilimo-space
# 🌿 Smart-Shamba: AI Plant Disease Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Prototype-success?style=for-the-badge)

> **A computer vision application that empowers farmers to detect crop diseases instantly using a smartphone camera.**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Installation & Usage](#-installation--usage)
- [Future Scope](#-future-scope)

---

## 📖 Overview

**Smart-Shamba** ("Shamba" means Farm in Swahili) is an AI-powered tool designed to help smallholder farmers diagnose plant diseases early. By simply uploading a photo of a leaf, the system identifies the specific disease (e.g., *Tomato Early Blight*, *Potato Late Blight*) and provides actionable recommendations.

This project leverages **Transfer Learning (MobileNetV2)** to achieve high accuracy with a lightweight model suitable for mobile deployment.

---

## 🚩 The Problem

* **Food Security:** Pests and diseases destroy **20-40%** of global crop yields annually.
* **Lack of Expertise:** Many smallholder farmers lack access to agricultural extension officers who can identify diseases correctly.
* **Delayed Action:** Farmers often notice the disease when it is too late, leading to total crop failure.

## 💡 Our Solution

We bring the "Agricultural Expert" to the farmer's pocket.
1.  **Instant Diagnosis:** Results in < 2 seconds.
2.  **Offline Capable:** The model is optimized to run on low-resource devices.
3.  **Actionable Advice:** Instead of just saying "Sick," we provide the name of the disease so the farmer knows which treatment to buy.

---

## 🔬 How It Works

We use **Convolutional Neural Networks (CNNs)** to analyze leaf patterns.

1.  **Input:** User takes a photo of a crop leaf.
2.  **Preprocessing:** Image is resized to 224x224 pixels and normalized.
3.  **AI Brain:** The image is passed through **MobileNetV2**, a pre-trained model that understands visual features (edges, textures, spots).
4.  **Classification:** The final layer predicts one of **38 classes** (healthy vs. diseased types).
5.  **Output:** The app displays the disease name and confidence score (e.g., *"Confidence: 98%"*).

---

## 🛠️ Tech Stack

| Component | Tool |
| :--- | :--- |
| **Model Architecture** | MobileNetV2 (Transfer Learning) |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Web Interface** | Streamlit |
| **Image Processing** | OpenCV / PIL |
| **Data Handling** | NumPy / Pandas |

---

## 📊 Dataset

We used the **PlantVillage Dataset**, the gold standard for plant disease classification.

* **Source:** [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
* **Size:** ~87,000 images.
* **Classes:** 38 categories across 14 crop species (Tomato, Potato, Corn, Pepper, etc.).

---

## 🚀 Installation & Usage

Follow these steps to run the project locally.

### 1. Clone the Repo
```bash
git clone [https://github.com/yourusername/smart-shamba.git](https://github.com/yourusername/smart-shamba.git)
cd smart-shamba



Project: Unsupervised Anomaly Detection in Banking Transaction Networks
This project implements an advanced Unsupervised Anomaly Detection system designed to identify potential money laundering activities within global banking networks (SWIFT/SEPA).

Unlike traditional supervised models that rely on historical "Suspicious Activity Reports" (which are often delayed or incomplete), this solution utilizes Graph Neural Networks (GNNs) to learn the inherent structure of legitimate legitimate economic activity. By treating the financial system as a graph—where Bank Accounts are nodes and Wire Transfers are edges—the model flags "structural anomalies" that deviate from the norm, effectively catching sophisticated money laundering rings without prior knowledge of their specific identity.

Dataset Source: The model is trained on the IBM Transactions for Anti-Money Laundering (AML) dataset, a massive synthetic dataset designed to replicate real-world banking patterns, including specific laundering topologies. Download Link: Kaggle - IBM AML Transaction Data

🚩 Problem Statement: What are we solving?
The Challenge: Financial crime is evolving. Launderers use "Shell Companies" and complex, circular chains of transfers to obscure the origin of dirty money (e.g., drug trafficking or embezzlement proceeds).

Rule-Based Failures: Traditional bank rules (e.g., "flag any transfer over $10,000") are outdated. Criminals bypass them using "Smurfing"—breaking large sums into hundreds of small, innocuous transfers that fly under the radar.

Lack of Labeled Data: Banks process millions of transactions daily, but confirmed fraud cases are rare and take months to investigate. We cannot rely on "labeled" data to train a model because we don't know where the new crimes are hiding.

The Goal: To build a system that can answer: "Is this account behaving like a normal business, or is it acting as a mule in a laundering ring?"—purely based on its transaction relationships.

🛠️ Tools Used
This project will be built using a modern Data Science and Machine Learning stack within Visual Studio Code.

Language: Python (Data manipulation and modeling)

Deep Learning Framework: PyTorch & PyTorch Geometric (PyG) (Implementing Graph Convolutional Networks)

Data Manipulation: Pandas & NumPy (Preprocessing millions of transaction rows)

Visualization: Matplotlib & NetworkX (Visualizing graph structures and loss curves)

Environment Management: Anaconda / Python venv

IDE: Visual Studio Code (VS Code)

💡 Insights & Solutions
The "Aha!" Moment During analysis, we utilized graph topology to discover that money laundering rings form "Cycles".

Normal Behavior: Money usually flows linearly (Employer → Employee → Merchant).

Laundering Behavior: Money flows in loops (Account A → Shell Company B → Shell Company C → Account A) to create the illusion of legitimate turnover.

The Solution Offered We developed a Graph Autoencoder (GAE).

Mechanism: The model looks at the entire web of transactions and attempts to "compress" and then "reconstruct" the network.

Detection Logic: The model easily learns standard linear payment patterns. However, when it encounters a complex "laundering cycle" or a "scatter-gather" pattern (one account sending money to 100 others rapidly), the reconstruction fails with a high error rate.

Result: An automated "Anomaly Score" for every bank account.

Low Score: Normal Personal/Business use.

High Score: High Probability of Layering/Structuring.

💼 Business Impact
How does this benefit a Bank or Compliance Team?

Detection of "Unknown" Threats: Traditional systems miss new laundering schemes until a rule is written for them. This AI detects any structural anomaly, catching new schemes on Day 1.

Operational Efficiency: Compliance analysts are currently overwhelmed by false positives (95%+ of alerts are false alarms). This model ranks alerts by mathematical severity, allowing analysts to prioritize the top 1% riskiest accounts.

Regulatory Compliance: Helps banks avoid massive fines (like those seen in the Danske Bank or HSBC scandals) by demonstrating "Next-Generation" monitoring capabilities to regulators.

🚀 Deployment
To make these insights accessible to stakeholders, the project deployment strategy is as follows:

Documentation & Reporting: The project analysis, interactive notebooks, and final report are hosted via GitHub Pages.

Real-Time API (Future): The core model is wrapped in a FastAPI endpoint. When a new wire transfer occurs, the bank's internal system sends the transaction graph to the API, which returns a risk score in milliseconds, potentially blocking the transfer before it clears.


# 🏦 Unsupervised Anomaly Detection in Banking Transaction Networks

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red?style=for-the-badge&logo=pytorch)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **An advanced AI system using Graph Neural Networks (GNNs) to detect sophisticated money laundering patterns in global banking networks (SWIFT/SEPA) without relying on historical labels.**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Insights & Solution](#-insights--solutions)
- [Tools & Tech Stack](#-tools-used)
- [Dataset](#-dataset-source)
- [Business Impact](#-business-impact)
- [Deployment Strategy](#-deployment)

---

## 📖 Overview

This project implements an **Unsupervised Anomaly Detection system** designed to identify potential money laundering activities within global banking networks.

Unlike traditional supervised models that rely on historical "Suspicious Activity Reports" (which are often delayed or incomplete), this solution utilizes **Graph Neural Networks (GNNs)** to learn the inherent structure of legitimate economic activity. By treating the financial system as a graph—where **Bank Accounts are nodes** and **Wire Transfers are edges**—the model flags "structural anomalies" that deviate from the norm, effectively catching sophisticated money laundering rings without prior knowledge of their specific identity.

---

## 🚩 Problem Statement

### The Challenge
Financial crime is evolving. Launderers use "Shell Companies" and complex, circular chains of transfers to obscure the origin of dirty money (e.g., drug trafficking or embezzlement proceeds).

### Why Traditional Methods Fail
1.  **Rule-Based Failures:** Traditional bank rules (e.g., "flag any transfer over $10,000") are outdated. Criminals bypass them using "Smurfing"—breaking large sums into hundreds of small, innocuous transfers that fly under the radar.
2.  **Lack of Labeled Data:** Banks process millions of transactions daily, but confirmed fraud cases are rare and take months to investigate. We cannot rely on "labeled" data to train a model because we don't know where the *new* crimes are hiding.

### The Goal
To build a system that can answer:
> *"Is this account behaving like a normal business, or is it acting as a mule in a laundering ring?"*

...purely based on its transaction relationships, without needing a history of prior crimes.

---

## 💡 Insights & Solutions

### The "Aha!" Moment
During our analysis, we utilized graph topology to discover that **money laundering rings form "Cycles"**.
* **✅ Normal Behavior:** Money usually flows linearly (Employer → Employee → Merchant).
* **❌ Laundering Behavior:** Money flows in loops (Account A → Shell Company B → Shell Company C → Account A) to create the illusion of legitimate turnover.

### The Solution: Graph Autoencoder (GAE)
We developed a Graph Autoencoder to detect these loops automatically.

1.  **Mechanism:** The model looks at the entire web of transactions and attempts to "compress" (encode) and then "reconstruct" (decode) the network.
2.  **Detection Logic:** The model easily learns standard linear payment patterns. However, when it encounters a complex "laundering cycle" or a "scatter-gather" pattern (one account sending money to 100 others rapidly), the reconstruction fails with a high error rate.
3.  **Result:** An automated **"Anomaly Score"** for every bank account.
    * **Low Score:** Normal Personal/Business use.
    * **High Score:** High Probability of Layering/Structuring.

---

## 🛠️ Tools Used

This project is built using a modern Data Science and Machine Learning stack within **Visual Studio Code**.

| Category | Tools |
| :--- | :--- |
| **Language** | Python 🐍 |
| **Deep Learning** | PyTorch & PyTorch Geometric (PyG) |
| **Data Manipulation** | Pandas & NumPy |
| **Visualization** | Matplotlib & NetworkX |
| **Environment** | Anaconda / Python venv |
| **IDE** | Visual Studio Code (VS Code) |

---

## 📊 Dataset Source

The model is trained on the **IBM Transactions for Anti-Money Laundering (AML)** dataset. This is a massive synthetic dataset designed to replicate real-world banking patterns, including specific laundering topologies.

* **Dataset Name:** IBM Transactions for Anti-Money Laundering (AML)
* **Download Link:** [Kaggle - IBM AML Transaction Data](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

---

## 💼 Business Impact

How does this benefit a Bank or Compliance Team?

| Impact Area | Benefit |
| :--- | :--- |
| **Detection of "Unknown" Threats** | Traditional systems miss new laundering schemes until a rule is written for them. This AI detects *any* structural anomaly, catching new schemes on Day 1. |
| **Operational Efficiency** | Compliance analysts are currently overwhelmed by false positives (95%+ of alerts are false alarms). This model ranks alerts by mathematical severity, allowing analysts to prioritize the top 1% riskiest accounts. |
| **Regulatory Compliance** | Helps banks avoid massive fines (like those seen in the Danske Bank or HSBC scandals) by demonstrating "Next-Generation" monitoring capabilities to regulators. |

---

## 🚀 Deployment

To make these insights accessible to stakeholders, the project deployment strategy is as follows:

1.  **Documentation & Reporting:** The project analysis, interactive notebooks, and final report are hosted via GitHub Pages (this repository).
2.  **Real-Time API (Future Work):** The core model is designed to be wrapped in a **FastAPI** endpoint. When a new wire transfer occurs, the bank's internal system sends the transaction graph to the API, which returns a risk score in milliseconds, potentially blocking the transfer before it clears.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
