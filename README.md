# 🕵️‍♂️ Unsupervised Anti-Money Laundering (AML) Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![VS Code](https://img.shields.io/badge/Editor-VS%20Code-007ACC)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## 📖 Project Description
This project implements an advanced **Unsupervised Anomaly Detection system** designed to identify potential money laundering activities within cryptocurrency transaction networks.

Unlike traditional supervised models that rely on historical labels (which are often scarce or outdated), this solution utilizes **Graph Neural Networks (GNNs)** to learn the inherent structure of legitimate financial behavior. By treating the financial network as a graph—where accounts are nodes and transactions are edges—the model flags "structural anomalies" that deviate from the norm, effectively catching suspicious actors without prior knowledge of their specific identity.

**Dataset Source:**
The model is trained on the **Elliptic Data Set**, a sub-graph of the Bitcoin blockchain containing over 200,000 transactions.
* **Download Link:** [Kaggle - Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

---

## 🚩 Problem Statement: What are we solving?
**The Challenge:**
Financial crime is becoming increasingly sophisticated. Money launderers use complex, layered chains of transfers ("smurfing" or "layering") to obscure the illicit origin of funds.
1.  **Rule-Based Failures:** Traditional "If/Then" rules (e.g., "flag transactions over $10k") generate too many false positives and are easily bypassed by criminals splitting amounts.
2.  **Lack of Labeled Data:** In the real world, banks do not have a pre-labeled list of all criminals. Most financial data is unlabeled, making standard supervised learning impossible to deploy effectively for new, unknown threats.

**The Goal:**
To build a system that can answer: *"Is this transaction suspicious based on its relationship to the rest of the network, even if we've never seen this specific crime pattern before?"*

---

## 🛠️ Tools Used
This project will be  built using a modern Data Science and Machine Learning stack within **Visual Studio Code**.

* **Language:** [Python](https://www.python.org/) (Data manipulation and modeling)
* **Deep Learning Framework:** [PyTorch](https://pytorch.org/) & [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/) (Implementing Graph Convolutional Networks)
* **Data Manipulation:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) (Preprocessing 200k+ transaction rows)
* **Visualization:** [Matplotlib](https://matplotlib.org/) & [NetworkX](https://networkx.org/) (Visualizing graph structures and loss curves)
* **Environment Management:** Anaconda / Python venv
* **IDE:** Visual Studio Code (VS Code)

---

## 💡 Insights & Solutions
### The "Aha!" Moment
During the analysis, we discovered that illicit transactions (money laundering) form distinct **topological patterns** compared to licit ones. While a normal user sends money directly to an exchange or a merchant, launderers create dense, cyclical clusters to hide the money trail.

### The Solution Offered
We developed a **Graph Autoencoder (GAE)**.
1.  **Mechanism:** The model compresses the transaction network into a low-dimensional code and attempts to reconstruct it.
2.  **Detection Logic:** The model learns to reconstruct "normal" transactions perfectly. When it encounters a laundering ring, the reconstruction fails (high error rate) because the pattern is mathematically "weird."
3.  **Result:** An automated **"Risk Score"** for every transaction.
    * *Low Score:* Safe / Normal Business.
    * *High Score:* Potential Laundering / Requires Manual Review.

---

## 💼 Business Impact
How does this benefit a Financial Institution or Compliance Team?

1.  **Detection of "Zero-Day" Crime:** Unlike rule-based systems that only catch *known* methods, this unsupervised approach detects *new* anomalies, protecting the bank from emerging threats.
2.  **Operational Efficiency:** By ranking transactions by "Risk Score," compliance officers can focus their limited time on the top 1% most suspicious cases rather than reviewing thousands of false alarms.
3.  **Regulatory Compliance:** Reduces the risk of massive fines (AML non-compliance) by demonstrating a state-of-the-art, proactive monitoring capability.

---

## 🚀 Deployment
To make these insights accessible to stakeholders, the project deployment strategy is as follows:

* **Documentation & Reporting:** The project analysis, interactive notebooks, and final report are hosted via **GitHub Pages**. This serves as the central knowledge hub for the technical implementation and business insights.
* *(Future Integration):* The core model is designed to be wrapped in a REST API (using FastAPI) which can then be connected to a frontend dashboard (built with tools like **Lovable** or **Streamlit**) for real-time transaction scoring by bank analysts.
