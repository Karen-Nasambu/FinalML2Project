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
