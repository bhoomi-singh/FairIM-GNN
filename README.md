# 🧠 Fairness Influence Maximisation using GNNs

A Graph Neural Network (GNN)-based Autoencoder designed to learn **structure-preserving and fairness-aware node embeddings** using reconstruction, adversarial debiasing, and contrastive learning.

---

## 🚀 Overview

This project implements a **Topology-Aware Graph Autoencoder** that:

- Learns node embeddings using **Graph Convolutional Networks (GCN)**
- Preserves graph structure via reconstruction loss
- Reduces bias from sensitive attributes using **fairness constraints**
- Improves robustness with **contrastive learning on augmented graphs**

---

## 🧩 Architecture

### 🔹 Encoder (GCN-based)
- Learns node embeddings via message passing
- Captures node features and graph topology

### 🔹 Decoder
- Reconstructs adjacency matrix from embeddings
- Preserves structural information

### 🔹 Autoencoder Pipeline
- End-to-end encoder-decoder training
- Multi-objective optimization

---

## ⚙️ Key Components

### 📌 Loss Functions

1. **Reconstruction Loss**
   - Binary cross-entropy on adjacency matrix

2. **First-Order Proximity Loss**
   - Preserves local neighborhood similarity

3. **Covariance Loss (Fairness Constraint)**
   - Minimizes correlation between embeddings and sensitive attributes

4. **Adversarial Loss**
   - Discriminator removes sensitive attribute information

5. **Contrastive Loss (NT-Xent)**
   - Aligns embeddings from augmented graph views

---

### 🔁 Graph Augmentation

- Random edge dropping (`drop_edges`)
- Creates multiple views for contrastive learning

---

## 🧪 Training Strategy

### Phase 1: Pretraining
- Train encoder-decoder using reconstruction + proximity loss

### Phase 2: Fairness Optimization
- Introduce covariance + adversarial loss

### Phase 3: Contrastive Learning (Optional)
- Train with augmented graph views

---

## 📦 Tech Stack

- TensorFlow / Keras  
- Spektral  
- NumPy  
- Pandas  
- NetworkX  

---

## 📁 Project Structure

```
├── Model.ipynb              # Main implementation notebook
├── README.md               # Project documentation
```

---

## 📊 Evaluation Metrics

- Reconstruction Accuracy  
- Proximity Preservation  
- Fairness (correlation reduction)  
- Contrastive Consistency  

---

## 🛠️ Installation

```bash
pip install spektral tensorflow numpy pandas networkx
```

---

## ▶️ Usage

```bash
jupyter notebook Model.ipynb
```

Steps:
1. Load dataset (adjacency, features, sensitive attributes)
2. Initialize encoder and decoder
3. Pretrain autoencoder
4. Apply fairness constraints
5. Train with contrastive learning (optional)
6. Evaluate performance

---

## 🔍 Future Improvements

- Support multiple sensitive attributes
- Scale using graph sampling techniques
- Benchmark on real-world datasets

---

## 📌 Contributions

This project explores:

- Graph Representation Learning  
- Fairness in Machine Learning  
- Self-Supervised Learning  

---

## 📜 License

Open for research and educational purposes.
