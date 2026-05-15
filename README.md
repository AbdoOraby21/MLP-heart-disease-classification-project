# Heart Disease Classification using MLP (Neural Networks)

## Problem Description

The goal of this project is to predict whether a patient has heart disease or not based on clinical features.  
This is a **binary classification** problem (0 = No Disease, 1 = Disease) solved using a **Multilayer Perceptron (MLP)** built with PyTorch.

---

## Dataset

- **Name:** Heart Disease UCI Dataset  
- **Source:** [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
- **Samples:** 303 (×100 augmented = 30,300)  
- **Features:** 13 clinical features  
- **Target:** `target` (0 or 1)

### Features Description

| Feature    | Description                          |
|------------|--------------------------------------|
| age        | Age of the patient                   |
| sex        | Sex (1 = male, 0 = female)           |
| cp         | Chest pain type (0–3)                |
| trestbps   | Resting blood pressure               |
| chol       | Serum cholesterol (mg/dl)            |
| fbs        | Fasting blood sugar > 120 mg/dl      |
| restecg    | Resting ECG results                  |
| thalach    | Maximum heart rate achieved          |
| exang      | Exercise induced angina              |
| oldpeak    | ST depression induced by exercise    |
| slope      | Slope of the peak exercise ST segment|
| ca         | Number of major vessels (0–3)        |
| thal       | Thalassemia type                     |

---

## Model Architecture

```
Input Layer  →  13 neurons
Hidden Layer 1  →  hidden_size neurons  →  Activation  →  Dropout(0.3)
Hidden Layer 2  →  hidden_size/2 neurons  →  Activation  →  Dropout(0.3)
Output Layer  →  1 neuron  →  Sigmoid
```

- **Loss Function:** Binary Cross Entropy (BCELoss)  
- **Optimizer:** Adam  
- **Epochs:** 50  

---

## Experiments

Three experiments were conducted by varying the **activation function**, **number of neurons**, and **learning rate**:

| Experiment | Activation | Hidden Size | Learning Rate |
|------------|------------|-------------|---------------|
| Exp 1      | ReLU       | 32          | 0.0005        |
| Exp 2      | Tanh       | 32          | 0.0005        |
| Exp 3      | ReLU       | 64          | 0.001         |

---

## Results

| Experiment                      | Test Accuracy | Best Val Accuracy | Notes                        |
|---------------------------------|---------------|-------------------|------------------------------|
| Exp 1: ReLU \| 32 \| lr=0.0005 | 0.7388        | ~0.73             | Steady learning, stable      |
| Exp 2: Tanh \| 32 \| lr=0.0005 | 0.7185        | ~0.71             | Slower convergence than ReLU |
| Exp 3: ReLU \| 64 \| lr=0.001  | **0.8564**    | ~0.85             | Best performance ✅           |

> **Best model:** Experiment 3 — ReLU activation, 64 neurons, lr=0.001 achieved the highest test accuracy of **85.64%**

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-mlp-classification.git
cd heart-mlp-classification
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download `heart.csv` from:  
[https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

Place the file in the **same folder** as `heart_mlp_experiments.py`

### 4. Run the Code

```bash
python heart_mlp_experiments.py
```

---

## Output

After running, you will see:
- Training logs every 10 epochs for each experiment
- Test Loss, Accuracy, and MSE for each experiment
- Comparison table printed in the terminal
- 3 plots:
  - Loss Curves (Train vs Validation) for all experiments
  - Accuracy Curves (Train vs Validation) for all experiments
  - Bar Chart comparing Test Accuracy across experiments

---

## Requirements

```
torch
pandas
numpy
matplotlib
scikit-learn
```

---

## Project Structure

```
heart-mlp-classification/
│
| results/
   ├── accuracy_comparison.png
   ├── loss_curves.png
   └── accuracy_curves.png
├── heart_mlp_experiments.py   # Main source code
├── heart.csv                  # Dataset (download separately)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Regularization Techniques Used

| Technique | Details                                      |
|-----------|----------------------------------------------|
| Dropout   | 0.3 rate after each hidden layer             |
| Scaling   | StandardScaler (mean=0, std=1) on all splits |

---

## Course Information

- **Course:** Neural Networks  
- **University:** Badr University in Assiut (BUA)  
<<<<<<< Updated upstream
- **Faculty:** Faculty of Artificial Intelligence
=======
- **Faculty:** Faculty of Artificial Intelligence
>>>>>>> Stashed changes
