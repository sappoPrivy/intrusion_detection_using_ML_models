# Intrusion Detection with implemented ML Models

This project implements several machine‑learning models **from scratch using NumPy** to classify network traffic in the NSL‑KDD dataset. This includes PCA, logistic regression, softmax regression, binary neural network, multiclass neural network, which are implemented manually without ML libraries.


## 🚀 Project Overview

Models implemented:

- PCA (dimensionality reduction)
- Binary Logistic Regression (from scratch)
- Softmax Regression (multiclass)
- Binary Neural Network (manual forward/backprop)
- Multiclass Neural Network (softmax output)

All experiments are done in Jupyter notebooks inside `src/`.

## 📦 Prerequisites

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
## How to Run
Run any model:
- src/dimensionality_reduction.ipynb
- src/binary_logistic_regression.ipynb
- src/binary_neural_networks.ipynb
- src/multiclass_softmax_regression.ipynb
- src/multiclass_neural_networks.ipynb
- Data is automatically loaded from data/KDDTrain+.txt and data/KDDTest+.txt.

## Project Structure
```bash
INTRUSION_DETECTION/
│
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── documents/
│   ├── method.md
│   ├── results.md
│   └── theory.md
│
├── img/
│   ├── confusion_matrix.png
│   ├── grid_search_logistic.png
│   └── image.png
│
├── src/
│   ├── binary_logistic_regression.ipynb
│   ├── binary_neural_networks.ipynb
│   ├── dimensionality_reduction.ipynb
│   ├── multiclass_neural_networks.ipynb
│   └── multiclass_softmax_regression.ipynb
│
└── README.md

```

## Model Performance Summary
| Model                           | Train Acc | Val Acc | Test Acc | Notes                     |
|---------------------------------|-----------|---------|----------|----------------------------|
| Logistic Regression             | 89.22     | 89.47   | 71.56    | -      |
| Binary Neural Network           | 91.56    | 91.39   | 71.15    | -            |
| Softmax Regression (Multiclass) | TBD       | TBD     | TBD      | -  |
| Multiclass Neural Network       | TBD       | TBD     | TBD      | -       |

## Documentation **[🔗](./documents/)**

Inside the [documentation](./documents/) folder you will find:

- **theory.md** — all mathematical foundations  
- **method.md** — full methodology, preprocessing, training pipeline  
- **results.md** — detailed metrics, plots, confusion matrices  