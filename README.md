Absolutely, Tenzin — here is a **fully GitHub‑compatible**, **cleanly formatted**, **KaTeX‑safe** version of your entire document.  
I’ve rewritten every formula so it renders correctly on GitHub (no stray `*`, no blank lines inside math blocks, no ambiguous parentheses, no MathJax‑only syntax).

Everything is structurally identical to your original — just fixed for GitHub.

---

# **Intrusion Detection Using From‑Scratch ML Models**

## **Dataset**

NSL‑KDD contains normal traffic and multiple attack types.

<pre>
features = [
    # categorical
    "protocol_type", "service", "flag",

    # numeric
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "dst_host_srv_count", "dst_host_serror_rate",
    "logged_in", "root_shell", "su_attempted"
]
</pre>

- Training set: (100779, 94)  
- Validation set: (25194, 94)  
- Test set: (22544, 94)  
- Labels: normal = 0, attack = 1

---

## **Dimensionality Reduction**

### **Principal Component Analysis**

1. **Covariance matrix**

$$
S = \frac{1}{n} (X - \overline{X})^\top \cdot (X - \overline{X})
$$

2. **Compute eigenvectors**, sorted by descending eigenvalues.

3. **Project data**

$$
X_{\text{proj}} = (X - \overline{X}) \cdot V
$$

where \(V\) contains the selected eigenvectors.

---

## **Logistic Regression**

### **Predicted probability**

$$
\hat{p} = \sigma(w^\top x_i + b)
$$

### **Predicted label**

$$
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

### **Binary cross‑entropy loss with L2 regularization**

$$
L(w,b)
= -\frac{1}{n}\sum_{i=1}^{n}
\left[
y_i \log \sigma(w^\top x_i + b)
+
(1 - y_i)\log(1 - \sigma(w^\top x_i + b))
\right]
+ \frac{\alpha}{2}\|w\|_2^2
$$

---

## **Stochastic Gradient Descent**

### **1. Random sampling**

Select a mini‑batch of size \(B\).

### **2. Compute gradients**

**Gradient w.r.t. weights**

$$
\frac{\partial L}{\partial w}
= X^\top \left( \sigma(Xw + b) - Y \right) + \alpha w
$$

**Gradient w.r.t. bias**

$$
\frac{\partial L}{\partial b}
= \sum_{i=1}^{n} \left( \sigma(w^\top x_i + b) - y_i \right)
$$

### **3. Update rule**

$$
\theta_{t+1} = \theta_t - \lambda \, \frac{\partial L}{\partial \theta}
$$

---

## **Generalization**

### **Grid Search**

| Learning Rates | Regularization Strengths |
|----------------|--------------------------|
| 1e‑4           | 0.0                      |
| 5e‑4           | 1e‑4                     |
| 1e‑3           | 1e‑3                     |
| 5e‑3           | 1e‑2                     |

- Batch size \(B = 256\)  
- Iterations \(T = 10000\)

**Selected hyperparameters**

- \(\alpha = 0.001\)  
- \(lr = 0.0005\)

`file:///grid_searh_logistic.png`

---

## **Performance Metrics**

**Precision**

$$
\text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

**Recall**

$$
\text{recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

**F1‑Score**

$$
F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

`file:///confusion_matrix.png`

- Training accuracy: 0.8922  
- Validation accuracy: 0.8947  
- Test accuracy: 0.7156  
- Precision: 0.9409  
- Recall: 0.5339  
- F1 score: 0.6812

---

## **Neural Networks**

### **Preactivation**

$$
z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}
$$

### **Activation**

$$
a^{(\ell)} = \sigma(z^{(\ell)})
$$

### **Gradient of the last layer**

$$
g^{(L)} =
\frac{\partial J}{\partial a^{(L)}} \odot \sigma'(z^{(L)})
$$

### **Gradient of hidden layers**

$$
g^{(\ell)} =
\left(W^{(\ell+1)}\right)^\top g^{(\ell+1)}
\odot \sigma'(z^{(\ell)})
$$

### **Weight gradient**

$$
\nabla_{W^{(\ell)}} J = g^{(\ell)} \, a^{(\ell-1)\top}
$$

### **Bias gradient**

$$
\nabla_{b^{(\ell)}} J = g^{(\ell)}
$$

---

## **Backpropagation Method**

**1. Initialize layers**

- \(W^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}\)  
- \(b^{(\ell)} \in \mathbb{R}^{n_\ell \times 1}\)

**2. Forward pass**

Compute and store all \(z^{(\ell)}\) and \(a^{(\ell)}\).

**3. Backward pass**

- Compute \(g^{(L)}\)  
- For each layer \(\ell\), compute \(g^{(\ell)}\)  
- Compute \(\nabla_{W^{(\ell)}} J\) and \(\nabla_{b^{(\ell)}} J\)

**4. Parameter update**

$$
W^{(\ell)} \leftarrow W^{(\ell)} - \eta \, \nabla_{W^{(\ell)}} J
$$

$$
b^{(\ell)} \leftarrow b^{(\ell)} - \eta \, \nabla_{b^{(\ell)}} J
$$

---

If you want, I can also:

- generate a **PDF‑ready** version  
- produce a **cleaner layout** with collapsible sections  
- add **diagrams** for forward/backward pass  
- or rewrite the whole thing in a more “paper‑like” academic style

Just tell me how polished you want this to look.
