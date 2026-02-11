# Intrusion Detection Using From Scratch ML Models

## Dataset
NSL-KDD data comprises of normal traffic and different types of attack traffic.

<pre>features = [
    # categorical
    "protocol_type", "service", "flag",

    # numeric
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "dst_host_srv_count", "dst_host_serror_rate",
    "logged_in", "root_shell", "su_attempted"
]</pre>

- Training set: (100779, 94)
- Validation set: (25194, 94)
- Test set: (22544, 94)
- Label normal traffic as 0 and attack traffic 1

## Dimensionality Reduction
### Principal Component Analysis
Reduces dimensions by projecting the input data on eigenvectors with largest eigenvalues, representing maximum variance.

1. **Compute covariance matrix**
    
    $$
    S = \frac1n (X-\overline{X})^T *(X-\overline{X})
    $$
    
2. **Compute eigenvectors of the matrix**: the eigenvectors needs to be sorted according to largest eignevalues, since we want to select eigenvectors with the largest eigenvalues (represents directions with maximum variance)
    
3. **Compute projection of data onto eigenvectors**

$$
X_{proj} = (X-\overline{X}) * (eigenvectors)
$$

## Logistic regression
**Predicted probability**: Sigmoid function.



$$
\hat{p} = q_\theta(x_i) = \sigma(\theta^\top x_i)
$$


**Predicted label**: Decision rule.



$$
\hat{y} = f(\hat{p}) =
\begin{cases}
1 & \text{if } \hat{p} \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
$$


**Loss function**: binary cross‑entropy loss with L2 regularization

$$
L(w, b)
= -\frac{1}{n}\sum_{i=1}^{n}
\left[
y_i \log \sigma(w^\top x_i + b)
+
(1 - y_i)\log(1 - \sigma(w^\top x_i + b))
\right]
+
\frac{\alpha}{2}\|w\|_2^2
$$

### Stochastic Gradient Descent
**SGD Algorithm**: Minimize the loss function by using gradients to update the weights.

**1. For each of the \(T\) iterations, perform weight updates**

**a. Random sampling**: Select \(B\) random samples from the dataset.

**b. Compute the gradient for \(w\) and \(b\) of the average loss \(L\)**

**Gradient w.r.t. weights**

$$
\frac{\partial}{\partial w} L(w, b)=\sum_{i=1}^{n} x_i \left(\sigma(w^\top x_i + b) - y_i\right) + \alpha w
$$

$$ \frac{\partial}{\partial w} L(w, b) = X^\top \left(\sigma(Xw + b) - Y\right) + \alpha w $$


**Gradient w.r.t. bias**

$$
\frac{\partial}{\partial b} L(w, b)=\sum_{i=1}^{n} \left(\sigma(w^\top x_i + b) - y_i\right)
$$


$$
\frac{\partial}{\partial b} L(w, b)=\sum_{i=1}^{n} \left(\sigma(Xw + b) - Y\right)
$$

 **c. Update rule for parameters**

$$
\theta_{t+1} := \theta_t + \nabla_\theta
$$

$$
\theta_{t+1} = \theta_t - \lambda \frac{\partial}{\partial \theta} L(\theta)
$$





## Generalization
### Grid Search

| Learning Rates (`lr`) | Regularization Strengths (`alfa`) |
|-----------------------|-----------------------------------|
| 1e-4                 | 0.0                               |
| 5e-4                 | 1e-4                              |
| 1e-3                 | 1e-3                              |
| 5e-3                 | 1e-2                              |

- Fixed batch size (`B`) at 256
- Fixed iterations (`T`) at 10000

**Hyperparameter selection**
- alfa (`alfa`) =  0.001
- learning rate (`lr`) =  0.0005

![alt text](/grid_searh_logistic.png)

### Performance metrics
**Precision**: 

$$\text{precision} = \frac{\text{TP}}{\text{TP + FP}}$$

**Recall**: 

$$\text{Recall} = \frac{\text{TP}}{\text{TP + FN}}$$

**F1-Score**:

$$F_{1} = 2 \cdot  \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

**Results**
![alt text](/confusion_matrix.png)
- Training accuracy 0.8921997638396888
- Validation accuracy  0.8946971501151068
- Test accuracy 0.7155784244144784
- Precision 0.9409421782722154
- Recall 0.5338580222862932
- F1 score 0.6812170627423686

## Neural Networks
**Preactivation**

$$
z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}
$$

**Activation**

$$
a^{(\ell)} = \sigma\!\left(z^{(\ell)}\right)
= \sigma\!\left(W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}\right)
$$

**Gradient of the last layer $L$**

$$
g^{(L)} = \frac{\partial J}{\partial z^{(L)}}
=  \frac{\partial J}{\partial a^{(L)}} \cdot  \frac{\partial a^{(L)}} {\partial z^{(L)}}
= \frac{\partial J}{\partial \hat{y}}  \cdot  \sigma'\!\left(z^{(L)}\right)
$$

**Gradient of an arbitrary layer $\ell$**

$$
g^{(\ell)} =  \frac{\partial J}{\partial z^{(\ell)}}
= \left(W^{(\ell+1)}\right)^\top g^{(\ell+1)} \odot  \sigma'\!\left(z^{(\ell)}\right)
$$

**Gradient of the loss w.r.t. weights**

$$
\nabla_{W^{(\ell)}} J =g^{(\ell)} \, {a^{(\ell-1)}}^\top
$$

**Gradient of the loss w.r.t. biases**

$$
\nabla_{b^{(\ell)}} J = g^{(\ell)}
$$

**Backpropagation Method**

**1. Initialize layers**: Create each layer with
- Weight matrix:  $W^{(\ell)} \in \mathbb{R}^{\,n_\ell \times n_{\ell-1}}$
- Bias vector:  $b^{(\ell)} \in \mathbb{R}^{\,n_\ell \times 1}$

**2. Forward pass**: Compute and store all preactivations  $z^{(\ell)}$ and activations $a^{(\ell)}$

**3. Backward pass (loop from last layer $L$ down to layer 1)**

**a. Compute gradient for the last layer** $g^{(L)} $

**b. Compute gradient for any hidden layer**  $g^{(\ell)}$

**c. Compute parameter gradients**  
- Weight gradient: $\nabla_{W^{(\ell)}} J$
- Bias gradient: $\nabla_{b^{(\ell)}} J$

**4. Update parameters** for each layer:

$$
W^{(\ell)} \leftarrow W^{(\ell)} - \eta \, \nabla_{W^{(\ell)}} J
$$

$$
b^{(\ell)} \leftarrow b^{(\ell)} - \eta \, \nabla_{b^{(\ell)}} J
$$
