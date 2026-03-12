# Methods

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

## Data Patitioning
- Training set: (100779, 94)
- Validation set: (25194, 94)
- Test set: (22544, 94)

## Data Preprocessing
- Normalization

## Binary Logistic Regression
- Label normal traffic as 0 and attack traffic 1
### Grid Search

| Learning Rates (`lr`) | Regularization Strengths (`alfa`) |
|-----------------------|-----------------------------------|
| 1e-4                 | 0.0                               |
| 5e-4                 | 1e-4                              |
| 1e-3                 | 1e-3                              |
| 5e-3                 | 1e-2                              |

- Fixed batch size (`B`) at 256
- Fixed iterations (`T`) at 10000

## Binary Neural Networks

## Multi-class Softmax Regression

## Multi-class Neural Networks