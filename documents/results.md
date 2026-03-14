# Results

## Binary Logistic Regression
**Hyperparameter selection**
- alfa (`alfa`) =  0.001
- learning rate (`lr`) =  0.0005
- 

![alt text](/img/grid_searh_logistic.png)


**Performance metrics**
- Training accuracy 0.8921997638396888
- Validation accuracy  0.8946971501151068
- Test accuracy 0.7155784244144784
- Precision 0.9409421782722154
- Recall 0.5338580222862932
- F1 score 0.6812170627423686

![alt text](/img/confusion_matrix.png)

## Binary Neural Networks
**Hyperparameter selection**
- learning Rate: 0.001

**Performance metrics**
- Train Accuracy: 91.97451850087816
- Validation Accuracy: 91.85123442089387
- Test Accuracy: 69.56618168914123

## Multi-class Softmax Regression
**Hyperparameter selection**
- learning rate =  0.0001
- alfa =  0.01

![alt text](/img/grid_search_softmax.png)

**Performance metrics**
- Training accuracy: 0.836860853947747
- Validation accuracy: 0.862467254108121
- Test accuracy: 0.6266855926188787
- Macro F1: 0.07519912622310268

## Multi-class Neural Networks
**Hyperparameter selection**
- Learning Rate: 0.001

**Performance metrics**
- Train Accuracy: 84.57119042657698
- Validation Accuracy: 84.4288322616496
- Test Accuracy: 62.810503903477645

## Stratification
| Model                           | Train Acc | Val Acc | Test Acc | Notes                     |
|---------------------------------|-----------|---------|----------|----------------------------|
| Logistic Regression             | 81.73     | 87.20   | 81.16    | Stratification      |
| Binary Neural Network           |  85.50   |  85.31  |  85.18   | Stratification     |
| Softmax Regression              | 76.92       | 82.05     | 77.06      | Stratification  |
| Multiclass Neural Network       | 82.13       | 82.17     |  82.10     | L2 reg (weight decay) + Stratification       |