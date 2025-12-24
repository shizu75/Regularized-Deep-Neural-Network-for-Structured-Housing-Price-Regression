# Regularized Deep Neural Network for Structured Housing Price Regression

## Abstract

This repository presents a **research-grade implementation of a deep neural network for housing price prediction**, designed to model **nonlinear relationships in structured tabular data**. The pipeline integrates **careful feature encoding**, **statistical preprocessing**, and a **regularized multi-layer neural architecture** to achieve stable generalization in a regression setting.

The work emphasizes **methodological clarity, reproducibility, and modeling rigor**, positioning it as a strong component of a doctoral research portfolio in applied machine learning, data-driven systems, or computational modeling.

---

## Research Motivation

Accurate valuation of residential properties is a classical yet challenging regression problem due to:
- Mixed categorical and numerical features  
- Strong feature correlations  
- Scale imbalance across variables  
- High risk of overfitting in expressive models  

This project addresses these challenges by combining:
- Explicit semantic encoding of categorical variables  
- Train–validation–test stratification  
- Feature standardization  
- Deep neural networks with **dropout and L2 regularization**

The resulting system demonstrates how **modern deep learning techniques can be applied rigorously to tabular economic data**, without relying on black-box automation.

---

## Dataset and Feature Engineering

### Dataset
- Source: Housing market dataset (`Housing.csv`)
- Target variable: `price`
- Feature types:
  - Binary categorical (e.g., main road access, air conditioning)
  - Ordinal categorical (furnishing status)
  - Continuous numerical attributes

### Encoding Strategy
Semantic encodings are applied to preserve interpretability:
- Binary features mapped to {0, 1}
- Ordinal furnishing status mapped to {0, 1, 2}
- No one-hot inflation, ensuring compact feature space

This encoding choice reflects **domain-aware preprocessing**, minimizing unnecessary dimensional expansion.

---

## Data Partitioning

The dataset is split hierarchically:
- Training set: 60%
- Validation set: 24%
- Test set: 16%

This structure ensures:
- Proper hyperparameter tuning on validation data
- Unbiased generalization assessment on held-out test data

---

## Model Architecture

The regression model is a **fully connected deep neural network** with progressive dimensionality reduction:

- Dense (300 units, Leaky ReLU, L2 regularization)
- Dropout (50%)
- Dense (150 units, Leaky ReLU, L2 regularization)
- Dropout (50%)
- Dense (45 units, Leaky ReLU, L2 regularization)
- Dense (12 units, ReLU)
- Output layer (1 unit, linear activation)

### Design Rationale
- **Leaky ReLU** prevents dead neuron issues
- **L2 regularization** controls weight magnitude
- **Dropout** mitigates co-adaptation and overfitting
- Gradual compression enforces hierarchical feature abstraction

---

## Preprocessing and Normalization

- Input features standardized using `StandardScaler`
- Target variable normalized to stabilize optimization
- Scaling parameters learned exclusively on training data

This ensures **numerical stability and fair gradient propagation** during training.

---

## Training Protocol

- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- Epochs: 100
- Validation monitoring enabled throughout training

Training dynamics are recorded and visualized to assess convergence behavior and detect overfitting.

---

## Evaluation Metrics

The model is evaluated using:
- Mean Squared Error on standardized targets
- Coefficient of Determination (R² score)

R² provides a **scale-independent measure of explained variance**, offering clear interpretability of predictive performance.

---

## Results and Observations

- Smooth convergence across training and validation curves
- No evidence of severe overfitting due to regularization
- Stable generalization on unseen test data
- Strong explanatory power as reflected by R² score

The results validate the effectiveness of **regularized deep learning for structured regression problems**.

---

## Scientific Contribution

This project demonstrates:
- Thoughtful feature encoding without excessive dimensionality
- Proper experimental protocol for regression modeling
- Application of deep learning principles beyond vision or language tasks
- Reproducible, interpretable, and scalable modeling practices

The methodology is directly extensible to:
- Real estate analytics
- Economic forecasting
- Tabular biomedical or engineering datasets

---

## Reproducibility

- Fully deterministic data preprocessing pipeline
- Explicit train/validation/test separation
- Standard open-source Python libraries
- Clear architectural and optimization choices

---

## Keywords

Deep Neural Networks, Regression Modeling, Housing Price Prediction, Regularization, Dropout, Tabular Data, Feature Engineering, Applied Machine Learning
