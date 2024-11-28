# deep-learning-challenge
UCI Data Analyst Module 21 - Neural Networks and Deep Learning

# Alphabet Soup Charity Neural Network Model

This repository contains the implementation of a neural network model to predict the success of organizations funded by Alphabet Soup. The goal is to determine whether an organization will be successful based on various features in the dataset. The project involves data preprocessing, model training, optimization, and evaluation.

## Overview

The purpose of this analysis is to create and optimize a deep learning model to predict if an Alphabet Soup-funded organization will be successful. Using the provided dataset, a neural network model is built, trained, and optimized to classify organizations as successful or not.

## Steps and Results

### Step 1: Data Preprocessing

#### Target Variable(s):
- The target variable for the model is **`IS_SUCCESSFUL`**, which indicates whether the organization was successful (1) or not (0).

#### Feature Variable(s):
- The features for the model include all other columns in the dataset, such as:
  - Categorical variables: `APPLICATION_TYPE`, `AFFILIATION`, etc.
  - Numerical features: `ASK_AMT`, `INCOME_AMT`, etc.

#### Variables Removed:
- **`EIN`**: A unique identifier for each organization, which does not contribute to prediction.
- **`NAME`**: The name of the organization, irrelevant for the predictive model.

#### Categorical Data Encoding:
- Categorical variables were encoded using **one-hot encoding** (`pd.get_dummies()`), converting categories into binary columns to be used by the neural network.

#### Data Splitting:
- The data was split into features (`X`) and target (`y`), followed by a train-test split using **train_test_split**.
- Data was scaled using **StandardScaler** to standardize the feature set for neural network processing.

---

### Step 2: Compiling, Training, and Evaluating the Model

#### Model Architecture:
- **Neurons & Layers**: The neural network was built with:
  - Input layer corresponding to the number of features.
  - Two hidden layers: 
    - 80 neurons in the first hidden layer
    - 30 neurons in the second hidden layer
  - Output layer with a single neuron for binary classification.
  
- **Activation Functions**:
  - Hidden layers: **ReLU (Rectified Linear Unit)** activation.
  - Output layer: **Sigmoid** activation to output values between 0 and 1 (binary classification).

#### Performance:
- The initial goal was to achieve **>75% accuracy**. While optimization efforts improved performance, the target was not consistently reached in early training stages.

#### Optimization Efforts:
1. Adjusted binning for categorical features to reduce noise.
2. Increased neurons in the hidden layers.
3. Experimented with adding more epochs.
4. Tried different combinations of activation functions and layers.

---

### Step 3: Model Optimization

#### Steps Taken to Improve Performance:
- Tuned hyperparameters such as:
  - **Number of neurons**: Increasing the neurons in hidden layers to improve model capacity.
  - **Epochs**: Increasing epochs to ensure the model converges properly.
- Attempted different feature transformations and experimented with other model architecture configurations.

#### Performance Results:
After optimization, the model’s performance improved but did not consistently exceed 75% accuracy.

---

### Step 4: Final Report

#### Overview of the Model:
- The deep learning model successfully identifies patterns in the dataset for organizational success prediction but requires further optimization to consistently exceed the target accuracy of 75%.

#### Key Results:
- **Target Variable(s)**: `IS_SUCCESSFUL`, indicating whether an organization was successful (1) or not (0).
- **Feature Variable(s)**: All columns in the dataset except for `EIN` and `NAME`, which were excluded due to their irrelevance to the prediction task.
- **Model Architecture**:
  - **Input Layer**: 128 neurons with **LeakyReLU** activation and **Dropout (0.3)** for regularization.
  - **First Hidden Layer**: 512 neurons with **PReLU** activation and **Dropout (0.3)** for regularization.
  - **Second Hidden Layer**: 256 neurons with **ELU** activation and **Dropout (0.3)**.
  - **Third Hidden Layer**: 128 neurons with **ReLU** activation and **Dropout (0.3)**.
  - **Fourth Hidden Layer**: 64 neurons with **Tanh** activation and **Dropout (0.3)**.
  - **Fifth Hidden Layer**: 32 neurons with **Sigmoid** activation and **Dropout (0.3)**.
  - **Output Layer**: A single neuron with **Sigmoid** activation for binary classification.
  
- **Activation Functions**:
  - **LeakyReLU**, **PReLU**, **ELU**, **ReLU**, **Tanh**, and **Sigmoid** were chosen to help the model learn complex patterns and ensure smoother outputs at various stages of the network.
  
- **Regularization**: 
  - **L2 regularization** was applied to all layers to prevent overfitting.
  - **Dropout** was added to each layer (set to 0.3) to further reduce overfitting.

- **Performance**: The model achieved improved performance over earlier iterations. However, while accuracy reached acceptable levels during training, the model did not consistently exceed the 75% accuracy target in test runs.

- **Optimization Efforts**:
  - The architecture was progressively expanded with more layers and different activation functions to capture complex patterns in the data.
  - Regularization techniques like **L2 regularization** and **Dropout** were incorporated to prevent overfitting and improve generalization.

The architecture aims to strike a balance between complexity and generalization by incorporating various advanced activation functions and regularization methods.

#### Recommendation for Future Work:
To achieve higher performance, consider using alternative models like **Random Forest** or **Gradient Boosting**, which might better handle mixed feature types and achieve higher predictive accuracy with less computational effort and tuning.

---

### Technologies Used

- **Python**: Programming language used for the project.
- **TensorFlow**: For building and training the neural network.
- **Keras**: High-level neural network API for model creation.
- **Scikit-learn**: For data preprocessing (StandardScaler, train-test split).
- **Pandas**: For data manipulation and preprocessing.

## Acknowledgments

- **ChatGPT:** Special thanks to ChatGPT, which assisted in the exploratory optimization of the deep learning model. Through iterative suggestions and adjustments, it helped refine the model's architecture and performance strategies.
  
---

### Files

- `AlphabetSoupCharity.ipynb`: Initial implementation of data preprocessing and model training.
- `AlphabetSoupCharity_Optimization.ipynb`: Optimized neural network model with hyperparameter adjustments.
- `AlphabetSoupCharity.h5`: Saved model after initial training.
- `AlphabetSoupCharity_Optimization.h5`: Optimized model after adjustments.

---

### Conclusion

This project demonstrates how to preprocess a dataset, design a neural network model, and apply optimization techniques. Though the model's performance did not meet the desired threshold, further tuning and alternative models can help improve the classification accuracy.
