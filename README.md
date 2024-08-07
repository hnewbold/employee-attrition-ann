# Predicting Employee Attrition with Deep Learning: An Artificial Neural Network (ANN) Approach using TensorFlow

**Author**: Husani Newbold

**Date**: 2024-07-30

## Table of Contents
1. [Introduction & Project Description](#introduction--project-description)
2. [Dataset](#dataset)
3. [Model Structure](#model-structure)
4. [Training the Model](#training-the-model)
5. [Results](#results)
6. [Improvements and Recommendations](#improvements-and-recommendations)
7. [Contributors](#contributors)

## Introduction & Project Description
This project utilizes a deep learning model, specifically an Artificial Neural Network (ANN), to predict whether an employee will leave a company based on common work-related attributes such as tenure, salary level, and work hours.

To ensure high-quality input data, data cleaning, exploratory data analysis (EDA), and preprocessing were performed. This included checking for outliers, scaling numerical features, and encoding categorical variables. These preprocessing steps were implemented in a scikit-learn pipeline to ensure a streamlined and efficient data transformation process.

Keras Tuner was used to efficiently find the best hyperparameters for the model. After tuning, the optimal model was replicated with the identified hyperparameters. The best model trained over 24 epochs, resulting in a test dataset loss of 0.10 and an accuracy of 0.97.

This process demonstrated the capability of deep learning models to make accurate predictions based on a limited set of variables, proving the effectiveness of careful tuning and model selection.

## Dataset
The dataset for this project was sourced from Kaggle and originates from Sailsfort Motors, an automobile company. Initially, it consisted of approximately 15,000 records. After removing duplicates, the dataset was reduced to around 12,000 records. The dataset contains the following 10 variables:

- **satisfaction_level**: Employee satisfaction level
- **last_evaluation**: Last evaluation score
- **number_project**: Number of projects completed while at work
- **average_monthly_hours**: Average monthly working hours
- **tenure**: Number of years at the company
- **work_accident**: Whether the employee had a work accident (1 = yes, 0 = no)
- **left**: Whether the employee left the company (1 = yes, 0 = no)
- **promotion_last_5years**: Whether the employee was promoted in the last five years (1 = yes, 0 = no)
- **department**: Department where the employee works
- **salary**: Salary level (low, medium, high)

## Model Structure
The following model structure was used by the Keras Tuner to identify the best hyperparameters for training. This neural network consists of:

- **Two hidden layers**: The number of units in each layer was chosen from a range of 32 to 512, with step increments of 32.
- **Activation functions**: The activation functions 'relu' and 'tanh' were tried to determine the best performing model.
- **Output layer**: A single neuron with a sigmoid activation function to output a probability of the employee leaving.
- **Learning rates**: Different learning rates (1e-2, 1e-3, 1e-4, 1e-5) were tested to find the optimal setting for model training.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

def model_builder(hp):
    model = Sequential()

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_layer_1 = hp.Int('layer_1', min_value=32, max_value=512, step=32)
    hp_layer_2 = hp.Int('layer_2', min_value=32, max_value=512, step=32)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    model.add(Dense(units=hp_layer_1, activation=hp_activation, input_shape=(X_train.shape[1],)))
    model.add(Dense(units=hp_layer_2, activation=hp_activation))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model
```

## Training the model
The best model hyperparameters, determined by Keras Tuner, were loaded to train the model. Early stopping was implemented to prevent overfitting.

```python
# Train model using hyperparameters from best model
model = tuner.hypermodel.build(best_hyperparameters)
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[stop_early])
```

### Training Metrics
Key metrics from the model training process are summarized below:
```
Epoch 24/100
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9768 - loss: 0.0761 - val_accuracy: 0.9755 - val_loss: 0.0917
```
These results indicate that the model performs well on both the training and validation datasets, achieving high accuracy and low loss for both.

## Results
### Classification Report
The classification report demonstrates the model's high performance across both categories ("Stayed" and "Left"). The model achieves an overall accuracy of 97%, indicating its effectiveness in predicting employee attrition

```
              precision    recall  f1-score   support

      Stayed       0.98      0.99      0.98      1996
        Left       0.94      0.90      0.92       403

    accuracy                           0.97      2399
   macro avg       0.96      0.94      0.95      2399
weighted avg       0.97      0.97      0.97      2399
```
## Improvements and Recommendations
While the model performs very well, achieving a high accuracy of 97%, there are always potential improvements and considerations for future work, consider the following:

- **Increase Training Data**: Incorporating additional labeled data can help improve the model's generalization capability, making it more robust to new, unseen data.
- **Further Hyperparameter Tuning**: Although Keras Tuner was used to find optimal hyperparameters, exploring a broader range of hyperparameters or using different tuning algorithms might yield even better results.
- **Regularization Techniques**: Implementing additional regularization techniques such as dropout, L1/L2 regularization, or batch normalization could help prevent overfitting and improve model generalization.
  
## Contributors
Husani Newbold (Author)



