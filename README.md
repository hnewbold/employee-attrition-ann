# Predicting Employee Attrition using an Artificial Neural Network (ANN)

**Author**: Husani Newbold

**Date**: 2024-07-30

## Table of Contents
1. [Introduction](#introduction)
2. [Project Description](#project-description)
3. [Model Structure](#model-structure)
4. [Training the Model](#training-the-model)
5. [Results](#results)
7. [Contributors](#contributors)

## Introduction
This project aims to develop an Artificial Neural Network (ANN) to predict which employees are likely to leave the company based on various factors such as tenure, salary, and working hours.

## Project Description
The goal of this project is to illustrate the power of deep learning models in extracting patterns and information using a few variables to make accurate and useful predictions. Furthermore, by using Keras Tuner, we efficiently tune this powerful model and find the best hyperparameters relatively quickly.

## Model Structure
The following model structure was used by the Keras Tuner to identify the best hyperparameters for trainin:

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
```python
# Train model using hyperparameters from best model
model = tuner.hypermodel.build(best_hyperparameters)
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[stop_early])
```

### Training Metrics
```
Epoch 9/100
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9728 - loss: 0.0889 - val_accuracy: 0.9713 - val_loss: 0.1019
```

## Results
### Classification Report
```
              precision    recall  f1-score   support

      Stayed       0.97      0.99      0.98      1996
        Left       0.97      0.86      0.91       403

    accuracy                           0.97      2399
   macro avg       0.97      0.93      0.95      2399
weighted avg       0.97      0.97      0.97      2399
```

### Test Loss and Accuracy
```
Test Accuracy: 0.9717
Test Loss: 0.1077
```

## Contributors
Husani Newbold (Author)



