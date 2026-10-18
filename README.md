# Titanic Survival Prediction – Machine Learning and Deep Learning

Binary classification on the **Titanic – Machine Learning from Disaster** dataset ([Kaggle competition](https://www.kaggle.com/c/titanic)): predicting whether a passenger survived from ticket class, sex, age, family aboard and fare.

## 1. Machine learning (scikit-learn)
- **Data:** 891 labelled training passengers and 418 test passengers, combined for consistent preprocessing.
- **Cleaning:** `Ticket`, `PassengerId` and `Cabin` (77% missing) dropped; missing `Fare` filled with the mean, `Embarked` with the mode.
- **Age imputation:** 263 missing ages filled with the median age of the passenger's ticket class (39 / 29 / 24 for 1st / 2nd / 3rd class), since age and class are strongly related.
- **Encoding:** `Sex` one-hot encoded.
- **Models:** Random Forest and Gradient Boosting classifiers trained on the 891 labelled passengers; survival predicted for the 418-passenger test set in Kaggle submission format.

## 2. Deep learning (Keras)
A feed-forward neural network built with TensorFlow/Keras on the same cleaned data.

- **Features:** ticket class, sex, age, siblings/spouses, parents/children and fare, standardised with `StandardScaler`.
- **Architecture:** five hidden ReLU layers (80–120–64–30–8 units) and a sigmoid output; binary cross-entropy loss, Adam optimiser; 20% of the training data held out for validation.
- **Overfitting diagnosed:** trained for 100 epochs, validation accuracy peaked at about 0.87–0.88 in the first 15 epochs, then fell to 0.80–0.83 while training accuracy kept rising.
- **Fixed with early stopping:** monitoring validation loss (patience 10, best weights restored), training stopped after 27 epochs and kept the weights from epoch 17, with **validation accuracy of about 0.87** and no sign of overfitting.

## Files
- `titanic-survival-prediction.ipynb` – machine learning notebook (Random Forest, Gradient Boosting)
- `titanic-survival-deep-learning-keras.ipynb` – deep learning notebook (Keras neural network with early stopping)

## Tools
Python · pandas · scikit-learn · TensorFlow/Keras · matplotlib

---
Part of my [data science portfolio](https://betulcelep.github.io/#data-science), completed during the *Mastering Applied Data Science with Deep Learning* programme.
