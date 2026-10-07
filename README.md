# Titanic Survival Prediction

Binary classification on the **Titanic – Machine Learning from Disaster** dataset ([Kaggle competition](https://www.kaggle.com/c/titanic)): predicting whether a passenger survived from ticket class, sex, age, family aboard and fare.

## Approach
- **Data:** 891 labelled training passengers and 418 test passengers, combined for consistent preprocessing.
- **Cleaning:** `Ticket`, `PassengerId` and `Cabin` (77% missing) dropped; missing `Fare` filled with the mean, `Embarked` with the mode.
- **Age imputation:** 263 missing ages filled with the median age of the passenger's ticket class (39 / 29 / 24 for 1st / 2nd / 3rd class), since age and class are strongly related.
- **Encoding:** `Sex` one-hot encoded.
- **Models:** Random Forest and Gradient Boosting classifiers trained on the 891 labelled passengers; survival predicted for the 418-passenger test set in Kaggle submission format.

## Files
- `titanic-survival-prediction.ipynb` – full notebook with code and outputs

## Tools
Python · pandas · scikit-learn

---
Part of my [data science portfolio](https://betulcelep.github.io/#data-science), completed during the *Mastering Applied Data Science with Deep Learning* programme.
