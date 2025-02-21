# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

  # Predict probabilities
Y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Compute ROC curve
fpr, tpr, _ = roc_curve(Y_test, Y_prob)
roc_auc = roc_auc_score(Y_test, Y_prob)
import matplotlib.pyplot as plt
# Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

test_df = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/test.csv")
test_proba = model.predict_proba(test_df.drop("id", axis=1))[:, 1]
submission_df = pd.DataFrame({'smoking': test_proba}, index=test_df["id"])
submission_df.to_csv("sample_submission.csv")

#THE RESULTS OF THE TRAINING
Meilleurs hyperparamètres (XGBoost - Randomized) : {'colsample_bytree': 0.6624074561769746, 'learning_rate': 0.05679835610086079, 'max_depth': 5, 'n_estimators': 558, 'subsample': 0.9464704583099741}
AUC-ROC moyenne (Validation Croisée Stratifiée) : 0.9305

In the competition we had a result of 88.8%
