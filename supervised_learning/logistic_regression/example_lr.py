import numpy as np
import pandas as pd
import scipy
original_data = pd.read_csv('data/adult_income.csv')

def process_data(data):
    """
    Process the adult income dataset
    """
    data = data.copy()
    # Replace missing values
    data = data.replace({' ?': np.nan})
    
    # Code gender
    data['female'] = data['sex'].replace({' Male': 0, ' Female': 1})
    # Code target
    data['target'] = data['target'].replace({' >50K': 1, ' <=50K': 0})
    # Create single column for capital wealth
    data['capital'] = data['capital_gain'] - data['capital_loss']
    to_drop = ['country', 'education', 'sex', 
           'capital_gain', 'capital_loss', 
           'working_class',
          'race', 'occupation']
    # Remove excess columns
    data = data.drop(columns=to_drop)
    data = pd.get_dummies(data)
    return data

data = process_data(original_data)

from sklearn.metrics import f1_score, roc_auc_score

def evaluate(model, X_test, y_test):
    """
    Test a model on a few classification metrics.
    """
    # Predictions and probabilities
    predictions = model.predict(X_test)
#     probabilities = model.predict_proba(X_test)[:, 1]
#     roc_auc = roc_auc_score(y_test, probabilities)
    f1_value = f1_score(y_test, predictions)
    accuracy = np.mean(predictions == y_test)
    
    # Get a baseline
    base_accuracy = np.mean(y_test == 0)
#     print('ROC AUC: {:.4f}'.format(roc_auc))
    print('F1 Score: {:.4f}'.format(f1_value))
    print('Accuracy: {:.2f}%'.format(100 * accuracy))
    print('Baseline Accuracy: {:.2f}%'.format(100 * base_accuracy))


from sklearn.model_selection import train_test_split
# Features and target
X = data.copy()
y = X.pop('target')

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
import sys
sys.path.insert(0, '../')

from logistic_regression import LogisticRegression
lr_ = LogisticRegression(learning_rate=.1, gradient_descent=True)
lr_.fit(X_train, y_train)
evaluate(lr_, X_test, y_test)


# from linear_model import LogisticRegression
# lr_1 = LogisticRegression()
# lr_1.fit(X_train, y_train)
# evaluate(lr_1, X_test, y_test)