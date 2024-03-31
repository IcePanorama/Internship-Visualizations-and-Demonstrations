# gets rid of sklearn ConvergenceWarning and UserWarning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def calc_model_accuracy(TEST_VALS : np.ndarray, PRED_VALS : np.ndarray) -> float:
    '''
    Calculates the accuracy of the original model based on its actual output vs
    what the output should be.

    Parameters
    ----------
    TEST_VALS : np.ndarray
        The ground truth values.
    PRED_VALS : np.ndarray
        The predictions.

    Returns
    -------
    float
        Accuracy of the predictions.

    '''
    total = 0
    for i in range(len(PRED_VALS)):
        if TEST_VALS[i] == PRED_VALS[i]:
            total += 1
    
    return total / len(PRED_VALS)


def calc_inference_accuracy(MODEL : LogisticRegression, INPUT : list[int], 
                        EXP_OUTPUT : list[int]) -> float:
    '''
    Tests the accuracy of a MIA.

    Parameters
    ----------
    MODEL
        A logistic regression model.
    INPUT : list[int]
        A list of input values.
    EXP_OUTPUT : list[int]
        A list of expected output values.

    Returns
    -------
    float
        Returns the accuracy of the provided model.

    '''
    total = 0
    for i in range(len(INPUT)):
        preds = MODEL.predict(INPUT[i].reshape(1,-1))
        if preds[0] == EXP_OUTPUT[i]:
            total += 1
    
    return total / len(INPUT)


if __name__ == "__main__":
    bcancer = datasets.load_breast_cancer()

    x = bcancer.data
    y = bcancer.target
    df = pd.DataFrame(x, columns = bcancer.feature_names) # columns=[age, sex,...]
    df['Diagnosis'] = y  # 1-Malignant, 0 - Benign
    
    train_df, test_df = train_test_split(df, test_size=0.5) #, random_state=42)
    
    # train original model
    x_train = train_df.drop("Diagnosis", axis = 1)
    y_train = train_df["Diagnosis"]
    
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # train adversarial/shadow model
    x_test = test_df.drop("Diagnosis", axis = 1)
    y_test = test_df["Diagnosis"]
    
    y_pred = model.predict(x_test)
    
    accuracy = calc_model_accuracy(y_test.values, y_pred)
    print("Accuracy of the original model:", accuracy)
    
    adversarial_model = LogisticRegression()
    adversarial_model.fit(x_test, y_pred)
    
    adv_accuracy = calc_inference_accuracy(adversarial_model, x_train.values, 
                                               y_train.values)
    print("Accuracy of Membership Inference Attack", adv_accuracy)