# -*- coding: utf-8 -*-
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
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def train_noisy_laplace_model(x_train : list, E : float, 
                              SENS : float) -> list:
    '''
    Adds laplace noise to data.

    Parameters
    ----------
    x_train : list
        A list of data to add laplace noise to.
    E : float
        Privacy budget, ε, of the model.
    SENS : float
        Sensitivity of the model.

    Returns
    -------
    list
        Returns x data with added laplace noise.

    '''
    scale = SENS / E
    
    return X_train + np.random.laplace(scale = scale, size = X_train.shape)


def get_average_accuracy_given_e(E_VALS : list, ROUNDS : int = 10) -> list:
    '''
    Computes the average accuracy of a given epsilon value either 10 or a 
    specified number of times and returns the results of that average accuracy 
    computation for each of the e values passed into the function

    Parameters
    ----------
    E_VALS : list
        A list of epsilon values.

    Returns
    -------
    list
        A list of the average accuracy of each epsilon value.

    '''
    results = []
    
    for i in range(len(E_VALS)):
        
        average = 0
        for j in range(ROUNDS):
            X_train_noisy = train_noisy_laplace_model(X_train, INPUTS[i], 
                                                      np.max(np.abs(X_train)))
            model.fit(X_train_noisy, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            average += accuracy
            
        average /= ROUNDS
        
        results.append(average)
    
    return results


def draw_graph(X_VALUES : list, Y_VALUES : list) -> None:
    '''
    Draws a graph of the model accuracy at different epsilon values.

    Parameters
    ----------
    X_VALUES : list
        List of x values.
    Y_VALUES : list
        List of y values.

    Returns
    -------
    None
    
    '''
    plt.style.use('_mpl-gallery')
    x = range(0, len(X_VALUES) + 1)
    x_labels = list(map(str, X_VALUES))
    x_labels.append("N/a")
    y = Y_VALUES
    
    fig, ax = plt.subplots()
    
    plt.plot(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set(ylim=(0,1), yticks=np.arange(0, 1.1, 0.1))
    
    ax.set_title("Model Accuracy at Different ε Values")
    ax.set_xlabel("ε Values")
    ax.set_ylabel("Accuracy")
    
    plt.show()


if __name__ == '__main__':
    bcancer = datasets.load_breast_cancer() # one of toy dataset in sklearn
    
    print(bcancer.DESCR)
    
    X = bcancer.data
    y = bcancer.target
    df = pd.DataFrame(X, columns = bcancer.feature_names) # columns=[age, sex,...]
    df['Diagnosis']= y # 1-Malignant, 0 - Benign
    
    print(df.shape)
    print(df.sample(n=5))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                        random_state=42)
    
    model = LogisticRegression()
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    INPUTS = [0.01, 0.1, 0.5, 1, 5, 10]
    results = get_average_accuracy_given_e(INPUTS)
    results.append(accuracy)
    
    print(f'{results = }')
    
    draw_graph(INPUTS, results)