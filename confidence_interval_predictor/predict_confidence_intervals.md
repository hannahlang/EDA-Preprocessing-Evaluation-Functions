# PredictConfidenceIntervals()

A class that provides confidence intervals for the "predicted probabilities" of a multi-class classifier. It returns a DataFrame that contains for each sample, the predicted probability, lower and uppper confidence probabilities (and associated labels at these intervals).

## Parameters

<b> estimator: </b> ***An initialized classifier***

An initialized classifier that has an attribute of predict_proba.

<b> num_bootstraps: </b> **int $\in$(1, inf)**, ***default = 100***

The number of models to train for bootstrapping.

## Attributes

<b>bootstrapped_models: </b> ***list***

A list of trained models used for bootstrapping.

<b>main_model: </b> ***Trained model***

The trained model used for predictions.

## Methods
        
### fit <i>(X, y, boot_size = None):</i>
A function that trains a base model as well as num_bootstraps number of models on samples of the training data.

#### Parameters: 
<b> X: </b> ***Pandas DataFrame of shape (n_samples, n_features)***

Training DataFrame, where n_samples is the number of samples and n_features is the number of features/columns to transform.

<b> y: </b> ***Pandas Series of shape (n_samples,)***

<b> boot_size: </b> **int $\in$(1, inf)**, ***default = None***
        
The number of samples to train on for each model. WHen set to none, the bootstrapping side becomes the size of the training set.

#### Returns: self

### predict <i>(X, c_i = 95, plot = False):</i>
A function that takes the training data and confidence intervals and returns a plot of the confidence intervals as well as a DataFrame that contains the pred_proba, lower and upper confidence intervals, lower confidence interval labels, predicted labels, and true labels.

#### Parameters: 
<b> X: </b> ***Pandas DataFrame of shape (n_samples, n_features)***

Training DataFrame, where n_samples is the number of samples and n_features is the number of features/columns to transform.

<b> c_i: </b> **int, float $\in$(1, 100)**, ***default = 95***
        
The confidence interval. The number represents the percentage.

<b> plot: </b> ***bool, default = False***
        
If set to true, the function plots the predicted probabilities of each sample with their confidence intervals. 

#### Returns: <i>Pandas DataFrame</i>

A DataFrame that contains the pred_proba, lower and upper confidence intervals, lower confidence interval labels, predicted labels, and true labels.
   
### find_nearest_idx <i>(boot_max_class, max_class_lb):</i>
Function that finds the index of the bootstrapped sample that is nearest to the lower confidence interval.

#### Parameters: 
<b> boot_max_class: </b> ***narray of shape (num_bootstraps, n_samples)***

A numpy array of all the bootstrapped predictions of the predicted class.

<b> max_class_lb: </b> **float $\in$(0, 1)**
        
The lower confidence interval probability value of the predicted class.

#### Returns: <i>int $\in$(0, num_bootstraps)</i>
Index of the nearest bootstrapped sample to the lower confidence interval.
        
