import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class PredictConfidenceIntervals(BaseEstimator, TransformerMixin):
    '''
    A class that creates confidence intervals for each prediction of the model. This is done through bootstrapping and finding quantiles of the bootstrapped predictions.
    '''
    def __init__(self, estimator, num_bootstraps = 100):
        '''
        estimator: An initialized sklearn classifier that has an attribute of predict_proba.
        
        num_bootstraps: The number of times to train the model and collect predictions. Default is set to 100. Must be an integer value.
        '''
        self.estimator = estimator
        self.num_bootstraps = num_bootstraps
        
    def fit(self, X, y, boot_size = None):
        '''
        A function that trains a base model as well as num_bootstraps number of models on samples of the training data.
        
        X: DataFrame of training features.
    
        y: Series of training labels.
        
        boot_size: The number of samples to train on for each model.
        '''
        self.main_model = clone(self.estimator)
        self.main_model.fit(X, y)
        # Fit number of models according to number of bootstrapped models.
        self.bootstrapped_models = []
        if boot_size == None:
            boot_size = len(X)
        for i in range(self.num_bootstraps):
            boot_strapped_model = clone(self.estimator)
            sample_idxs = np.random.choice(np.arange(len(X)), boot_size)
            X_sampled = X.iloc[sample_idxs]
            y_sampled = y.iloc[sample_idxs]
            self.bootstrapped_models.append(boot_strapped_model.fit(X_sampled, y_sampled))
        return self
    
    def predict(self, X, c_i = 95, plot = False):
        '''
        A function that takes the training data and confidence intervals and returns a plot of the confidence intervals as well as a DataFrame that contains the pred_proba, lower and upper confidence intervals, lower confidence interval labels, predicted labels, and true labels.
        
        X: DataFrame of training features.
    
        y: Series of training labels.
        
        c_i: The confidence interval. Can be an integer or float value. The number represents the percentagel.
        
        plot: If set to true, the function plots the predicted probabilities of each sample with their confidence intervals. Default is set to False.
        
        returns: DataFrame of confidence interval information as stated above.
        '''

        #Gather the model predictions.
        predicted_label = self.main_model.predict(X)
        predicted_probabilities = self.main_model.predict_proba(X)

        #Finding the index and values of the maximum predicted probability.
        predicted_label_idxs = predicted_probabilities.argmax(axis = 1)
        predicted_label_prob = predicted_probabilities.max(axis = 1)

        #Saving the results of each of the n_bootstraps models.
        boot_strapped_predictions = np.zeros((self.num_bootstraps, len(X), len(self.main_model.classes_)))
        boot_max_class_predictions = np.zeros((self.num_bootstraps, len(X)))
        
        for i in range(self.num_bootstraps):
            probabilities = self.bootstrapped_models[i].predict_proba(X)
            boot_strapped_predictions[i] = probabilities
            #Saving the predicted probabilities of the predicte class of each sample for each bootstrapped model.
            boot_max_class_predictions[i]= probabilities[np.arange(0, len(X)), predicted_label_idxs]
        
        quant = c_i/100

        #Gathering the confidence intervals for each class for each sample.
        lower_bound_all_class = np.quantile(boot_strapped_predictions, ((1-quant)/2), axis = 0)
        upper_bound_all_class = np.quantile(boot_strapped_predictions, ((1-quant)/2)+quant, axis = 0)
        confidence_intervals_all_class = np.array((lower_bound_all_class,upper_bound_all_class)).T

        #Gathering the confidence intervals for only the predicted class for each sample.
        lower_bound_max_class = np.quantile(boot_max_class_predictions, ((1-quant)/2), axis = 0)
        upper_bound_max_class = np.quantile(boot_max_class_predictions, ((1-quant)/2)+quant, axis = 0)
        confidence_intervals_max_class = np.array((lower_bound_max_class,upper_bound_max_class)).T
       
        #Finding the index of the models that brought the lower confidence probability.
        lower_conf_max_label_idxs = self.find_nearest_idx(boot_max_class_predictions,
                                                         lower_bound_max_class)

        #Finding the label with the maximum probability when the predicted class had its lower confidence probability.
        lower_conf_label = boot_strapped_predictions[lower_conf_max_label_idxs, 
                                                     np.arange(boot_strapped_predictions.shape[1]), :].argmax(axis = 1)
        lower_conf_named_label = self.main_model.classes_[lower_conf_label.reshape(len(X), 1)]

        #Creating a DataFrame the provides for each sample the predicted probability, probability confidence intervals, 
        #and labels of predicted and lower confidence probabilities.
        confidence_int_df = pd.DataFrame({'pred_prob':predicted_label_prob,
                                         'lower_ci_val':confidence_intervals_max_class[:, 0],
                                          'upper_ci_val': confidence_intervals_max_class[:, 1],
                                         'pred_label': predicted_label,
                                         'lower_conf_label': lower_conf_named_label.reshape(-1)}, 
                                         index = X.index)
        if plot:
            plt.figure(figsize = (8, 6), facecolor = 'Beige', edgecolor = 'Black')
            print('{:.2%} of the predictions were predicted with probability above 0.5.\n\n'
                  .format((predicted_label_prob>0.5).mean()))
            print('{:.2%} of the lower boundaries of the confidence intervals had probabilities above 0.5.\n\n'
                  .format((confidence_int_df.lower_ci_val>0.5).mean()))
            lower_error = predicted_label_prob - lower_bound_max_class
            upper_error = upper_bound_max_class - predicted_label_prob
            plt.errorbar(np.arange(len(predicted_label_prob)), 
                         predicted_label_prob[predicted_label_prob.argsort()], 
                         yerr = np.array((lower_error[predicted_label_prob.argsort()],
                                          upper_error[predicted_label_prob.argsort()])),
                         fmt = 'bo',
                         ecolor = 'tab:green', capsize = 2,
                         barsabove = True)
            plt.axhline(y=0.5, color='Black', linestyle='--')
            plt.xlabel('Sample')
            plt.ylabel('Probability')
            plt.title('Confidence Intervals of Predictions')
            plt.grid()
            plt.show()
        return confidence_int_df    
    
    def find_nearest_idx(self, boot_max_class, max_class_lb):
        '''
        Function that finds the index of the bootstrapped sample that is nearest to the lower confidence interval.
        
        boot_max_class: A numpy array of all the bootstrapped predictions of the predicted class.
        
        max_class_lb: A float value that represents the lower confidence interval probability value of the predicted class.
        
        returns: Index of the nearest bootstrapped sample to the lower confidence interval.
        '''
        lb_nearest_idx = np.abs(boot_max_class.T - max_class_lb.reshape(len(max_class_lb), 1)).argmin(axis = 1)
        return lb_nearest_idx
