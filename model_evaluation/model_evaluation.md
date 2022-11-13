# Model Evaluation Functions

## cross_validate_model <i>(model, X, y, scorer, k = 5, plot = True, verbose = True, std = False):</i>

A function that takes an initialized model and outputs the scores of k-fold cross-validation.
    
**model:** ***An initialized model.***
    
**X:** ***Pandas DataFrame (n_samples, n_features).***

A DataFrame of features.
    
**y:** ***Pandas Series (n_samples,)***

A series of y-values.
    
**scorer:** ***Str***

A string for an sklearn scorer.
    
**k:** ***Int, default = 5*** 

An integer for the number of k-fold cross-validations to run.
    
**plot:** ***bool, default = True*** 

When plot is set to True, a learning curve for the model is plotted. 
    
**verbose:** ***Bool, default = True***

When verbose is set to true, the function prints the average train and validation scores. 
    
**std:** ***bool, default = False*** 

When std is set to True, the standard deviation of validation scores is returned.
    
**Returns** 

Average train and validation scores.

    
## plot_roc_auc <i>(model, X, y):</i>

A function that plots the graph of the ROC curve.
    
**model:** ***A pre-trained classification model.***
    
**X:** ***Pandas DataFrame (n_samples, n_features).***

A DataFrame of features.
    
**y:** ***Pandas Series (n_samples,)***

A series of y-values.
    
**Returns:** 

A graph of the ROC curve.

## select_best_threshold_fpr_tpr <i>(model, X, y_true, plot = True):</i>

A function that selects the best threshold (predict_proba) for the model, based on the false positive and true positive rate.
    
**model:** ***A pre-trained classification model.***
    
**X:** ***Pandas DataFrame (n_samples, n_features).***

A DataFrame of features.
    
**y:** ***Pandas Series (n_samples,)***

A series of y-values.
    
**plot:** ***bool, default = True***

When set to True, the model plots the ROC curve.
    
**Returns:** 

A plot of the ROC curve, the roc_auc score, the best threshold based on fpr/tpr, and a tuple of the best fpr/tpr


## select_best_threshold_p_r <i>(X, y_true, model, plot = True):</i>

A function that selects the best threshold (predict_proba) for the model, based on the precision and recall scores.
    
**model:** ***A pre-trained classification model.***
    
**X:** ***Pandas DataFrame (n_samples, n_features).***

A DataFrame of features.
    
**y:** ***Pandas Series (n_samples,)***

A series of y-values.
    
**plot:** ***bool, default = True***

When set to True, the model plots the precision-recall curve.
    
**Returns:** 

A plot of the precision-recall curve, the f1 score, the best threshold based on precision/recall, and a tuple of the best precision/recall.
 
    
## confusion_matrix_class_report <i>(y_true, y_pred, figsize = (10, 10), rotation = 0, normalize = 'true'):</i>

A function that outputs the classification report and a plot of the normalized confusion matrix.
    
**y_true:** ***Pandas Series, numpy array (n_samples,)*** 

An array or Series of the true labels
    
**y_pred:** ***Pandas Series, numpy array (n_samples,)*** 

An array or Series of the predicted labels.

**figsize:** ***tuple (n,m), default = (10, 10)***

Tuple of integer dimensions for the size of the figure.

**rotation:** ***int, default = 0***

Integer value of the degree rotation for the tick labels in the graph.

**normalize:** ***{‘true’, ‘pred’, ‘all’}, default=None***
Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. If None, confusion matrix will not be normalized.
    
**Returns:** 

A heatmap for the confusion matrix.
   
    
## plot_feature_importance <i>(model, top_features = 30, figsize = (14, 8)):</i>

A function that plots the feature importance for the model and returns a series of feature importance values.
    
**model:** ***A pre-trained model:***  {LogisticRegression, Ridge, RandomForestRegressor, RandomForestClassifier, LGBMRegressor, LGBMClassifier}

**top_features:** <i>int $\in$ (1, num_features), default = 30 </i>

A positive integer that gives the number of the top features to plot in the graph. 

**figsize:** ***tuple (n,m), default = (14, 8)***

A tuple of integer values for the figsize of the confusion matrix.

**Returns:** 

A Series of feature importance values with the features as the indices.
