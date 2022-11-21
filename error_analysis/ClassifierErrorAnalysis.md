# ClassifierErrorAnalysis():
A class that performs error analysis in various forms. Methods are described below.

## Parameters:

**X_train:** ***Pandas DataFrame (n_samples, n_features)***

A DataFrame of the raw training data.

**y_train:** ***Pandas Series(n_samples,)***

A Series of the training labels.

**X_test:** ***Pandas DataFrame (n_samples, n_features)***

A DataFrame of the raw test data.

**y_test:** ***Pandas Series(n_samples,)***

A Series of the test labels.

**pipe_model:** ***A fitted sklearn model pipeline***

A fitted sklearn model pipeline for processing and predicting from the raw data.

**processing_steps:** ***list***

A list of string names for the relevant functions to use for pre-processing in the pipeline of "pipe_model". Not all pre-processing steps need to be included. Particularly not numerical scaling, since it distorts numerical feature values and doesn't help for analysis.

**y_pred:** ***narray of shape (n_samples,), default = None***

An array that holds the predictions of X_test. When set to None, the class uses the model to get the predictions.

**y_pred_proba:** ***narray of shape (n_samples, n_classes), default = None***

An array that holds the predicted probabilities of each class for X_test. When set to None, the class uses the model to get the predictions.

## Attributes:

**X:** ***Pandas DataFrame***

A DataFrame of the processed features in the test set.

**y_pred:** ***Pandas Series (n_samples,)***

A Series of the predicted labels of the test set.

**y_pred_proba:** ***Pandas DataFrame (n_samples, n_classes)***

A DataFrame of the predicted probabilities of the test set for each of the classes.

**errors_df:** ***Pandas DataFrame***

The errors DataFrame containing the pre-processed features, true label, and predicted labels of the errors.

## Methods:

### create_error_df()
A function that creates a DataFrame of only the errors made on the test set. Columns include the pre-processed features, true label, and predicted label.

#### Parameters: None

#### Returns: <i>Pandas DataFrame</i>

The errors DataFrame containing the pre-processed features, true label, and predicted labels of the errors.

### analyze_errors_by_class(rel_cols = None, which_class = None)
A function that plots a confusion matrix of the predictions of X_test. It also has a barplot of the averages of selected numerical columns by class on the train set compared to the averages of the columns by class for the errors on the test set.
        
#### Parameters

**rel_cols:** ***list, default = None*** 

A list of strings of numerical column names to use for comparing averages across classes.
        
**which_class:** ***str, default = None***

A string of the label to plot along the training averages by class in the barplot.
        
#### Returns <i>Matplotlib Plot</i>

A plot of the confusion matrix/barplot.

### node_sketch(y_true, y_pred, axs, cluster_labels = None)
A function that takes a true label, predicted label, and cluster label, and outputs a node graph that connects true, predicted, cluster to their labels.

#### Parameters

**y_true:** ***str***

A string for the true label.
        
**y_pred:** ***str*** 

A string for the predicted label.
        
**axs:** ***int*** 

An integer of the axes index for the graph.
        
**cluster_labels:** ***str, default = None*** 

A string for the cluster label. When set to None, the plot of the nodes does not include a cluster label.
        
#### Returns

A node graph described above.

### analyze_errors_idx(idx, rel_cols = None, cluster_label = None)
A function that analyzes errors by index by outputting a side-by-side node sketch and bar graph. The node sketch connects true/predicted/cluster labels to their label. The bar graph plots the values of the sample for specified columns, compared to the average values of the columns by class in the training data.

#### Parameters

**idx:** ***int, float, str, datetime*** 

An index from the dataframe of errors.
        
**rel_cols:** ***list, default = None*** 

A list of strings of numerical column names to use for comparing averages across classes. WHen set to None all numerical columns are used.
        
**cluster_labels:** ***str, default = None***

A string for the cluster label. When set to None, the plot of the nodes does not include a cluster label.

#### Returns <i>Matplotlib Plot</i>

The side-by-side plot described above.

### errors_by_idx_widget(self, rel_cols = None, cluster_label = None)
A function that outputs a widget that allows you to select an index, and perform the "analyze_errors_idx" function on the sample.

#### Parameters

**rel_cols:** ***list, default = None** 

A list of strings of numerical column names to use for comparing averages across classes.
        
**cluster_labels:** ***str, default = None***

A string for the cluster label. When set to None, the plot of the nodes does not include a cluster label.

#### Returns

The side-by-side plot described above.
