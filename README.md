# EDA-Preprocessing-Evaluation-Functions/Classes


## Folders

## confidence_interval_predictor
### predict_confidence_intervals.md
A file explaining in detail the functions in predict_confidence_intervals.py.
### predict_confidence_intervals.py
#### 1. PredictConfidenceIntervals()
A class that providence lower confidence predictions as well as regular predictions for each sample in the dataset for a multi-class classifier.

## eda
### eda.md
A file explaining in detail the functions in eda_functions.py.
### eda_functions.py
#### 1. plot_col_distributions
A function that takes the DataFrame of features and plots the distribution of the features in subplots. 
If the feature is an integer or object column, the function plots a bar graph. If the column is a float column, it plots a histogram.

#### 2. plot_col_vs_num_target
A function that plots the relationship between each feature in the list and the numerical target. 
If the column is numerical, a scatterplot is plotted. If the column is an object, a side-by-side boxplot is plotted.

#### 3. plot_col_vs_cat_target
A function that plots the relationship between each feature in the list and the categorical target. 
If the column is numerical, a side-by-side boxplot is plotted. If the column is an object, a barplot of value counts is plotted.

#### 4. check_quasi_constant_features
A function that takes the DataFrame of features and outputs a DataFrame that shows the category with the largest percentage of values for each feature.

#### 5. corr_feat_target
A function that plots a bar graph showing the correlation between each numerical feature and the target.

#### 6. correlation_matrix
A function that takes the DataFrame of features and outputs an lower triangle correlation heatmap.

#### 7. association_features_binary_target
A function that returns the names of the features that have a direct association with the target, and the corresponding p-values of the chi-squared tests and t-tests.

## error_analysis
### ClassifierErrorAnalysis.md
A file explaining in detail the class in ClassifierErrorAnalysis.py.

### ClassifierErrorAnalysis.py
#### 1. ClassifierErrorAnalysis()
A class that performs error analysis for classification models in various forms. Methods are described below.

## feature_processing
### NumericalScaler.md
A file explaining in detail the NumericalScaler() class.
### OneHotEncoder.md
A file explaining in detail the OneHotEncoder() class.
### feature_processing.py
#### 1. NumericalScaler()
A class that scales numerical variables for regression models. Method is min/max scaling or standardization. It keeps the data in a DataFrane.
#### 2. OneHotEncoder()
A class that one-hot-encodes categorical variables and returns the features as a DataFrame concatenated to the original DataFrame.

## model_evaluation
### filter_features_class.md
A file explaining in detail the FilterFeatures() class.
### model_evaluation.md 
A file explaining in detail the functions in model_evaluation.py
### model_evaluation.py 
#### 1. cross_validate_model
A function that takes an initialized model and outputs the train and validation scores of k-fold cross-validation.
#### 2. plot_roc_auC
A function that takes a pre-trained model and the data and plots the graph of the ROC curve.
#### 3. select_best_threshold_fpr_tpr
A function that takes a pre-trained model and the data and selects the best threshold (predict_proba) for the model based on the false positive and true positive rate.
#### 4. select_best_threshold_p_r
A function that takes a pre-trained model and the data and selects the best threshold (predict_proba) for the model based on the precision and recall scores.
#### 5. confusion_matrix_class_report
A function that outputs the classification report and a plot of the normalized confusion matrix.
#### 6. plot_feature_importance
A function that plots the feature importance for the model and returns a series of feature importance values.
#### 7. FilterFeatures()
A class that determines features to remove from the data.



