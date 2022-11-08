import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve

def cross_validate_model(model, X, y, scorer, k = 5, plot = True, verbose = True, std = False):
    '''
    A function that takes an initialized model and outputs the scores of k-fold cross-validation.
    
    model: An initialized model.
    
    X: A DataFrame of features.
    
    y: A Series of y-values.
    
    scorer: Str. A string for an sklearn scorer.
    
    k: Int. An integer for the number of k-fold cross-validations to run. Default is 5.
    
    plot: Bool. When plot is set to True, a learning curve for the model is plotted. Default is True.
    
    verbose: Bool. When verbose is set to true, the function prints the average train and validation scores. Default is True.
    
    std: Bool. When std is set to True, the standard deviation of validation scores is returned.
    
    returns: Average train and validation scores.
    '''
    cross_val_dict = cross_validate(model, X, y, 
                                    scoring = scorer, 
                                    cv = k,
                                    return_train_score = True)
    if verbose:
        print(f"Average train {scorer} {cross_val_dict['train_score'].mean()}")
        print(f"Average validation {scorer} {cross_val_dict['test_score'].mean()}")
        if std:
            print(f"Standard deviation of validation {scorer} {cross_val_dict['test_score'].std()}")
    if plot == True:
        train_sizes_abs, train_scores, test_scores = learning_curve(estimator = model, 
                                                                    X = X, y = y, 
                                                                    cv = k, 
                                                                    scoring = scorer)
        fig, axs = plt.subplots(figsize = (8, 6))
        axs.plot(train_sizes_abs, np.mean(train_scores, axis = 1),  color = 'red')
        axs.plot(train_sizes_abs, np.mean(test_scores, axis = 1), color = 'blue')
        axs.set_xlabel('Training sample size')
        axs.set_ylabel(scorer)
        axs.legend(['Train', 'Validation'])
        axs.grid(color = 'grey')
        axs.tick_params(rotation = 0)
        axs.set_facecolor('whitesmoke')
        plt.show()
    if std:
        return cross_val_dict['train_score'].mean(),cross_val_dict['test_score'].mean(), cross_val_dict['train_score'].std(), cross_val_dict['test_score'].std()
    else:
        return cross_val_dict['train_score'].mean(), cross_val_dict['test_score'].mean()


class FilterFeatures():
    '''
    A class that determines features to remove from the data.
    '''
    def __init__(self, model):
        '''
        model: A pre-trained model. Options are LogisticRegression, Ridge, RandomForest, LightGBM.
        '''
        self.model = model
    
    def get_feature_importance(self, threshold):
        '''
        A function that gets the feature importance for the model, plots the feature importance values, and finds an "elbow point" for the cut-off of feature importance.
        
        threshold: float. A float value indicating the minimum threshold for the feature importance values in the model.
        
        returns: A feature importance series (indexed by feature), the feature importance value for the "elbow point", a list of features that pass the feature importance threshold.
        '''
        if isinstance(model, (LogisticRegression, Ridge)):
            feature_importance = model.coef_[0]
            feature_names = model.feature_names_in_
        elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            feature_importance = model.feature_importances_
            feature_names = model.feature_names_in_
        elif isinstance(model, (lgb.LGBMRegressor, lgb.LGBMClassifier)):
            feature_importance = model.feature_importances_
            feature_names = model.feature_name_
        feature_importance_series = pd.Series(feature_importance, index = feature_names).sort_values(ascending = False)
        elbow_pt = self.elbow_pt_feat_imp(feature_importance_series.values)
        return feature_importance_series, elbow_pt, feature_importance_series[feature_importance_series<threshold].index
    
    def elbow_pt_feat_imp(self, feat_imp_array):
        '''
        A function that takes an array of feature importances and outputs the value of the elbow point. It also plots a graph of feature importances that marks the elbow point.
        
        feat_imp_array: An array of feature importances.
        
        returns: The feature importance value of the "elbow point".
        '''
        line_pt1 = [0, feat_imp_array[0]]
        line_pt2 = (len(feat_imp_array)-1, feat_imp_array[len(feat_imp_array)-1])
        pt_dist_line = np.zeros(len(feat_imp_array))
        for i, feat in enumerate(feat_imp_array):
            pt_dist_line[i] = self.distance_btwn_pt_line(line_pt1, line_pt2, 
                                                   [i, feat])
        elbow_pt_loc = np.argmax(pt_dist_line)
        elbow_pt = feat_imp_array[elbow_pt_loc]
        plt.title('Best cut off pt for feature importance: {}'.format(elbow_pt))
        plt.plot(feat_imp_array)
        plt.plot(elbow_pt_loc,elbow_pt,'r*')
        plt.axvline(elbow_pt_loc, 0, feat_imp_array[1], linestyle = '--', c = 'green')
        plt.xlabel('Feature')
        plt.ylabel('Feature Importance')
        return elbow_pt
    
    def distance_btwn_pt_line(self, lp1, lp2, p0):
        '''
        A function that calculates the distance between a point and a line.
        
        lp1: A tuple of values that gives the coordinates of the first two-dimensional endpoint of a line.
        
        lp2: A tuple of values that gives the coordinates of the first two-dimensional endpoint of a line.
        
        p0: A tuple of values that gives the coordinates of the point that we want to calculate the distance of to the line.
        
        returns: The distance between the point and the line.
        '''
        numerator = np.abs((lp2[0] - lp1[0]) *(lp1[1] - p0[1]) -  (lp1[0] - p0[0]) *(lp2[1] - lp1[1]))
        denominator = np.sqrt((lp2[0] - lp1[0])**2 + (lp2[1] - lp1[1])**2)
        distance = numerator / denominator
        return distance

    
def plot_roc_auc(model, X, y):
    '''
    A function that plots the graph of the ROC curve.
    
    model: A pre-trained classification model.
    
    X: A DataFrame of features.
    
    y: A Series of y-values
    
    returns: A graph of the ROC curve.
    '''
    fpr, tpr, threshold = roc_curve(y, model.predict_proba(X)[:,1])
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Validation Set')
    plt.show()

def select_best_threshold_fpr_tpr(model, X, y_true, plot = True):
    '''
    A function that selects the best threshold (predict_proba) for the model, based on the false positive and true positive rate.
    
    model: A pre-trained classification model.
    
    X: A DataFrame of features.
    
    y_true: A Series of y-values
    
    plot: Bool. When set to True, the model plots the ROC curve. The default value is True.
    
    returns: A plot of the ROC curve, the roc_auc score, the best threshold based on fpr/tpr, and a tuple of the best fpr/tpr
    '''
    probabilities = model.predict_proba(X)[:,1]
    fpr, tpr, thresh = roc_curve(y_true,probabilities)
    max_geom_mean_idx = np.argmax(np.sqrt((tpr * (1-fpr))))
    min_dist_idx = np.argmin(np.sqrt((0-fpr)**2 + (1-tpr)**2))
    roc_auc = roc_auc_score(y_true, probabilities)
    best_idx = min_dist_idx
    best_thresh = thresh[best_idx]
    best_fpr_tpr = (fpr[best_idx], tpr[best_idx])
    y_pred = model.predict_proba(X)[:,1]>best_thresh
    print('roc_auc_score: {:.3f}, best_fpr_tpr: ({:.3f},{:.3f}), best_thresh: {:.3f}'.format(roc_auc, best_fpr_tpr[0],  best_fpr_tpr[1], best_thresh))
    if plot:
        plt.plot(fpr, tpr)
        plt.scatter(fpr[max_geom_mean_idx], 
                    tpr[max_geom_mean_idx], 
                    marker="o", 
                    label = 'geom_mean: ({:.2f},{:.2f})'.format(fpr[max_geom_mean_idx], 
                    tpr[max_geom_mean_idx]))
        plt.scatter(fpr[min_dist_idx], 
                    tpr[min_dist_idx], 
                    marker = 'o', 
                    label = 'distance: ({:.2f}, {:.2f})'.format(fpr[min_dist_idx], 
                    tpr[min_dist_idx]))
        plt.legend()
        plt.xlabel('TPR')
        plt.ylabel('FPR')
    return roc_auc, best_thresh, best_fpr_tpr

def select_best_threshold_p_r(X, y_true, model, plot = True):
    '''
    A function that selects the best threshold (predict_proba) for the model, based on the precision and recall scores.
    
    model: A pre-trained classification model.
    
    X: A DataFrame of features.
    
    y_true: A Series of y-values
    
    plot: Bool. When set to True, the model plots the precision-recall curve. The default value is True.
    
    returns: A plot of the precision-recall curve, the f1 score, the best threshold based on precision/recall, and a tuple of the best precision/recall.
    '''
    probabilities = model.predict_proba(X)[:,1]
    p, r, thresh = precision_recall_curve(y_true,probabilities)
    max_geom_mean_idx = np.argmax(np.sqrt((p * (r))))
    min_dist_idx = np.argmin(np.sqrt((1-p)**2 + (1-r)**2))
    best_idx = min_dist_idx
    best_thresh = thresh[best_idx]
    y_pred = model.predict_proba(X)[:,1]>best_thresh
    f1 = f1_score(y_true, y_pred)
    prec = p[best_idx]
    rec = r[best_idx]
    print('f1_score: {:.3f}, best_prec: {:.3f}, best_rec: {:.3f}, best_thresh: {:.3f}'.format(f1, prec,  rec, best_thresh))
    if plot:
        plt.plot(p, r)
        plt.scatter(p[max_geom_mean_idx], 
                    r[max_geom_mean_idx], 
                    marker="o", 
                    label = 'geom_mean: ({:.2f},{:.2f})'.format(p[max_geom_mean_idx], 
                    r[max_geom_mean_idx]))
        plt.scatter(p[min_dist_idx], 
                    r[min_dist_idx], 
                    marker = 'o', 
                    label = 'distance: ({:.2f}, {:.2f})'.format(p[min_dist_idx], 
                    r[min_dist_idx]))
        plt.legend()
        plt.xlabel('Precision')
        plt.ylabel('Recall')
    return f1, best_thresh, prec, rec
    
def confusion_matrix_class_report(y_true, y_pred):
    '''
    A function that outputs the classification report and a plot of the normalized confusion matrix.
    
    y_true: A Series or numpy array of the true labels.
    
    y_pred: A Series or numpy array of the predicted labels.
    
    returns: A heatmap for the confusion matrix.
    '''
    print(classification_report(y_true, y_pred))
    cm=confusion_matrix(y_true, y_pred, labels = y_true.unique(), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation='horizontal',cmap='Purples')
    
def plot_feature_importance(model):
    '''
    A function that plots the feature importance for the model and returns a series of feature importance values.
    
    model: A pre-trained model. Options are LogisticRegression, Ridge, RandomForest, LightGBM.
    
    returns: A Series of feature importance values with the features as the indices.
    '''
    if isinstance(model, (LogisticRegression, Ridge)):
        feature_imp = model.coef_[0]
        features = model.feature_names_in_
    elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        feature_imp = model.feature_importances_
        features = model.feature_names_in_
    elif isinstance(model, (lgb.LGBMRegressor, lgb.LGBMClassifier)):
        feature_imp = model.feature_importances_
        features = model.feature_name_
    feat_imp_series = pd.Series(np.abs(feature_imp), index = features).sort_values(ascending = False)
    plt.figure(figsize = (14, 8))
    feat_imp_series.head(20).plot(kind = 'bar', edgecolor = 'k')
    plt.xlabel('Feature')
    plt.ylabel('Relative Importance')
    plt.show()
    return feat_imp_series
