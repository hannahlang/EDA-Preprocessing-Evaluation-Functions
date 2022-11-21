import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from ipywidgets import interact, interactive, fixed, interact_manual
import networkx as nx

class ClassifierErrorAnalysis():
    '''
    A class that performs error analysis in various forms. Methods are described below.
    '''
    def __init__(self, X_train, y_train, X_test, y_test, pipe_model, processing_steps, y_pred = None, y_pred_proba = None):
        '''
        The init function creates an error_df containing the indices of errors of the model, their relevant data, and corresponding predictions/predicted probabilities.
        
        X_train: Pandas DataFrame of the raw training data.
        y_train: Pandas Series of the training labels.
        X_test: Pandas DataFrame of the raw test data.
        y_test: Pandas Series of the test labels.
        pipe_model: A model pipeline that has already been fit to the data.
        processing_steps: A list of string names for the relevant functions to use for pre-processing in the pipeline of "pipe_model". Not all pre-processing steps need to be included. Particularly not numerical scaling, since it distorts numerical feature values and doesn't help for analysis.
        y_pred: A numpy array of shape (n_samples,) that holds the predictions of X_test.
        y_pred_proba: A numpy array of shape (n_samples, n_classes) that holds the predicted probabilities of X_test for each class.
        '''
        self.X = X_test
        self.X_train = X_train
        self.y_train = y_train
        for step in processing_steps:
            self.X_train = pipe_model[step].transform(self.X_train)
            self.X = pipe_model[step].transform(self.X)
        self.y = y_test.rename('true_class')
        if y_pred is None:
            self.y_pred = pd.Series(pipe_model.predict(X_test), index = self.y.index)
        else:
            self.y_pred = pd.Series(y_pred, index = self.y.index)
        if y_pred_proba is None:
            self.y_pred_proba = pd.DataFrame(pipe_model.predict_proba(X_test), index = self.y.index, columns = pipe_model[-1].classes_)
        else:
            self.y_pred_proba = pd.DataFrame(y_pred_proba, index = self.y.index, columns = pipe_model[-1].classes_)
        self.errors_df = self.create_error_df()

    def create_error_df(self):
        '''
        A function that creates a DataFrame of only the errors made on the test set. Columns include the features, true label, and predicted label.
        
        returns: The errors DataFrame.
        '''
        #Get the total dataframe with all the original values
        data_idxs = self.y[self.y_pred != self.y].index
        errors_df = pd.concat([self.X, self.y],axis = 1).loc[data_idxs]
        errors_df['pred_label'] = self.y_pred.loc[data_idxs]
        errors_df['pred_prob'] = self.y_pred_proba.loc[data_idxs].max(axis = 1)
        errors_df = errors_df.sort_values(by = 'pred_prob', ascending = False)
        return errors_df
    
    def analyze_errors_by_class(self, rel_cols = None, which_class = None):
        '''
        A function that plots a confusion matrix of the predictions of X_test. It also has a barplot of the averages of selected numerical columns by class on the train set compared to the averages of the columns by class for the errors on the test set.
        
        rel_cols: A list of strings of numerical column names to use for comparing averages across classes.
        
        which_class: A string of the label to plot along the training averages in the barplot.
        
        returns: A plot of the confusion matrix/barplot.
        '''
        print('-'*20+'F1-score: {:.3f}, Precision score: {:.3f}, Recall score {:.3f}'
              .format(f1_score(self.y, self.y_pred, average = "weighted"),
                      precision_score(self.y, self.y_pred, average = "weighted"), 
                      recall_score(self.y, self.y_pred, average = 'weighted'))
              + '-'*20 + '\n')
        
        fig, axs = plt.subplots(1, 2, 
                                figsize = (12, 6), 
                                gridspec_kw={'width_ratios': [1, 1.7]},
                                facecolor="aliceblue",
                                linewidth=4, edgecolor="Black")

        cm=confusion_matrix(self.y, self.y_pred, labels = self.y.unique())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=self.y.unique())
        disp.plot(xticks_rotation='vertical', cmap='Purples',ax = axs[0])
        
        axs[0].tick_params(labelrotation=45)
        axs[0].set_title('Confusion Matrix')
        axs[0].set_facecolor('Lavender')

        if rel_cols is None:
            rel_cols = [col for col in self.X.columns if self.X[col].dtypes == int or self.X[col].dtypes == float]
        
        train_averages = []
        col_labels = []
        error_averages = []
        for label in self.y_train.unique():
            train_averages.append(self.X_train[self.y_train == label][rel_cols].mean())
            col_labels.append(label + '_train_avg')

        if which_class is not None:
            errors_avg = self.errors_df[self.errors_df.true_class == which_class][rel_cols].mean()
            avgs_df = pd.concat(train_averages + [errors_avg], axis = 1)
            avgs_df.columns = col_labels + [which_class + ' class_error_avg']
        else:
            for label in self.y_train.unique():
                error_averages.append(self.errors_df[self.errors_df.true_class == label][rel_cols].mean())
                col_labels.append(label + '_error_avg')
            avgs_df = pd.concat(train_averages + error_averages, axis = 1)
            avgs_df.columns = col_labels
            
        avgs_df.plot(kind = 'bar', colormap = 'Blues', edgecolor = 'Black', ax = axs[1])
        axs[1].set_title('Average score vs. error score', fontsize = 12)
        # axs[1].set_xticks(rotation = 45, fontsize = 10)
        axs[1].set_ylabel('Percentage score', fontsize = 11)
        axs[1].tick_params(labelrotation=45)
        axs[1].set_facecolor('Lavender')
        plt.suptitle('Model Performance By Class', fontsize = 14)
        plt.tight_layout()
        plt.show()
        
    def node_sketch(self, y_true, y_pred, axs, cluster_labels = None):
        '''
        A function that takes a true label, predicted label, and cluster label, and outputs a node graph that connects true, predicted, cluster to their labels.
        
        y_true: A string for the true label.
        
        y_pred: A string for the predicted label.
        
        axs: An integer of the axes index for the graph.
        
        cluster_labels: A string for the cluster label. Default is None. When set to None, the plot of the nodes does not include a cluster label.
        
        returns: A node graph described above.
        '''
        #fig, axs = plt.subplots(figsize = (6,4))
        if cluster_labels is None:
            From = ['True\nlabel', 'Predicted\nlabel']
            To = [y_true, y_pred]

        else:
            From = ['True\nlabel', 'Cluster\nlabel','Predicted\nlabel']
            To = [y_true, cluster_labels, y_pred]

        df = pd.DataFrame({'from':From,
                           'to':To})

        if cluster_labels is None:
            pos = {'True\nlabel':(1,1.5), 'Predicted\nlabel':(1, 2.7), y_true:(3,1.5),y_pred:(3,2.7)}
             # Define Node Colors
            NodeColors = {'True\nlabel':'royalblue', 'Predicted\nlabel':'lightcyan',y_true:'deepskyblue',y_pred:'skyblue'}
        else:
            if len(np.unique(np.array(To))) == 2:
                # Define Node Positions
                pos = {'True\nlabel':(1,1), 'Cluster\nlabel':(1,2.2), 'Predicted\nlabel':(1,3.4), y_true:(3,1.5), y_pred:(3,2.7)}

                # Define Node Colors
                NodeColors = {'True\nlabel':'royalblue', 'Cluster\nlabel':'paleturquoise', 'Predicted\nlabel':'lightcyan',y_true:'deepskyblue',y_pred:'skyblue'}
            elif len(np.unique(np.array(To))) == 3:
                # Define Node Positions
                pos = {'True\nlabel':(1,1),'Cluster\nlabel':(1,2.2),'Predicted\nlabel':(1,3.4),y_true:(3,1),cluster_labels: (3, 2.2),y_pred:(3,3.4)}

                # Define Node Colors
                NodeColors = {'True\nlabel':'royalblue', 'Cluster\nlabel':'paleturquoise', 'Predicted\nlabel':'lightcyan',y_true:'deepskyblue',cluster_labels: 'powderblue',y_pred:'skyblue'}

        Labels = {}
        i = 0
        for a in From:
            Labels[a]=a
        for i in To:
            Labels[i]=i

        G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph(ax = axs) )

        # Define the colormap and set nodes to circles, but the last one to a triangle
        Circles = []
        Colors_Circles = []
        for n in G.nodes:
            Circles.append(n)
            Colors_Circles.append(NodeColors[n])

        nodes = nx.draw_networkx_nodes(G, pos, nodelist = Circles,node_size=3e3,node_shape='o',node_color='white',alpha=1, ax = axs)

        nodes = nx.draw_networkx_nodes(G, pos, nodelist = Circles,node_size=3e3,node_shape='o',node_color=Colors_Circles,edgecolors='black',alpha=0.5, ax = axs)


        nx.draw_networkx_labels(G, pos, Labels, font_size=12, ax = axs)

        # Again by making the node_size larer, I can have the arrows end before they actually hit the node
        edges = nx.draw_networkx_edges(G, pos, node_size=9e3,
                                       arrowstyle='->',width=2, ax = axs)

        axs.set_xlim(0,4)
        axs.set_ylim(0,4)
        axs.set_facecolor('Lavender')
        axs.set_title('Predictions')
        
        
    def analyze_errors_idx(self, idx, rel_cols = None, cluster_label = None):
        '''
        A function that analyzes errors by index by outputting a side-by-side node sketch and bar graph. The node sketch connects true/predicted/cluster labels to their label. The bar graph plots the values of the sample for specified columns, compared to the average values of the columns by class in the training data.
        
        idx: An index from the dataframe of errors.
        
        rel_cols: A list of strings of numerical column names to use for comparing averages across classes.
        
        cluster_labels: A string for the cluster label. Default is None. When set to None, the plot of the nodes does not include a cluster label.
        
        returns: The side-by-side plot described above.
        '''
        student_data = self.errors_df.loc[idx]
        print('\nThe model predicted {} with probability {:.3f}.\n'
              .format(student_data.pred_label, student_data.pred_prob))
      
        #Visualize the cluster labels vs. the predicted label vs. the true label.
        fig, axs = plt.subplots(1, 2, figsize = (12, 6), 
                                gridspec_kw={'width_ratios': [1, 1.7]},
                                facecolor="Ivory",
                                linewidth=4, edgecolor="Black")
        if cluster_label is None:
            self.node_sketch(student_data.true_class,student_data.pred_label, axs[0])
        else:
            self.node_sketch(student_data.true_class,student_data.pred_label,cluster_label, axs[0])

        #Sketch a graph of the errors values compared to the the mean value of each class.
        if rel_cols is None:
            rel_cols = [col for col in self.X.columns if self.X[col].dtypes == int]
            rel_cols += [col for col in self.X.columns if self.X[col].dtypes == float]
        mean_vals = []
        index_labels = []
        for label in self.y_train.unique():
            mean_vals.append(self.X_train[self.y_train == label][rel_cols].mean().values)
            index_labels.append(label + '_avg')
        mean_vals.append(student_data[rel_cols].values)
        index_labels.append('Sample Values')
        df = pd.DataFrame(mean_vals, index = index_labels, columns = rel_cols)
        df.transpose().plot(kind = 'bar', colormap = 'Blues',edgecolor = 'Black', ax = axs[1])
        axs[1].set_ylabel('Percentage', fontsize = 12)
        axs[1].set_facecolor('Lavender')
        axs[1].tick_params(labelrotation=45)
        axs[1].set_title('Sample values vs. class averages')
        plt.suptitle(f'Errors for Sample IDX: {idx}', fontsize = 14)
        plt.tight_layout()
        
        
    def errors_by_idx_widget(self, rel_cols = None, cluster_label = None):
        '''
        A function that outputs a widget that allows you to select an index, and perform the "analyze_errors_idx" function on the sample.
        '''
        interact(self.analyze_errors_idx, 
                 idx = self.errors_df.index,
                 rel_cols = fixed(rel_cols),
                 cluster_label = fixed(cluster_label))
