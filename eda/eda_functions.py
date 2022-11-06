import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

def plot_col_distributions(df, col_list = None, num_graph_cols = 4, figsize = (20, 20)):
    '''
    This function takes the DataFrame of features and plots the distribution of the features in subplots. 
    If the column is an integer or object column it plots a bar graph. If the column is a float column, it plots a histogram.
    
    df: Pandas DataFrame of features.
    
    col_list: List of columns to plot. Default is set to None, and in this case, all the columns are considered.
    
    num_graph_cols: The number of columns in the subplot. Must be a positive integer value.
    
    figsize: Tuple of integer dimensions for the size of the figure.
    
    '''
    if col_list is None:
        col_list = list(df.columns)
    if len(col_list) % num_graph_cols == 0:
        num_rows = len(col_list) // num_graph_cols
    else:
        num_rows = (len(col_list) // num_graph_cols) + 1
    fig, axs = plt.subplots(num_rows, num_graph_cols, figsize = figsize)
    for idx, col in enumerate(col_list):
        if df[col].dtypes == float:
            graph_type = 'histplot'
        elif df[col].dtypes == object:
            graph_type = 'countplot'
        elif df[col].dtypes == int:
            if len(df[col].unique())>30:
                graph_type = 'histplot'
            else:
                graph_type = 'countplot'
        axs_idx = (idx// num_graph_cols, idx % num_graph_cols)
        if graph_type == 'countplot':
            sns.countplot(x = col, data =df, ax = axs[axs_idx], edgecolor = 'k')
        elif graph_type == 'histplot':
            sns.histplot(x = col, data = df, ax = axs[axs_idx], edgecolor = 'k')
        axs[axs_idx].set_xlabel(col)
        axs[axs_idx].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    
def plot_col_vs_num_target(X, y, col_list = None, figsize = (20, 20), num_graph_cols = 4):
    '''
    This function plots the relationship between each feature in the list and the target. 
    If the column is numerical, a scatterplot is plotted. If the column is an object, a side-by-side boxplot is plotted.
    
    X: Pandas DataFrame of features.
    
    y: Pandas Series of y values
    
    col_list: List of columns to plot. Default is set to None, and in this case, all the columns are considered.
    
    num_graph_cols: The number of columns in the subplot. Must be a positive integer value.
    
    figsize: Tuple of integer dimensions for the size of the figure.
    
    '''
    if col_list is None:
        col_list = list(X.columns)
        
    df = pd.concat([X, y], axis = 1)
    if len(col_list) % num_graph_cols == 0:
        num_rows = len(col_list) // num_graph_cols
    else:
        num_rows = (len(col_list) // num_graph_cols) + 1
    fig, axs = plt.subplots(num_rows, num_graph_cols, figsize = figsize)
    
    for idx, col in enumerate(col_list):
        if df[col].dtypes == float:
            graph_type = 'regplot'
        elif df[col].dtypes == object:
            graph_type = 'boxplot'
        elif df[col].dtypes == int:
            graph_type = 'regplot'
        axs_idx = (idx// num_graph_cols, idx % num_graph_cols)
        if graph_type == 'boxplot':
            sns.boxplot(x = col, y = y.name, data = df, ax = axs[axs_idx])
        elif graph_type == 'regplot':
             sns.regplot(x = col, y = y.name, data = df, ax = axs[axs_idx])
        axs[axs_idx].set_xlabel(col)
        axs[axs_idx].set_ylabel('Sale Price')
    plt.show()
    plt.tight_layout()

    
def plot_col_vs_cat_target(X, y, col_list = None, num_cols = 4, figsize = (20, 30)):
    '''
    This function plots the relationship between each feature in the list and the categorical target. 
    If the column is numerical, a side-by-side boxplot is plotted. If the column is an object, a barplot of value counts is plotted.
    
    X: Pandas DataFrame of features.
    
    y: Pandas Series of y values
    
    col_list: List of columns to plot. Default is set to None, and in this case, all the columns are considered.
    
    num_graph_cols: The number of columns in the subplot. Must be a positive integer value. Default is set to 4.
    
    figsize: Tuple of integer dimensions for the size of the figure.
    
    '''
    if col_list is None:
        col_list = list(X.columns)
        
    #Determining dimensions of subplots
    df = pd.concat([X, y], axis = 1)
    if len(col_list)% num_cols == 0:
        num_rows = len(col_list)//num_cols
    else:
        num_rows = len(col_list)// num_cols + 1
    
    #Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize = figsize)
    
    #Plotting each column
    for i, col in enumerate(col_list):
        if num_rows == 1:
            axs_i = axs[i%num_cols]
        else: 
            axs_i = axs[i//num_cols, i%num_cols]
        if df[col].dtypes == int:
            df_rel = df.groupby(y.name)[col].value_counts(normalize=True).rename('percent').reset_index()
            sns.barplot(data = df_rel, x = y.name, y = 'percent', hue = col,ax = axs_i, edgecolor = 'k'), 
        else:
            sns.boxplot(data = df, x = y.name, y = col,ax = axs_i), 
            axs_i.set_title('Distribution for {}'.format(col))
        axs_i.set_title('Distribution for {}'.format(col))
    plt.tight_layout()
    
    
def check_quasi_constant_features(df, col_list = None):
    '''
    This function takes the DataFrame of features and outputs a DataFrame that shows the category with the largest percentage of values for each feature.
    
    df: Pandas DataFrame of features.
    
    col_list: List of columns to plot. Default is set to None, and in this case, all the integer and object columns are considered.
   
    returns: A DataFrame that outputs the largest category of each column, and its corresponding percentage. The index of the DataFrame is the column names.
    '''
    if col_list is None:
        [col for col in X.columns if X[col].dtypes == int or X[col].dtypes == object]
        
    cat_name = []
    percentage = []
    for col in col_list:
        top_cat = df[col].value_counts(normalize = True).head(1)
        cat_name.append(top_cat.index[0])
        percentage.append(top_cat.values[0])
    return pd.DataFrame({'cat_name': cat_name, 
                         'percentage': percentage}, 
                        index = col_list).sort_values(by = 'percentage', ascending = False)


def corr_feat_target(X, y, col_list = None, figsize = (14, 6)):
    '''
    This function plots a bar graph showing the correlation between each numerical feature and the target.
    
    X: Pandas DataFrame of features.
    
    y: Pandas Series of y values
    
    col_list: List of columns to plot. Default is set to None, and in this case, all the numerical columns are considered.
   
    figsize: Tuple of integer dimensions for the size of the figure.
    
    '''
    if col_list is None:
        col_list = [col for col in X.columns if X[col].dtypes == int or X[col].dtypes == float]
    plt.figure(figsize = figsize)
    df = pd.concat([X, y], axis = 1)
    df[col_list].corrwith(df[y.name]).plot(kind = 'bar', edgecolor = 'black')
    plt.title('Correlation between features and target')
    plt.ylabel('Correlation')
    plt.show()

    
def correlation_matrix(df, col_list = None, figsize = (24, 20)):
    '''
    This function takes the DataFrame of features and outputs an lower triangle correlation heatmap.
    
    df: Pandas DataFrame of features.
    
    col_list: List of columns to plot. Default is set to None, and in this case, all the integer and object columns are considered.
    
    figsize: Tuple of integer dimensions for the size of the figure.
    '''
    if col_list is None:
        col_list = [col for col in df.columns if df[col].dtypes == int or df[col].dtypes == float]
    plt.figure(figsize = figsize)
    mask = np.triu(np.ones_like(df[col_list].corr()))
    sns.heatmap(df[col_list].corr(), annot = True, mask = mask)

    
def association_features_binary_target(X, y, col_list = None, alpha = 0.05):
    '''
    This returns the names of the features that have a direct association with the target, and the corresponding p-values of the chi-squared tests and t-tests.
    
    X: Pandas DataFrame of features.
    
    y: Pandas Series of y values
    
    col_list: List of columns to plot. Default is set to None, and in this case, all the numerical columns are considered.
   
    alpha: Float value in (0,1) that gives the p-value to use for the statistical test.
    
    returns: A Series that shows columns that are directly associated with the target variable and the corresponding p-values of the statistical test.
    
    '''
    if col_list is None:
        col_list = list(X.columns)
    dict_p_vals = {}
    for col in col_list:
        if X[col].dtypes == int or X[col].dtypes == float:
            samples_dict = {}
            vals = y.unique()
            for val in vals:
                samples_dict[val] = X[y == val][col].values
            results = f_oneway(samples_dict[vals[0]], samples_dict[vals[1]])[1]
            dict_p_vals[col] = results
        else:
            contingency = pd.crosstab(X[col],y)
            dict_p_vals[col] = chi2_contingency(contingency)[1]
            dict_p_vals[col] = results.pvalue
    p_vals_series = pd.Series(dict_p_vals).sort_values()
    return p_vals_series[p_vals_series<=alpha]