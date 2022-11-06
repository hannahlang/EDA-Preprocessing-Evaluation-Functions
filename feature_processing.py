import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class NumericalScaler(BaseEstimator, TransformerMixin):
    '''
    A class that scales numerical variables for regression models. Method is min/max scaling or standardization.
    
    '''
    def __init__(self, method = 'min_max_scaling', col_list = None, inplace = False):
        '''
        method: Either 'min_max_scaling' or 'standardization'. Default is 'min_max_scaling'.
        
        col_list: List of column names to be transformed. Default is 'None', which means that all numerical columns are taken into consideration.
        
        inplace: True/False. Default value is False, and in this case the original DataFrame is not overwritten. When inplace is set to True, the original DataFrame is overwritten.
        '''
        self.inplace = inplace
        self.col_list = col_list
        self.method = method
        
    def fit(self, X, y):
        '''
        A function that fits the scaler to the values in the training data.
        
        X: A Series or DataFrame of features.
        
        y: A Series of y-values.
        '''
        self.num_col_dict = {}
        if type(X) == pd.core.series.Series:
            self.type = 'Series'
            X = X.to_frame()
        else:
            self.type = 'DataFrame'
        if self.col_list is None: 
            self.num_cols = [col for col in X.columns if X[col].dtypes == int]
            self.num_cols += [col for col in X.columns if X[col].dtypes == float]
        else:
            if type(self.col_list) != list:
                raise ValueError('col_list must be entered as a list of strings')
            self.num_cols = self.col_list
        if self.method == 'min_max_scaling':
            for col in self.num_cols:
                parameters = {}
                parameters['min'] = X[col].min()
                parameters['max'] = X[col].max()
                self.num_col_dict[col] = parameters
        elif self.method == 'standardization':
            for col in self.num_cols:
                parameters = {}
                parameters['mean'] = X[col].mean()
                parameters['stdev'] = X[col].std()
                self.num_col_dict[col] = parameters
        else:
            raise ValueWarning('Valid methods are min_max_scaling or standardization')
        return self
    
    def transform(self, X):
        '''
        A function that transforms the variables in the DataFrame, based on the parameters from the train set.
        
        X: A DataFrame of features.
        
        returns: A transformed DataFrame of features.
        '''
        if self.inplace:
            transformed_df = X
        else:
            transformed_df = X.copy()
        if self.type == 'Series':
            transformed_df = transformed_df.to_frame()
        if self.method == 'min_max_scaling':
            for col in self.num_cols:
                parameters = self.num_col_dict[col]
                transformed_df[col] = (transformed_df[col] - parameters['min'])/(parameters['max'] - parameters['min'])
        else:
            for col in self.num_cols:
                parameters = self.num_col_dict[col]
                transformed_df[col] = (transformed_df[col] - parameters['mean'])/ (parameters['stdev'])
        if self.type == 'Series':
            transformed_df = transformed_df[self.num_cols[0]]
        return transformed_df
    
    def inverse_transform(self, X):
        '''
        A function that reverses the process of scaling on the features in the DataFrame X.
        
        X: A DataFrame of features.
        
        returns: A reverse transformed DataFrame of features.
        '''
        if self.inplace:
            inv_transformed_df = X
        else:
            inv_transformed_df = df.copy()
        if self.col_list is None:
            self.col_list = self.num_cols
        if type(self.col_list) != list:
            raise ValueWarning('col_list must be given as a list of column names.')
        if self.method == 'min_max_scaling':
            for col in self.col_list:
                parameters = self.num_col_dict[col]
                inv_transformed_df[col] = inv_transformed_df[col]* (parameters['max'] - parameters['min']) + parameters['min']
        else:
            for col in col_list:
                parameters = self.num_col_dict[col]
                inv_transformed_df[col] = inv_transformed_df[col]*parameters['stdev'] + parameters['mean']
        return inv_transformed_df

    
class OneHotEncoder(BaseEstimator, TransformerMixin):
    '''
    A class the OneHotEncodes categorical variables.
    '''
    def __init__(self, col_list = None, drop_last = True, inplace = False):
        '''
        col_list: List of column names to be transformed. Default is 'None', which means that all numerical columns are taken into consideration.
        
        drop_last: True/False. Default is True. When drop_last is set to True, the category with the lowest frequency of values does not have its own column. When drop_last is set to False, all categories are given their own column.
        
        inplace: True/False. Default value is False, and in this case the original DataFrame is not overwritten. When inplace is set to True, the original DataFrame is overwritten.
        '''
        self.col_list = col_list
        self.drop_last = drop_last
        self.inplace = inplace
        
    def fit(self, X, y):
        '''
        A function that fits the one-hot-encoder to the categorical values in the columns in the training data.
        
        X: A Series or DataFrame of features.
        
        y: A Series of y-values.
        '''
        self.col_cat_dict = {}
        if self.col_list is None:
            self.cat_cols = [col for col in X.columns if X[col].dtypes == object]
        else:
            if type(self.col_list) != list:
                raise ValueError('col_list must be entered as a list of strings')
            self.cat_cols = self.col_list
        for col in self.cat_cols:
            self.col_cat_dict[col] = X[col].unique()
        return self
    
    def transform(self, X):
        '''
        A function that transforms the variables in the DataFrame, based on the categories from the train set.
        
        X: A DataFrame of features.
        
        returns: A one-hot-encoded DataFrame of features. The columns that are one-hot-encoded are dropped.
        '''
        if self.inplace:
            transformed_df = X
        else:
            transformed_df = X.copy()
        for col in self.cat_cols:
            transformed_df = self.one_hot_encode(col, self.col_cat_dict[col], transformed_df)
        return transformed_df
    
    def one_hot_encode(self, col, cat_names, df):
        '''
        A function that one-hot-encodes a column.
        
        col: A string name for a column in the DataFrame.
        
        cat_names: A list of strings of the different category names in the column.
        
        df: A DataFrame of features.
        
        returns: A DataFrame with the one-hot-encoded column appended to the original DataFrame. The column itself is dropped.
        '''
        new_col_names = [col +'_' + str(name) for name in cat_names]
        one_hot_encoding_dict = {}
        if self.drop_last:
            for i in range(len(cat_names)-1):
                one_hot_encoding_dict[new_col_names[i]] = (df[col].values == cat_names[i]) * 1
        else:
            for i in range(len(bcat_names)):
                one_hot_encoding_dict[new_col_names[i]] = (df[col].values == cat_names[i]) * 1
        one_hot_df =  pd.DataFrame(one_hot_encoding_dict, index = df.index)
        df.drop(col, axis = 1, inplace = True)
        df = pd.concat([df, one_hot_df], axis = 1)
        return df