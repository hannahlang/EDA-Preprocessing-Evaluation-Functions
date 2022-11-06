## EDA Functions

### plot_col_distributions ***(df, col_list = None, num_graph_cols = 4, figsize = (20, 20))***:

A function that takes the DataFrame of features and plots the distribution of the features in subplots. 
If the column is an integer or object column it plots a bar graph. If the column is a float column, it plots a histogram.

<b>df: </b> ***Pandas DataFrame (n_samples, n_features)***

<b>col_list: </b> ***List or None, default = None***

Include a list of strings of column names:
- list: list or arrany of valid column names
- None: When set to none, all of the columns will be plotted.

<b>num_graph_cols: </b> ***int***

The number of columns in the subplot. Must be a positive integer value.

<b>figsize: </b> ***tuple (n,m), default = (20, 20)***

Tuple of integer dimensions for the size of the figure.

<b>Returns</b> Subplot

A subplot of the distributions of each column.

### plot_col_vs_num_target ***(X, y, col_list = None, figsize = (20, 20), num_graph_cols = 4)***:

A function that plots the relationship between each feature in the list and the target. 
If the column is numerical, a scatterplot is plotted. If the column is an object, a side-by-side boxplot is plotted.
    
<b>X: </b> ***Pandas DataFrame (n_samples, n_features)***

A Pandas DataFrame of features.
    
<b>y: </b> ***Pandas Series (n_samples,)***

A Pandas Series of target values.
    
<b>col_list: </b> ***List or None, default = None***

Include a list of strings of column names:
- list: list or arrany of valid column names
- None: When set to none, all of the columns will be plotted.

<b>num_graph_cols: </b> ***int, default = 4***

The number of columns in the subplot. Must be a positive integer value.

<b>figsize: </b> ***tuple (n,m), default = (20, 20)***

Tuple of integer dimensions for the size of the figure.

<b>Returns</b> Subplot

A subplot of the distributions of each column.


### plot_col_vs_cat_target ***(X, y, col_list = None, num_cols = 4, figsize = (20, 30))***:

A function that plots the relationship between each feature in the list and the categorical target. 
If the column is numerical, a side-by-side boxplot is plotted. If the column is an object, a barplot of value counts is plotted.
    
<b>X: </b> ***Pandas DataFrame (n_samples, n_features)***

A Pandas DataFrame of features.
    
<b>y: </b> ***Pandas Series (n_samples,)***

A Pandas Series of target values.
    
<b>col_list: </b> ***List or None, default = None***

Include a list of strings of column names:
- list: list or arrany of valid column names
- None: When set to none, all of the columns will be plotted.

<b>num_cols: </b> ***int, default = 4***

The number of columns in the subplot. Must be a positive integer value.

<b>figsize: </b> ***tuple (n,m), default = (20, 30)***

Tuple of integer dimensions for the size of the figure.

<b>Returns</b> Subplot

A subplot of the distributions of each column against the target.

### check_quasi_constant_features ***(df, col_list = None)***:

This function takes the DataFrame of features and outputs a DataFrame that shows the category with the largest percentage of values for each feature.
    
<b>df: </b> ***Pandas DataFrame (n_samples, n_features)***

<b>col_list: </b> ***List or None, default = None***

Include a list of strings of column names:
- list: list or arrany of valid column names
- None: When set to none, all of the integer and object columbs will be used.
    
<b> Returns: </b> ***DataFrame***
A DataFrame that shows the category with the largest percentage of values for each feature.

### corr_feat_target ***(X, y, col_list = None, figsize = (14, 6))***:

This function plots a bar graph showing the correlation between each numerical feature and the target.

<b>X: </b> ***Pandas DataFrame (n_samples, n_features)***

A Pandas DataFrame of features.
    
<b>y: </b> ***Pandas Series (n_samples,)***

A Pandas Series of target values.
    
<b>col_list: </b> ***List or None, default = None***

Include a list of strings of column names:
- list: list or arrany of valid column names
- None: When set to none, all of the numerical columns will be plotted.

<b>figsize: </b> ***tuple (n,m), default = (14, 6)***

Tuple of integer dimensions for the size of the figure.

<b>Returns</b> Bar Graph

A bargraph of the correlation of each column with the target.

### correlation_matrix ***(df, col_list = None, figsize = (24, 20))***:

This function takes the DataFrame of features and outputs an lower triangle correlation heatmap.
    
<b>df: </b> ***Pandas DataFrame (n_samples, n_features)***

Pandas DataFrame of features
    
<b>col_list: </b> ***List or None, default = None***

Include a list of strings of column names:
- list: list or arrany of valid column names
- None: When set to none, all of the numerical columns will be plotted.
    
<b>figsize: </b> ***tuple (n,m), default = (24, 20)***

<b> Returns </b> Correlation Matrix

### association_features_binary_target ***(X, y, col_list = None, alpha = 0.05)***:

A function that returns the names of the features that have a direct association with the target, and the corresponding p-values of the chi-squared tests and t-tests.
    
<b>X: </b> ***Pandas DataFrame (n_samples, n_features)***

A Pandas DataFrame of features.
    
<b>y: </b> ***Pandas Series (n_samples,)***

A Pandas Series of target values.

<b>col_list: </b> ***List or None, default = None***

Include a list of strings of column names:
- list: list or arrany of valid column names
- None: When set to none, all of the columns will be used.
   
<b>alpha: </b> ***float in (0,1), default = 0.05*** 

A value that gives the p-value to use for the statistical test.
    
<b>returns: </b> ***Pandas Series (n_features,)***

A Series that shows columns that are directly associated with the target variable and the corresponding p-values of the statistical test.
