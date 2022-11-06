# OneHotEncoder()

## Parameters

<b> col_list: </b> ***list or None, default = None***

Include a list of strings of column names:
- list: list or arrany of valid column names
- None: When set to none, all the categorical columns will be one-hot-encoded.

<b>drop_last: </b> ***bool, default = True***

When set to True, the category with the lowest frequency in the column will not have its own column. When set to False, all categories in the column will have their own column.

<b>inplace: </b> ***bool, default = False***

When set to True, the original DataFrame will be overwritten.

## Attributes
<b>col_cat_dict: </b> ***dict***

A dictionary that stores the categories of each object column in the training data.

<b>cat_cols: </b> ***list***

A list of object columns to be transformed.

## Methods

### fit(X, y)
#### Parameters: 
<b> X: </b> ***Pandas DataFrame of shape (n_samples, n_features)***

Training DataFrame, where n_samples is the number of samples and n_features is the number of features/columns to transform.

<b> y: </b> ***Pandas Series of shape (n_samples,)***

#### Returns: self

Fitted estimator.

### transform(X)
#### Parameters: 
<b> X: </b> ***Pandas DataFrame of shape (n_samples, n_features)***

DataFrame to transform, where n_samples is the number of samples and n_features is the number of features/columns to transform (must be the same as the training set).

#### Returns: X_transformed

Transformed DataFrame with the one-hot-encoded columns appended to the original DataFrame. The original object columns themselves are dropped.

### one_hot_encode(col, cat_names, df)
#### Parameters: 
<b> col: </b> ***str {column name written as a string}***
<b> cat_names: </b> ***list {list of column names written as strings}***
<b> df: </b> ***Pandas DataFrame {DataFrame of features}***

#### Returns: df_transformed

A DataFrame with the one-hot-encoded column appended to the original DataFrame. The column itself is dropped.

