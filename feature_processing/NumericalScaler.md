# NumericalScaler()

## Parameters
<b> method: </b> ***{'min_max_scaling', 'standardization'}, default = 'min_max_scaling'***

Specify the method of the scaler:
- 'min_max_scaling': Features are scaled according to the min/max values of the training set. $\frac{value-min}{max-min}$
- 'standardization': Features are scaled according to the mean and standard deviation of the training set. $\frac{x-\mu}{\sigma}$

<b> col_list: </b> ***list or None, default = None***

Include a list of strings of column names:
- list: list or arrany of valid column names
- None: When set to none, all the numerical columns will be scaled.

<b>inplace: </b> ***bool, default = False***

When set to True, the original DataFrame will be overwritten.

## Attributes
<b>num_col_dict: </b> ***dict***

A dictionary that stores the min/max or mean/stdev of each numerical column in the training data.

<b>num_cols: </b> ***list***

A list of numerical columns to be transformed.

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

Transformed DataFrame

### inverse_transform(X)
#### Parameters: 
<b> X: </b> ***Pandas DataFrame of shape (n_samples, n_features)***

Transformed DataFrame reverse transform, where n_samples is the number of samples and n_features is the number of features/columns that were transformed (must be the same as the training set).

#### Returns: X_reverse_transformed.

Reverse transformed DataFrame

