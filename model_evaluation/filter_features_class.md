# FilterFeatures():
A class that determines features to remove from the data based on feature importance

## Parameters:

**model:** ***A pre-trained model***: <i>{ogisticRegression, Ridge, RandomForestClassifier, RandomForestRegressor, LGBMClassifier, LGBMRegressor}</i>

## Methods:

### get_feature_importance(threshold)

A function that gets the feature importance for the model, plots the feature importance values, and finds an "elbow point" for the cut-off of feature importance.
        
#### Parameters:

**threshold:** ***float (-inf, inf)*** 

A float value indicating the minimum threshold for the feature importance values in the model.
        
#### Returns: <i>Pandas Series (n_features,), float, list</i>

A feature importance series (indexed by feature), the feature importance value for the "elbow point", a list of features that pass the feature importance threshold.
    
### elbow_pt_feat_imp(feat_imp_array)

A function that takes an array of feature importances and outputs the value of the elbow point. It also plots a graph of feature importances that marks the elbow point.
        
#### Parameters:

**feat_imp_array:** ***array (n_features,)*** 

An array of feature importances.
        
#### Returns: float

The feature importance value of the "elbow point".


### distance_btwn_pt_line(lp1, lp2, p0)

A function that gets the feature importance for the model, plots the feature importance values, and finds an "elbow point" for the cut-off of feature importance.
        
#### Parameters:

**lp1:** ***array (2,)*** 

An array that gives the coordinates of the first two-dimensional endpoint of a line.

**lp2:** ***array (2,)*** 

An array that gives the coordinates of the second two-dimensional endpoint of a line.

**p0:** ***array (2,)*** 

An array that gives the coordinates of the point that we want to calculate the distance of to the line.
        
#### Returns: <i>float</i>

The distance between the point and the line.
    
### elbow_pt_feat_imp(feat_imp_array)

A function that takes an array of feature importances and outputs the value of the elbow point. It also plots a graph of feature importances that marks the elbow point.
        
#### Parameters:

**feat_imp_array:** ***array (n_features,)*** 

An array of feature importances.
        
#### Returns: float

The feature importance value of the "elbow point".
      
