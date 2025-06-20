import pandas
import numpy as np
from sklearn.linear_model import LinearRegression

#  Age regression
def age_regression(biomarker_data, age_data):

    # Reshape age data for sklearn
    X = age_data.values.reshape(-1, 1)
    
    # Initialize model
    model = LinearRegression()
    
    # Container for residuals
    residuals_dict = {}
    
    # Perform regression for each biomarker
    for col in biomarker_data.columns:
        # Get target values, dropping NaN
        y = biomarker_data[col]
        mask = ~(y.isna() | age_data.isna())
        
        if mask.sum() > 0:  # Only perform regression if we have valid data
            # Fit model
            model.fit(X[mask], y[mask])
            
            # Predict for all points
            y_pred = pandas.Series(index=y.index, data=np.nan)
            y_pred[mask] = model.predict(X[mask])
            
            # Calculate residuals
            residuals_dict[col] = y - y_pred
        else:
            residuals_dict[col] = pandas.Series(np.nan, index=y.index)
    
    # Create DataFrame with residuals
    biomarker_data_residuals = pandas.DataFrame(residuals_dict)
    
    return biomarker_data_residuals

# Z-scoring
def z_scoring(biomarker_data, diagnosis_data):

    biomarker_data['DIAGNOSIS'] = diagnosis_data
    
    # extract data for control subjects
    biomarker_data_control = biomarker_data[biomarker_data['DIAGNOSIS'] == 'CN']

    for col in [col for col in biomarker_data.columns if col != 'DIAGNOSIS']:        
        # compute the mean and standard deviation of the control population
        mean_control = biomarker_data_control[[col]].mean()
        std_control = biomarker_data_control[[col]].std()

        # z-score the data
        biomarker_data[[col]] = (biomarker_data[[col]]-mean_control)/std_control

    biomarker_data = biomarker_data.drop('DIAGNOSIS', axis=1)
    return biomarker_data