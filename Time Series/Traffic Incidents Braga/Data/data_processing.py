import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from numpy import nan

cv_splits = 3  # time series cross validator

'''
Load dataset
'''
def load_data(path):
    raw_data = pd.read_csv(path,infer_datetime_format=True)
    return raw_data 

'''
Split data into training and validation sets 
'''
def split_data(training, perc=10):
    train_raw = np.arange(0, int(len(training)*(100-perc)/100))                    # contains the first 90% of the data
    validation_raw = np.arange(int(len(training)*(100-perc)/100+1), len(training)) # contains the last 10%
    return train_raw, validation_raw 

'''
Convert data to univariate, using a single feature (incident_date) as input
'''
def to_univariate(df):
    df_uni = df.drop(columns=['cause_of_incident', 'city_name', 'description', 'cause_of_incident', 'from_road', 'to_road', 
                              'affected_roads', 'incident_category_desc','magnitude_of_delay_desc', 'length_in_meters',
                              'delay_in_seconds', 'latitude','longitude'])
    return df_uni

'''
Convert data to multiariate, using multiple features (incident, length_in_meters, latitude, longitude) as input
'''
def to_multivariate(df):
    df_uni = df.drop(columns=['cause_of_incident', 'city_name', 'description', 'cause_of_incident', 'from_road', 'to_road', 
                              'affected_roads', 'incident_category_desc','magnitude_of_delay_desc','delay_in_seconds', ])
    return df_uni

'''
Prepare data to have the number of daily incidents
'''
def to_daily(method, df):
    if method == 'univariate':
        df_uni = to_univariate(df)
        df_uni['incident_date'] = df_uni['incident_date'].str[:10]                # delete the last 10 characters
        df_uni['Incidents'] = pd.DataFrame([1 for x in range(len(df_uni.index))]) # create a column with 1 to sum the incidents per day
        df_uni = df_uni.set_index('incident_date')                                # set the column incident_date to index
        df_uni.index = pd.to_datetime(df_uni.index)                               # convert the date in index from string to Date type
        daily_groups = df_uni.resample('D')                                       # sum groupy by day
        daily_data = daily_groups.sum()
    elif method == 'multivariate':
         df_uni = to_multivariate(df)
         df_uni['incident_date'] = df_uni['incident_date'].str[:10]                # delete the last 10 characters
         df_uni['Incidents'] = pd.DataFrame([1 for x in range(len(df_uni.index))]) # create a column with 1 to sum the incidents per day
         df_uni = df_uni.set_index('incident_date')                                # set the column incident_date to index
         df_uni.index = pd.to_datetime(df_uni.index)                               # convert the date in index from string to Date type
         daily_groups = df_uni.resample('D')                                       # sum groupy by day
         daily_data_1 = daily_groups['Incidents'].sum()
         daily_data_2 = daily_groups['length_in_meters', 'latitude','longitude'].median() # median by date
         daily_data = pd.concat([daily_data_1, daily_data_2], axis=1) # concatenate the incident sum, length_in_temers, latitude and longitude median
    return daily_data

'''
Deal with missing values in the data
'''
def missing_values(method, missing_method, df):
    if missing_method == 'dropout':
        df = df.replace(0, np.nan)        # replace instances with 0 incidents with NaN
        df = df.dropna(how='all', axis=0) # remove all instances with NaN
    elif missing_method == 'masking':
        df = df.replace(0, -99)
        if method == 'multivariate':
            df.loc[df.Incidents == -99, 'length_in_meters'] = -99
            df.loc[df.Incidents == -99, 'latitude'] = -99
            df.loc[df.Incidents == -99, 'longitude'] = -99
    elif missing_method == 'interpolate':
        df = df.replace(0, np.nan) # replace instances with 0 incidents with NaN
        df = df.interpolate(method='linear', limit_direction='forward').astype(int)
    return df


'''
Remove Outliers
'''
def remove_outlier(df_uni):
    q1 = df_uni['Incidents'].quantile(0.25)
    q3 = df_uni['Incidents'].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_uni.loc[(df_uni['Incidents'] > fence_low) & (df_uni['Incidents'] < fence_high)]
    return df_out

'''
Normalize the data to the range [-1, 1]
'''
def data_normalization(method, df, norm_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=norm_range)
    if method == 'multivariate':
        df[['Incidents','length_in_meters', 'latitude','longitude']] = scaler.fit_transform(df) # Multivariate
    elif method == 'univariate':
        df[['Incidents']] = scaler.fit_transform(df) # Univariate
    return scaler

'''
Denormalize the data from a scaler
'''
def data_denormalization(method, df, scaler):
    values = scaler.inverse_transform(df)
    values = values.flatten()
    return values.astype(int)

'''
Convert the data to supervised
'''
def to_supervised(df, timesteps):
    data = df.values
    x, y = list (), list ()
    dataset_size = len(data)
    for curr_pos in range(dataset_size):
        input_index = curr_pos + timesteps
        label_index = input_index +1
        if label_index < dataset_size:
            x.append(data[curr_pos:input_index,:])
            y.append(data[curr_pos:label_index,0])
    return np.array(x).astype('float32'), np.array(y).astype('float32')

'''
Prepare the training, validation and testing sets from a configuration
'''
def prepare_train(df, config):
    timesteps = config[0]
    X, y = to_supervised(df, timesteps)
    # time series cross validator
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    for train_index, test_index in tscv.split(X):
        train_idx, val_idx = split_data(train_index, perc=10) # further split into training and validation sets
        # build data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_index], y[test_index]
        #model.compile(loss = rmse, optimizer = tf.keras.optimizers.Adam(), metrics = ['mae', rmse])
    return X_train, y_train, X_val, y_val, X_test, y_test

'''
Plot Time Series data
'''
def plot_incidents(data):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(data)), data)
    plt.title('Number of incidents per Day')
    plt.ylabel('Incidents')
    plt.xlabel('Days')
    plt.show()
