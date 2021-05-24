import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit,# learning_curve

# for replicability purposes
tf.random.set_seed(91195003) 
np.random.seed(91190530)
# for an easy reset backend session state 
tf.keras.backend.clear_session()

'''
Load dataset
'''
def load_data(path):
    return pd.read_csv(path)

'''
Prepare the data for the LSTM
'''
def prepare_data(df):
    df_aux = df.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'], inplace=False)
    df_aux = df_aux.sum().to_frame()
    df_aux.columns = ['cases']
    return df_aux

'''
Split the data into training and validation sets 
'''
def split_data(data, val_percentage):
    train_idx = np.arange(0, int(len(data) * (100 - val_percentage) / 100))
    val_idx = np.arange(int(len(data) * (100 - val_percentage) / 100 + 1), len(data)) 
    return train_idx, val_idx

'''
Normalise the data
'''
def normalise_data(df, norm_range=(-1, 1)):
    # range = [-1, 1] for LSTM due to the internal use of tanh by the memory cell 
    scaler = MinMaxScaler(feature_range=norm_range)
    df[['cases']] = scaler.fit_transform(df[['cases']])
    return scaler

'''
Plot time series data
'''
def plot_data(data): 
    plt.figure(figsize=(8,6)) 
    plt.plot(range(len(data)), data) 
    plt.title('Confirmed Cases of COVID-19') 
    plt.ylabel('Cases')
    plt.xlabel('Days') 
    plt.show()

'''
Prepare the training data for the LSTM
'''
def to_supervised(df, timesteps):
    data = df.values
    x, y = list(), list()
    # iterate over the training set to create x and y 
    dataset_size = len(data)
    for curr_pos in range(dataset_size):
        # end of the input sequence corresponds to the current position + the number of timesteps of the input sequence
        input_index = curr_pos + timesteps
        # end of the labels corresponds to the end of the input sequence + 1
        label_index = input_index + 1
        # if we have enough data for this sequence
        if label_index < dataset_size:
            x.append(data[curr_pos:input_index, :])
            y.append(data[input_index:label_index, 0])
    # using np.float32 for GPU performance
    return np.array(x).astype('float32'), np.array(y).astype('float32')

'''
Loss function (root mean square error)
'''
def rmse(y_real, y_prediction):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_prediction - y_real)))

'''
Build the model
'''
def build_model(timesteps, features, h_neurons=64, activation='tanh'): 
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features))) 
    model.add(tf.keras.layers.Dense(h_neurons, activation=activation)) 
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    # plot model summary (and save it as PNG)
    #tf.keras.utils.plot_model(model, 'covid19_model.png', show_shapes=True)
    
    return model

'''
Plot learning curves
'''
def plot_learning_curves(history, epochs):
    # accuracy and losses
    print(history)
    #accuracy = history.history['accuracy']
    #val_accuracy = history.history['val_accuracy']
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    # creating figure
    plt.figure(figsize=(8, 8))
    plt.subplot[1, 2, 1]
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training/Validation Accuracy')
    plt.subplot[1, 2, 2]
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training/Validation Loss')
    plt.show()

'''
Compile and fit the model
'''
def compile_and_fit(model, epochs, batch_size): 
    # compile the model
    model.compile(loss=rmse, optimizer=tf.keras.optimizers.Adam(), metrics=['mae', rmse])
    # fit the model
    hist_list = list()
    loss_list = list()
    # time series cross validator
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    for train_index, test_index in tscv.split(x):
        train_idx, val_idx = split_data(train_index, 10) # split into training and validation sets
        # build the data
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        x_test, y_test = x[test_index], y[test_index]
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, shuffle=False) 
        metrics = model.evaluate(x_test, y_test)
        hist_list.append(history)
        loss_list.append(metrics[2]) 
    #plot_learning_curves(hist_list, approach='history') 
    #plot_learning_curves(loss_list, approach='loss') 
    
    return model, hist_list, loss_list

'''
Recursive multi-step forecast
'''
def forecast(model, df, timesteps, multisteps, scaler):
    input_seq = df[-timesteps:].values # get the last known sequence
    inp = input_seq
    forecasts = list()
    for step in range(1, multisteps+1):
        # to do ...
        print(inp.shape)
        prediction = model.predict(inp)
        forecasts.append(prediction)
        inp = np.append(inp, [[prediction]], axis=0) # insert prediction to the sequence
        np.delete(inp, 0)                            # remove oldest value of the sequence
    return forecasts

'''
Plot forecasts
'''
def plot_forecast(data, forecasts):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(data)), data, color='green', label='Confirmed')
    plt.plot(range(len(data)-1, len(data)+len(forecasts)-1), forecasts, color='red', label='Forecasts') 
    plt.title('Confirmed Cases of COVID-19')
    plt.ylabel('Cases')
    plt.xlabel('Days')
    plt.legend()
    plt.show()

'''
Main execution
'''

timesteps = 5       # number of days that make up a sequence
univariate = 1      # number of features used by the model
multisteps = 5      # number of days to forecast
cv_splits = 3       # time series cross validator
epochs = 25         # number of iterations
batch_size = 7      # number of sequences

df = load_data('/Users/goncalo/Documents/University/CSC/Classes/Class 5/dataset.csv')
df = prepare_data(df)
plot_data(df)
scaler = normalise_data(df) # scaling data to [-1, 1]

x, y = to_supervised(df, timesteps)
print("Training shape:", x.shape)
print("Training labels shape:", y.shape)

model = build_model(timesteps, univariate)
model, hist_list, loss_list = compile_and_fit(model, epochs, batch_size)

forecasts = forecast(model, df, timesteps, multisteps, scaler)
#plot_forecast(df_data, forecasts)