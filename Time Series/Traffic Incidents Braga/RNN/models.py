import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

from Data.data_processing import to_supervised, prepare_train

# for replicability purposes
tf.random.set_seed(91195003) 
np.random.seed(91190530)
# for an easy reset backend session state 
tf.keras.backend.clear_session()

multisteps = 5 # number of days to forecast (we will forecast the next 5 days)

'''
Define loss function (root mean square error)
'''
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

'''
Build a LSTM model from a configuration
'''
def build_lstm(config):
    timesteps, features, h_neurons, activations = config[0:4]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation=activations))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # show model summary (and save it as PNG)
    tf.keras.utils.plot_model(model, 'Traffic_lstm.png', show_shapes=True)
    return model

''''
Build a LSTM Autoencoder model from a configuration
'''
def build_lstm_autoencoder(df,config):
    timesteps, features, h_neurons,activations, = config[0:4]
    X_train,y_train,X_val, y_val,X_test, y_test = prepare_train(df, config)
    n_output = y_train.shape[1]
    model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features))) #If use Drop, or interpolate use this as input
    model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(timesteps, features))) #If use Masking use this as input
    model.add(tf.keras.layers.LSTM(100, activation=activations))
    model.add(tf.keras.layers.RepeatVector(n_output))
    model.add(tf.keras.layers.LSTM(h_neurons,return_sequences=True))
    model.add(tf.keras.layers.Dense(h_neurons, activation=activations))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    #model summary (and save it as PNG)
    tf.keras.utils.plot_model(model, 'Traffic_model.png', show_shapes=True)
    model.compile(loss = rmse, optimizer = 'Adam', metrics = ['mae', rmse])
    return model
'''
Build a GRU model from a configuration
'''
def build_gru(config):
    timesteps, features, h_neurons, activations= config[0:4]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation=activations))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # show model summary (and save it as PNG)
    tf.keras.utils.plot_model(model, 'Traffic_gru.png', show_shapes=True)
    return model

'''
Build a GRU Autoencoder model from a configuration
'''
def build_gru_autoencoder(df,config):
    timesteps, features, h_neurons,activations, epochs, batch_size = config
    X_train,y_train,X_val, y_val,X_test, y_test = prepare_train(df, config)
    n_output = y_train.shape[1]
    model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features))) #If use Drop, or interpolate use this as input
    model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(timesteps, features))) #If use Masking use this as input
    model.add(tf.keras.layers.GRU(100, activation=activations))
    model.add(tf.keras.layers.RepeatVector(n_output))
    model.add(tf.keras.layers.GRU(h_neurons,return_sequences=True))
    model.add(tf.keras.layers.Dense(h_neurons, activation=activations))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    #model summary (and save it as PNG)
    tf.keras.utils.plot_model(model, 'Traffic_model.png', show_shapes=True)
    model.compile(loss = rmse, optimizer = 'Adam', metrics = ['mae', rmse])
    return model


'''
Compile model and fit it to the data
'''
def compile_and_fit(df, model, config, method):
    timesteps, features, h_neurons, activations, epochs, batch_size = config
    #univariate = 1
    # compile the model
    model.compile(loss=rmse, optimizer=tf.keras.optimizers.Adam(), metrics=['mae', rmse])
   if method == 'univariate':
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_train(df, config)
        history = model.fit(X_train, y_train, 
                            validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, shuffle=False) 
        metrics = model.evaluate(X_test, y_test)
    elif method == 'multivariate':
         X_train,y_train,X_val, y_val,X_test, y_test = prepare_train(df, config)
         y_train = y_train.reshape((y_train.shape[0], y_train.shape[1],1))
         history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
         epochs=epochs, batch_size=batch_size, shuffle=False) 
         metrics = model.evaluate(X_test, y_test)
    hist = history
    loss = metrics[0]
    loss_mae = metrics[1]
    loss_rmse = metrics[2]
    plot_learning_curves(history.history['loss'], history.history['val_loss']) 
    return hist, loss, loss_mae, loss_rmse

'''
Generate a list with all the possible configurations with the parameters timestep, features, h_neurons, activations epochs and batch_size
'''
def generate_configs(timestep, features, h_neurons, activations, epochs, batch_size):
    # create configs
    configs = list()
    for i in timestep:
        for y in features:
            for j in h_neurons:
                for s in activations:
                    for k in epochs:
                        for l in batch_size:
                            config = [i, y, j, s, k, l]
                            configs.append(config)
    print('Total number of defined configurations', len(configs))
    return configs

'''
Check the training performances from a model for a configuration
'''

'''
def call_models(df, model_name, config, n_repeats=1):
    timesteps, h_neurons, epochs, batch_size = config
    univariate = 1 # Quando fazer modelo multivariate mudar
    to_supervised(df, timesteps)
    if model_name == 'lstm':
        model = build_lstm(config)
    else:
        model = build_gru(config)
    hist, loss, loss_mae, loss_rmse = compile_and_fit(df, model, config)
    print('Configuration: timestep=%s, h_neurons=%s, epochs=%s, batch_size=%s' % (config[0], config[1], config[2], config[3]))
    print('loss:', loss)
    print('mae:', loss_mae)
    print('rmse:', loss_rmse)
    return (str(config), loss_mae, loss_rmse)
    '''
def call_models(df, model_name, config, method, n_repeats=1):
    timesteps = config[0]
   # timestepp, filters, kernel_size, pool_size = config_cnn 
    to_supervised(df, timesteps)
    if model_name == 'lstm':
        model = build_lstm(config)
        hist, loss, loss_mae, loss_rmse = compile_and_fit(df, model, config, method)
        print(f'Configuration : timestep=%s, h_neurons=%s, activation=%s, epochs=%s, batch_size=%s\nloss: %s\nmae: %s\nrmse: %s' % (config[0], config[2], config[3], config[4], config[5], loss, loss_mae, loss_rmse))
    elif model_name == 'lstm_auto':
        model = build_lstm_autoencoder(df,config)
        hist, loss, loss_mae, loss_rmse = compile_and_fit(df, model, config, method)
        print(f'Configuration : timestep=%s, h_neurons=%s, activation=%s, epochs=%s, batch_size=%s\nloss: %s\nmae: %s\nrmse: %s' % (config[0], config[2], config[3], config[4], config[5], loss, loss_mae, loss_rmse))    
    elif model_name == 'gru_auto':
        model = build_gru_autoencoder(df,config)
        hist, loss, loss_mae, loss_rmse = compile_and_fit(df, model, config, method)
        print(f'Configuration : timestep=%s, h_neurons=%s, activation=%s, epochs=%s, batch_size=%s\nloss: %s\nmae: %s\nrmse: %s' % (config[0], config[2], config[3], config[4], config[5], loss, loss_mae, loss_rmse))    
    elif model_name == 'gru':
        model = build_gru(config)
        hist, loss, loss_mae, loss_rmse = compile_and_fit(df, model, config, method)
        print('Configuration: timestep=%s, h_neurons=%s, activation=%s, epochs=%s, batch_size=%s\nloss: %s\nmae: %s\nrmse: %s' % (config[0], config[2], config[3], config[4], config[5], loss, loss_mae, loss_rmse))
    elif model_name == 'srnn':
        model = build_Simple_Rnn(config) 
        hist, loss, loss_mae, loss_rmse = compile_and_fit(df, model, config, method)
        print('Configuration: timestep=%s, h_neurons=%s, activation=%s, epochs=%s, batch_size=%s\nloss: %s\nmae: %s\nrmse: %s' % (config[0], config[2], config[3], config[4], config[5], loss, loss_mae, loss_rmse))
   # elif model_name == 'cnn':
        #model = build_cnn(config_cnn)
        #hist, loss, loss_mae, loss_rmse = compile_and_fit(df, model, config_cnn)
        #print('Configuration: timestep=%s, filters=%s, kernel_size=%s, pool_size=%s\nloss: %s\nmae: %s\nrmse: %s' % (config_cnn[0], config_cnn[1], config_cnn[2], config_cnn[3], loss, loss_mae, loss_rmse))
    return (str(config), loss, loss_mae, loss_rmse)

'''
Train a model for each configuration and sort the configurations by performance
'''
def grid_search(df, model_name, configs, method):
    # evaluate configs
    scores = []
    i =0
    for config in configs:
        #print(i)
       # for i in range (len(configs)):
        print(f'-------------------------------------------------------------------------------------')
        print(f'----------------Configuracao {i} A ser Executada-------------------------------------')
        print(f'-------------------------------------------------------------------------------------')
        scores.append(call_models(df, model_name,config,method))
            # sort configs by error in ascending order
        scores.sort(key=lambda tup: tup[1])
        i = i+1
    return scores

'''
Recursive multi-step forecast
'''
def forecast(model, df, timesteps, multisteps, scaler):
    input_seq = df[-timesteps:].values # get the last known sequence
    inp = input_seq
    forecasts = list()
    for step in range(1, multisteps+1):
        print(inp.shape)
        prediction = model.predict(inp)
        forecasts.append(prediction)
        inp = np.append(inp, [[prediction]], axis=0) # insert prediction to the sequence
        np.delete(inp, 0)                            # remove oldest value of the sequence
    return forecasts

'''
Plot learning curves
'''
def plot_learning_curves(loss, val_loss):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

'''
Plot forecast
'''
def plot_forecast(data, forecasts):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(data)), data, color='green', label='Confirmed')
    plt.plot(range(len(data)-1, len(data)+len(forecasts)-1), forecasts, color='red', label='Forecasts')
    plt.title('Number of Incidents')
    plt.ylabel('Incidents')
    plt.xlabel('Days')
    plt.legend()
    plt.show()
