import numpy as np

# - - - - - - - - - - - -
# Multi-Layer Perceptron
# - - - - - - - - - - - -
class MLP:

    '''
    Constructor
    '''
    def __init__(self, epochs = 1, batch_size = 32, output_neurons = 2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_neurons = output_neurons

    '''
    Prepare data
    '''
    def prepare_data(self, training_set):
        # reshape 
        x = np.array([i[1] for i in trainig_set], dtype = 'float32').reshape(-1, len(training_set[0][1]))
        # get labels (actions)
        y = np.array([i[0] for i in trainig_set])
        return x, y

    '''
    Build model
    '''
    def build(self):
        pass

    '''
    Fit the training data to the model
    '''
    def fit(self, trainig_set):
        pass

    '''
    Predict action from observation
    '''
    def predict(self, obs):
        pass
    

