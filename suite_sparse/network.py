from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def create_model(n_rows, n_statistics, n_params):
    '''
    Creates a neural network model designed to map a matrix of n_rows sampled rows
    by n_statistics statistics applied to those rows, to a vector of n_params
    MueLu params. All data is normalized.

    The network topology also performs dimension reduction (PLS) by including
    a layer of size 3
    '''

    # Define model topology
    
    # Input Layer
    visible = Input(shape=(n_rows, n_statistics, 1))
    reshape = Flatten()(visible)
    
    # Convolutional layers to reduce dimensionality
    # Dense layers
    dense1 = Dense(1024, activation='relu')(reshape)
    dense1a = Dense(512, activation='relu')(dense1)
    dense1b = Dense(256, activation='relu')(dense1a)
    dense2 = Dense(128, activation='relu')(dense1b)
    dense3 = Dense(64, activation='relu')(dense2)
    dense4 = Dense(32, activation='relu')(dense3)
    dense5 = Dense(16, activation='relu')(dense4)
    dense6 = Dense(8, activation='relu')(dense5)
    output = Dense(n_params)(dense6)

    # Compute model
    model = Model(inputs=visible, outputs=output)

    return model

def train_model(model, x, y, train_split=0.8):
    '''
    Trains a model by fitting x as input vs y as output. Note that this
    function automatically splits x and y into train and test sets. "split"
    denotes the proportion of the data to be used as the training set. The
    function will return the trained model and the history dict acquired from
    training.
    '''
    
    # Split the data into train and test
    split_idx = int(train_split * len(x))
    x_train = np.stack(x[:split_idx], axis=0)
    y_train = np.stack(y[:split_idx], axis=0)
    x_test  = np.stack(x[split_idx:], axis=0)
    y_test  = np.stack(y[split_idx:], axis=0)

    # Compile the model
    print('Compiling model...')
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # Train the model using our training and testing data
    print('Training model...')
    history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_split=0.1)
    
    return model, history

def save_model(model, path, fmt='hdf5'):
    '''
    Saves a model in specified format to the passed path
    '''
    if fmt == 'hdf5':
        model.save(path)

def load_data(encodings_dir, labels_dir, n_permutations=1):
    '''
    Loads matrix encodings and labels from the given dirs 
    into lists of numpy ndarrays. n_permutations refers to the number of times
    that we shuffle the rows of the read encoding to generate more data
    '''

    # List of ndarrays we will return
    encodings = list()
    labels    = list()

    # paths to encoding and label files
    encoding_paths = list()
    label_paths    = list()

    # Get all csv files in the encodings directory
    for root, dirs, files in os.walk(encodings_dir):
        for _fname in files:
            if _fname.endswith('.csv'):
                encoding_paths.append(os.path.join(root, _fname))

    # Get all csv files in the labels directory
    for root, dirs, files in os.walk(labels_dir):
        for _fname in files:
            if _fname.endswith('.csv'):
                label_paths.append(os.path.join(root, _fname))
    
    # Loop over each label, and load all encodings with the same matrix name as
    # the label to the encodings list
    for label_path in label_paths:
        
        # Get the matrix name from the label path
        mtx_name = label_path.split('/')[2].split('.')[0]

        # Get all encoding paths with the same matrix name in them
        mtx_encoding_paths = [p for p in encoding_paths if mtx_name in p]

        # Load (encoding, label) pairs to lists
        for mtx_encoding_path in mtx_encoding_paths:

            for _ in range(n_permutations):
                try:
                    # Read encodings and labels
                    mtx = np.genfromtxt(mtx_encoding_path, dtype=np.float32, delimiter=',', skip_header=1, usecols=(1,2))
                    lab = np.genfromtxt(label_path,        dtype=np.float32, delimiter=',', skip_header=1)

                    # Normalize encodings
                    mtx -= mtx.min(axis=0)
                    denom = mtx.max(axis=0) - mtx.min(axis=0)
                    for d in denom:
                        if abs(d) < 1e-8:
                            #print(f'Encountered zero denominator in file: {mtx_encoding_path}')
                            continue
                    mtx /= denom
                    
                    # Randomly shuffle array to gen more data
                    np.random.shuffle(mtx)

                    mtx.shape += 1,
                    
                    # lab[0] = math.log(lab[0] + 10) / 8.5
                    # lab[1] = (lab[1] - 0.2) / 1.6
                    # lab[2] = (lab[2] - 0.2) / 1.6 
                    # lab[3] = (lab[3] - 500) / 9500
                    # lab[4] = (lab[4] - 1) / 99

                    encodings.append(mtx)
                    labels.append(lab)
                except Exception as e:
                    print(e)
                    print(label_path)

    print(len(encodings))

    # Check to make sure we have the same number of data
    assert(len(encodings) == len(labels))

    # Return the constructed lists
    return encodings, labels

def main():

    # Data directories
    encodings_dir = 'data/encodings_pca'
    labels_dir    = 'data/labels'

    # Load the encondings and labels
    x, y = load_data(encodings_dir, labels_dir, n_permutations=10)
   
    # Data parameters
    n_rows       = 1601 
    n_statistics = 2
    n_params     = 5

    # Create the model
    model = create_model(n_rows, n_statistics, n_params)
    
    # Train the model
    model, history = train_model(model, x, y, train_split=0.95) 

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Save the model
    save_model(model, 'data/models/model_1_0.h5')

def get_params(mtx_encoding_path):
    
    mtx = np.genfromtxt(mtx_encoding_path, dtype=np.float32, delimiter=',', skip_header=1, usecols=(1,2))

    # Normalize encodings
    mtx -= mtx.min(axis=0)
    denom = mtx.max(axis=0) - mtx.min(axis=0)
    for d in denom:
        if abs(d) < 1e-8:
            #print(f'Encountered zero denominator in file: {mtx_encoding_path}')
            continue
    mtx /= denom
    mtx.shape += 1,
    mtx = np.stack([mtx])
    
    # Load the pre trained model
    model = load_model('data/models/model_1_0.h5')

    # Compute the optimal params for the given mtx encoding
    params = model.predict(mtx)[0]
    
    # Denormalize
    drop_tol    = math.exp(params[0] * 8.0 - 10.0)
    rel_damp    = params[1] * 1.6 + 0.2
    sa_damp     = params[2] * 1.6 + 0.2
    coarse_size = (int) (params[3] * 9500 + 500)
    rel_sweeps  = (int) (params[4] * 99 + 1)

    # Path we are going to write to
    mtx_name = mtx_encoding_path.split('/')[-1].split('.')[0]
    param_file_path = f'data/muelu_params/{mtx_name}.xml'

    with open(param_file_path, 'w') as xml_file:

        # Write header file
        xml_file.write('<ParameterList name=\"MueLu\">\n')

        xml_file.write('\t<Parameter name=\"multigrid algorithm\" type=\"string\" value=\"sa\"/>\n')
        xml_file.write('\t<Parameter name=\"verbosity\" type=\"string\" value=\"low\"/>\n')
        xml_file.write('\t<Parameter name=\"max levels\" type=\"int\" value=\"20\"/>\n')

        xml_file.write(f'\t<Parameter name=\"coarse: max size\" type=\"int\" value=\"{coarse_size}\"/>\n')
        xml_file.write(f'\t<Parameter name=\"aggregation: drop tol\" type=\"double\" value=\"{drop_tol}\"/>\n')
        xml_file.write(f'\t<Parameter name=\"sa: damping factor\" type=\"double\" value=\"{sa_damp}\"/>\n')

        xml_file.write('<!-- Symmetric Gauss-Seidel smoothing -->\n')
        xml_file.write('<Parameter name=\"smoother: type\" type=\"string\" value=\"RELAXATION\"/>\n')
        xml_file.write('\t<ParameterList name=\"smoother: params\">\n')
        xml_file.write('\t<Parameter name=\"relaxation: type\" type=\"string\" value=\"Symmetric Gauss-Seidel\"/>\n')
        xml_file.write(f'\t<Parameter name=\"relaxation: sweeps\" type=\"int\" value=\"{rel_sweeps}\"/>\n')
        xml_file.write(f'\t<Parameter name=\"relaxation: damping factor\" type=\"double\" value=\"{rel_damp}\"/>\n')
        xml_file.write('\t</ParameterList>\n')
        xml_file.write('</ParameterList>\n')

if __name__ == '__main__':
    #main()
    mtx_encoding_path = sys.argv[1]
    get_params(mtx_encoding_path)
