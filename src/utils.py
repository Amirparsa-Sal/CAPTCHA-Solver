import math
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import numpy as np

def set_seeds():
    '''A function to set seeds for reproducibility.'''
    
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value= 42

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    # tf.random.set_seed(seed_value)
    # for later versions: 
    tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

def plot_loss(history, validation = False):
    '''
    A function to plot the training history loss.
    :param history: The history object returned by the fit method.
    :param validation: A boolean to indicate if the validation loss should be plotted.
    '''
    for i in range(5):
        plt.plot(history.history[f'{"val_" if validation else ""}out{i}_loss'], label = 'out{i} {"val" if validation else "train"} loss')
    plt.title(f'{"Validation" if validation else "Train"} loss')
    plt.legend()
    plt.show()

def plot_total_accuracy(history):
    '''
    A function to plot the training history accuracy in validation data.
    :param history: The history object returned by the fit method.
    '''
    result = np.array(history.history['val_out0_accuracy'])
    for i in range(1, 5):
        result = np.multiply(result,history.history[f'val_out{i}_accuracy'])
    plt.plot(result)
    plt.show()