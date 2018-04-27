# coding: utf-8
from keras.models import Model
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.utils import plot_model

import time
import os

def default_callbacks(model,
                      prefix='',
                      batch_size=32,
                      logdir='../logs'):
    """
    Function returns a list of default callbacks for use with training runs.
    The method also draws the model structure as a side effect
    """
    now = time.strftime("%c")
    run_name = os.path.join(prefix, now)

    logpath = create_run_directory(logdir, run_name)
    draw_model_plots(model, logdir, run_name)    # needs graphviz and pydot
    write_model_graph(model, logdir, run_name)
    tb = default_tensor_board(run_name, batch_size, logdir)
    csv_logger = default_csv_logger(run_name, logdir)
    checkpoint = default_model_checkpoint(run_name, logdir)

    return logpath, [tb, csv_logger, checkpoint]

def create_run_directory(logdir='../logs',
                         run_name=''):
    """
    Function creates the path to the log directory
    """
    logpath = os.path.join(logdir, run_name)
    os.makedirs(logpath)
    return logpath

def write_model_graph(model,
                      logdir='../logs',
                      run_name=''):
    """
    Function writes the model graph in json format in the run path
    """
    logpath = os.path.join(logdir, run_name)
    model_json = model.to_json()
    with open(os.path.join(logpath, "model_graph.json"), "w") as f:
        f.write(model_json)

def default_model_checkpoint(run_name='',
                             logdir='../logs'):
    """
    Function returns a default checkpoint callback to save the model on every run
    """
    filename = os.path.join(logdir, run_name, 'model-{epoch:02d}-{val_acc:.4f}.hdf5')
    return ModelCheckpoint(filename, monitor='val_loss', period=50)

def default_tensor_board(run_name='',
                         batch_size=32,
                         logdir='../logs'):
    """
    Function returns a tensorboard callback initialized with the defaults
    """

    tensorboard = TensorBoard(log_dir=os.path.join(logdir, run_name),
                              #histogram_freq=0,
                              batch_size=batch_size,
                              write_graph=True,
                              #write_grads=True,
                              #write_images=True,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)
    return tensorboard

def default_csv_logger(run_name='',
                       logdir='../logs'):
    """
    Function returns a csv logger that saves results of every epoch in csv format
    """
    filename = os.path.join(logdir, run_name, 'training.log')
    return CSVLogger(filename, separator=',', append=False)

def draw_model_plots(model,
                     logdir='../logs',
                     run_name=''):
    """
    Function draws the model structure into a png file
    """
    directory = os.path.join(logdir, run_name)
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, 'model.png')
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
