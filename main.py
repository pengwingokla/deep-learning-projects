import numpy as np
import matplotlib.pyplot as plt
import time
import keras
import os
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from keras.datasets import mnist

import tensorflow as tf

#----- Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



def train_fully_connected_sgd():
    x_train, y_train, x_test, y_test = prep_MNIST()

    #-----Model
    model = Sequential([
        Flatten(input_shape=(28, 28)), 
        Dense(1024, activation='relu'),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ]) 
    
    # PART 1.1---------------------------------------------------
    # Define hyperparameters
    LEARNING_RATE = 0.1
    EPOCH = 10
    BACTH_SIZE = 128
    VAL_SPLIT = 0.1
    
    model.compile(optimizer=SGD(learning_rate=LEARNING_RATE),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    #----- Train
    start_time = time.time()
    history = model.fit(x_train, y_train,
                        epochs=EPOCH, 
                        batch_size=BACTH_SIZE,
                        # Automatically splits 10% of the 
                        # training data for validation
                        validation_split=VAL_SPLIT)
    
    #----- Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    # Calculate wall-clock time
    end_time = time.time()
    tot_time = end_time - start_time
    
    # Print training, validation and test statistics
    store_stats(history, 'FC_SGD', EPOCH, test_loss, test_acc, tot_time)

    print("\nFinal Test Loss    : {:.4%}".format(test_loss))
    print("Final Test Accuracy: {:.4%}".format(test_acc))
    print("Total Training Time: {:.2f} seconds".format(tot_time))

    plot_history(history, test_loss, test_acc, 'FC_SGD')

    # PART 1.2---------------------------------------------------
    # Define hyperparameters
    LEARNING_RATE = 0.1
    SGD_MOMENTUM = 0.9
    EPOCH = 10
    BACTH_SIZE = 128
    VAL_SPLIT = 0.1
    
    model.compile(optimizer=SGD(learning_rate=LEARNING_RATE, 
                                momentum=SGD_MOMENTUM),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    #----- Train
    start_time = time.time()
    history = model.fit(x_train, y_train,
                        epochs=EPOCH, 
                        batch_size=BACTH_SIZE,
                        # Automatically splits 10% of the 
                        # training data for validation
                        validation_split=VAL_SPLIT)
    
    #----- Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    # Calculate wall-clock time
    end_time = time.time()
    tot_time = end_time - start_time
    
    # Print training, validation and test statistics
    store_stats(history, 'FC_SGD_Momentum', EPOCH, test_loss, test_acc, tot_time)

    print("\nFinal Test Loss    : {:.4%}".format(test_loss))
    print("Final Test Accuracy: {:.4%}".format(test_acc))
    print("Total Training Time: {:.2f} seconds".format(tot_time))

    plot_history(history, test_loss, test_acc, 'FC_SGD_Momentum')


def prep_MNIST():
     #----- Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Training set dimensions: {x_train.shape}, Labels: {y_train.shape}")
    print(f"Test set dimensions    : {x_test.shape}, Labels: {y_test.shape}")
    #----- Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

def plot_history(history, test_loss, test_accuracy, file_name):
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Plot train and val loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.axhline(y=test_loss, color='red', linestyle='--', label='Final Test Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot train and val accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.axhline(y=test_accuracy, color='red', linestyle='--', label='Final Test Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save plot
    plt.savefig(f"{file_name}.png")
    
    plt.tight_layout()
    # plt.show()
    
def store_stats(history, func, epoch, test_loss, test_acc, training_time):
    file_exists = os.path.isfile('training_stats.csv')
    with open('training_stats.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Function', 
                            'Epoch', 
                            'Training Time',
                            'Training Loss', 
                            'Training Accuracy', 
                            'Validation Loss', 
                            'Validation Accuracy',
                            'Test Loss',
                            'Test Accuracy'])
        writer.writerow([f'{func}', {epoch},
                         {training_time}, 
                         history.history['loss'], 
                         history.history['accuracy'], 
                         history.history['val_loss'], 
                         history.history['val_accuracy'],
                         {test_loss},
                         {test_acc}])


def train_fully_connected_adam():
    x_train, y_train, x_test, y_test = prep_MNIST()

    #-----Model
    model = Sequential([
        Flatten(input_shape=(28, 28)), 
        Dense(1024, activation='relu'),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ]) 
    
    # PART 1.3---------------------------------------------------
    LEARNING_RATE = 0.001
    EPOCH = 10
    BACTH_SIZE = 128
    VAL_SPLIT = 0.1
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE, 
                                 beta_1=0.99, 
                                 beta_2=0.999),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    #----- Train
    start_time = time.time()
    history = model.fit(x_train, y_train,
                        epochs=EPOCH, 
                        batch_size=BACTH_SIZE,
                        validation_split=VAL_SPLIT)
    
    #----- Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    # Calculate wall-clock time
    end_time = time.time()
    tot_time = end_time - start_time
    
    # Print training, validation and test statistics
    store_stats(history, 'FC_Adam', EPOCH, test_loss, test_acc, tot_time)

    print("\nFinal Test Loss    : {:.4%}".format(test_loss))
    print("Final Test Accuracy: {:.4%}".format(test_acc))
    print("Total Training Time: {:.2f} seconds".format(tot_time))

    plot_history(history, test_loss, test_acc, 'FC_Adam')


def train_fully_connected_bn_sgd():
    x_train, y_train, x_test, y_test = prep_MNIST()

    #-----Model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(1024),
        BatchNormalization(),
        Activation('relu'),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dense(10),
        BatchNormalization(),
        Activation('softmax')
    ])

    LEARNING_RATE = 0.001
    SGD_MOMENTUM = 0.9
    EPOCH = 10
    BACTH_SIZE = 128
    VAL_SPLIT = 0.1
    
    model.compile(optimizer=SGD(learning_rate=LEARNING_RATE, 
                                 momentum=SGD_MOMENTUM),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    #----- Train
    start_time = time.time()
    history = model.fit(x_train, y_train,
                        epochs=EPOCH, 
                        batch_size=BACTH_SIZE,
                        validation_split=VAL_SPLIT)
    
    #----- Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    # Calculate wall-clock time
    end_time = time.time()
    tot_time = end_time - start_time
    
    # Print training, validation and test statistics
    store_stats(history, 'FC_BN_SGD', EPOCH, test_loss, test_acc, tot_time)

    print("\nFinal Test Loss    : {:.4%}".format(test_loss))
    print("Final Test Accuracy: {:.4%}".format(test_acc))
    print("Total Training Time: {:.2f} seconds".format(tot_time))

    plot_history(history, test_loss, test_acc, 'FC_BN_SGD')

if __name__ == "__main__":
    train_fully_connected_sgd()
    train_fully_connected_adam()
    train_fully_connected_bn_sgd()