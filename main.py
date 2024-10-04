import numpy as np
import matplotlib.pyplot as plt
import time
import keras
import os
import csv
import random
import itertools
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.datasets import mnist
from scikeras.wrappers import KerasClassifier

import tensorflow as tf

#----- Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#----- Helper functions
def prep_MNIST():
     #----- Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Training set dimensions: {x_train.shape}, Labels: {y_train.shape}")
    print(f"Test set dimensions    : {x_test.shape}, Labels: {y_test.shape}")
    #----- Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

def plot_history(history, test_loss, test_accuracy, file_name, model_name):
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Plot train and val loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.axhline(y=test_loss, color='red', linestyle='--', label='Final Test Loss')
    plt.title(f'Loss Over Epochs: NN Trained with {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot train and val accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.axhline(y=test_accuracy, color='red', linestyle='--', label='Final Test Accuracy')
    plt.title(f'Accuracy Over Epochs: NN Trained with {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save plot
    plt.savefig(f"{file_name}.png")
    
    plt.tight_layout()
    # plt.show()
    
def store_stats(history, func, epoch, lr, test_loss, test_acc, training_time):
    with open(f'{func}_stats.csv', mode='a', newline='') as file:
    # with open('RS_stats.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header only if the file is new or empty
        if file.tell() == 0:
            writer.writerow(['Function', 
                            'Epoch',
                            'Learning Rate',
                            'Training Time',
                            'Training Loss', 
                            'Training Accuracy', 
                            'Validation Loss', 
                            'Validation Accuracy',
                            'Test Loss',
                            'Test Accuracy'])
        for i in range(epoch):
            writer.writerow([func, 
                             i + 1, #epoch
                             lr,
                             training_time,
                             history.history['loss'][i], 
                             history.history['accuracy'][i], 
                             history.history['val_loss'][i], 
                             history.history['val_accuracy'][i],
                             test_loss,
                             test_acc])

#----- PART 1
#----- 1.1
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
    store_stats(history, func='FC_SGD', 
                epoch=EPOCH, lr=LEARNING_RATE, 
                test_loss=test_loss, test_acc=test_acc, 
                training_time=tot_time)

    print("\nFinal Test Loss    : {:.4%}".format(test_loss))
    print("Final Test Accuracy: {:.4%}".format(test_acc))
    print("Total Training Time: {:.2f} seconds".format(tot_time))

    plot_history(history, test_loss, test_acc, 'FC_SGD', 'SGD')

#----- 1.2 Add Momentum
def train_fully_connected_momentum_sgd(func, learning_rate, momentum, batch_size):
    x_train, y_train, x_test, y_test = prep_MNIST()

    #-----Model
    model = Sequential([
        Flatten(input_shape=(28, 28)), 
        Dense(1024, activation='relu'),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ]) 

    # PART 1.2---------------------------------------------------
    EPOCH = 10
    VAL_SPLIT = 0.1
    
    model.compile(optimizer=SGD(learning_rate=learning_rate, 
                                momentum=momentum),
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    
    #----- Train
    start_time = time.time()
    history = model.fit(x_train, y_train,
                        epochs=EPOCH, 
                        batch_size=batch_size,
                        validation_split=VAL_SPLIT)
    
    #----- Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    # Calculate wall-clock time
    end_time = time.time()
    tot_time = end_time - start_time
    
    # Print training, validation and test statistics
    store_stats(history, func=func, 
                epoch=EPOCH, lr=0.1, 
                test_loss=test_loss, test_acc=test_acc, 
                training_time=tot_time)

    print("\nFinal Test Loss    : {:.4%}".format(test_loss))
    print("Final Test Accuracy: {:.4%}".format(test_acc))
    print("Total Training Time: {:.2f} seconds".format(tot_time))

    plot_history(history, test_loss, test_acc, 'SGD_Momentum','Momentum SGD')

    return history.history, test_loss, test_acc, tot_time

#----- 1.3 Adam
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
    store_stats(history, func='Adam', 
                epoch=EPOCH, lr=LEARNING_RATE, 
                test_loss=test_loss, test_acc=test_acc, 
                training_time=tot_time)

    print("\nFinal Test Loss    : {:.4%}".format(test_loss))
    print("Final Test Accuracy: {:.4%}".format(test_acc))
    print("Total Training Time: {:.2f} seconds".format(tot_time))

    plot_history(history, test_loss, test_acc, 'Adam','Adam')

#----- 1.4 Add Batch Normalization
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
    store_stats(history, func='BN_SGD', 
                epoch=EPOCH, lr=0.1, 
                test_loss=test_loss, test_acc=test_acc, 
                training_time=tot_time)

    print("\nFinal Test Loss    : {:.4%}".format(test_loss))
    print("Final Test Accuracy: {:.4%}".format(test_acc))
    print("Total Training Time: {:.2f} seconds".format(tot_time))

    plot_history(history, test_loss, test_acc, 'BN_SGD', 'SGD and Batch Normalization')

#----- PART 2
#----- 2.1
def train_momentum_sgd_gs_lr():
    x_train, y_train, x_test, y_test = prep_MNIST()

    learning_rates = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    
    SGD_MOMENTUM = 0.9
    EPOCH = 10
    BATCH_SIZE = 128
    VAL_SPLIT = 0.1

    best_val_acc = -np.inf
    best_lr = None 
    best_history = None  
    best_test_loss = None
    best_test_acc = None
    best_training_time = None

    for lr in learning_rates:
        print(f"\n-------------------------------")
        print(f"Training with learning rate: {lr}")

        #-----Model
        model = Sequential([
            Flatten(input_shape=(28, 28)), 
            Dense(1024, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')
        ]) 

        # PART 1.2---------------------------------------------------
        LEARNING_RATE = lr
        SGD_MOMENTUM = 0.9
        EPOCH = 10
        VAL_SPLIT = 0.1
        
        model.compile(optimizer=SGD(learning_rate=LEARNING_RATE, 
                                    momentum=SGD_MOMENTUM),
                    loss=SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])
        
        #----- Train
        start_time = time.time()
        history = model.fit(x_train, y_train,
                            epochs=EPOCH,
                            validation_split=VAL_SPLIT)
        
        #----- Evaluate
        test_loss, test_acc = model.evaluate(x_test, y_test)
        val_acc = history.history['val_accuracy'][-1] # val_acc of the last epoch

        # print(f"Validation accuracy for lr={lr}: {val_acc:.4f}")
        print(f"Test accuracy for lr={lr}: {test_acc:.4f}")

        # Calculate wall-clock time
        end_time = time.time()
        tot_time = end_time - start_time
        
        # If this learning rate gives a better validation accuracy, update best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = lr
            best_history = history
            best_test_loss = test_loss
            best_test_acc = test_acc
            best_training_time = end_time - start_time
        
        # Print training, validation and test statistics
        store_stats(history, func=f'SGD_Momentum_GS_{lr}', 
                    epoch=EPOCH, lr=LEARNING_RATE, 
                    test_loss=test_loss, test_acc=test_acc, 
                    training_time=tot_time)

        print("\nFinal Test Loss    : {:.4%}".format(test_loss))
        print("Final Test Accuracy: {:.4%}".format(test_acc))
        print("Total Training Time: {:.2f} seconds".format(tot_time))

        plot_history(history, test_loss, test_acc, f'SGD_Momentum_GS_{lr}',f'Momentum SGD LR={lr}')
    
    print(f"\nBest learning rate: {best_lr}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Total Training Time: {best_training_time:.2f} seconds")

#----- 2.2
def grid_search_sgd(learning_rates, momentums, batch_sizes):
 
    # Perform grid search
    best_params = None
    best_val_acc = -np.inf  # Track best validation accuracy
    best_result = None

    # Iterate over all combinations of hyperparameters
    for lr, mo, bs in itertools.product(learning_rates, momentums, batch_sizes):
        print(f"Training with lr={lr}, momentum={mo}, batch_size={bs}")

        # Train the model with current hyperparameter combination
        his, test_loss, test_acc, training_time = train_fully_connected_momentum_sgd(f'GridSearch+LR={lr}_M={mo}_BS={bs}',
                                                                                     lr, mo, bs)

        # Extract validation accuracy from the last epoch
        val_acc = his['val_accuracy'][-1]
        val_loss = his['val_loss'][-1]

        print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

        # Update best parameters if current model is better
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (lr, mo, bs)
            best_result = (val_acc, val_loss, test_acc, test_loss, training_time)

    # Report the best parameters and results
    print(f"\nBest Parameters: Learning Rate={best_params[0]}, Momentum={best_params[1]}, Batch Size={best_params[2]}")
    print(f"Best Validation Accuracy: {best_result[0]:.4f}, Validation Loss: {best_result[1]:.4f}")
    print(f"Best Test Accuracy: {best_result[2]:.4f}, Test Loss: {best_result[3]:.4f}")
    print(f"Total Training Time: {best_result[4]:.2f} seconds")

#----- 2.3
def random_search_sgd(n_rand_comb):
    def sample_hyperparameters():
        learning_rate = np.random.uniform(0.001, 0.1)   # Log-uniform
        momentum = np.random.uniform(0.0, 0.9)          # Uniform distribution
        batch_size = random.choice([32, 64, 128])       # Discrete uniform
        return learning_rate, momentum, batch_size

    best_params = None
    best_val_acc = -np.inf  
    best_result = None

    for _ in range(n_rand_comb):
        lr, mo, bs = sample_hyperparameters()
        print(f"\nTraining with lr={lr}, momentum={mo}, batch_size={bs}")

        his, test_loss, test_acc, training_time = train_fully_connected_momentum_sgd(f'RS+LR={lr}_M={mo}_BS={bs}',
                                                                                         lr, mo, bs)
        
        val_acc = his['val_accuracy'][-1]
        val_loss = his['val_loss'][-1]

        print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (lr, mo, bs)
            best_result = (val_acc, val_loss, test_acc, test_loss, training_time)
 
    print(f"\nBest Parameters: Learning Rate={best_params[0]}, Momentum={best_params[1]}, Batch Size={best_params[2]}")
    print(f"Best Validation Accuracy: {best_result[0]:.4f}, Validation Loss: {best_result[1]:.4f}")
    print(f"Best Test Accuracy: {best_result[2]:.4f}, Test Loss: {best_result[3]:.4f}")
    print(f"Total Training Time: {best_result[4]:.2f} seconds")

#----- PART 3
def train_CNN_sgd():
    x_train, y_train, x_test, y_test = prep_MNIST()

    model = Sequential([
        Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')  # Output layer for 10 classes (for MNIST)
    ])

    LEARNING_RATE = 0.001
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
                        validation_split=VAL_SPLIT)
    
    #----- Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    # Calculate wall-clock time
    end_time = time.time()
    tot_time = end_time - start_time
    
    # Print training, validation and test statistics
    store_stats(history, func='CNN_SGD', 
                epoch=EPOCH, lr=LEARNING_RATE, 
                test_loss=test_loss, test_acc=test_acc, 
                training_time=tot_time)

    print("\nFinal Test Loss    : {:.4%}".format(test_loss))
    print("Final Test Accuracy: {:.4%}".format(test_acc))
    print("Total Training Time: {:.2f} seconds".format(tot_time))

    plot_history(history, test_loss, test_acc, 'CNN_SGD', 'CNN SGD')

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')
    

if __name__ == "__main__":
    # PART 1 -------------------------
    train_fully_connected_sgd()
    train_fully_connected_momentum_sgd(func='SGD_Momentum', learning_rate=0.1, momentum=0.9, batch_size=128)
    train_fully_connected_adam()
    train_fully_connected_bn_sgd()
    # PART 2 -------------------------
    train_momentum_sgd_gs_lr()

    learning_rates = [0.1, 0.03, 0.01, 0.003]
    momentums = [0.0, 0.5, 0.9]
    batch_sizes = [32, 64, 128]
    grid_search_sgd(learning_rates, momentums, batch_sizes)

    random_search_sgd(n_rand_comb=10)

    # PART 3 -------------------------
    train_CNN_sgd()