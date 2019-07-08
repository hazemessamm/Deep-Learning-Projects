from keras.datasets import boston_housing
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data() #Loading our data

#Normalizing our data
mean = train_data.mean(axis=0) #getting the mean for every column so our mean has shape=(training.shape[1])
train_data -= mean #subtracting the mean from every column of our train_data so we can have zero mean
std = train_data.std(axis=0) #getting the standard deviation for every column
train_data /= std #dividing the standard deviation from every column in the train_data
test_data -= mean
test_data /= std

#Building the model in a function because we will use it many times in our K-folds algorithm
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k=4 #Number of folds
num_val_samples = len(train_data) // k #Dividing the training data to the number of folds we want
num_epochs = 500 #Assigning No. of epochs
all_scores = [] #List for storing the scores for each fold
all_mae_history= [] #stores all mae history

for i in range(k):
    print('Processing fold No.: ', i)
    validation_data = train_data[i*num_val_samples: (i+1)*num_val_samples] #Splitting our training data to folds
    validation_targets = train_targets[i*num_val_samples: (i+1)*num_val_samples] #Splitting our training labels to folds
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0) #Concatenating every fold we used to our training data
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0) #Concatenating every fold we used to our training labels
    model = build_model() #Calling our already compiled model
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, verbose=2, batch_size=1) #Verbose = 0 means training in silent mode, it doesn't display anything
    mae_history = history.history['mean_absolute_error']
    all_mae_history.append(mae_history)
    val_mse, val_mae = model.evaluate(validation_data, validation_targets, verbose = 0)  #Evaluating our model on the validation data, also in silent mode (verbose=0)
    all_scores.append(val_mae) #Appending the scores to 'all_scores' list

average_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

