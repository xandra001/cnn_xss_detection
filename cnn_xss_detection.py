'''
File: cnn_xss_detection.py
Author: Alexandra Dragan
GitHub: xandra001
Date: 2024-04-20
Copyright: (c) 2024 - Alexandra Dragan

Description: This file contains the code for the CNN model for XSS detection.
'''

# Importing necessary libraries
import numpy as np
import pandas as pd
import keras as kr
import sklearn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import joblib

# Loading the XSS dataset downloaded from Kaggle (https://www.kaggle.com/datasets/syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning)
df_start = pd.read_csv("XSS_dataset.csv")

# Visualizing the first few rows of the dataset
df_start.head()

# Selecting only the 'Sentence' and 'Label' columns
df_start = pd.read_csv("XSS_dataset.csv", usecols=['Sentence', 'Label'])

df_start.head()

# Encode labels
label_encoder = LabelEncoder()
df_start['Label'] = label_encoder.fit_transform(df_start['Label'])

# Calcolo il numero di istanze benigne (Label = 0) e di istanze maligne (Label = 1)
print(f'Benign instances (0): {np.sum(df_start["Label"] == 0)}')
print(f'Malign instances (1): {np.sum(df_start["Label"] == 1)}')
print(f'Total istances: {len(df_start)}')

# Removing rows containing null values
df_start.dropna(inplace=True)

# Counting unique and duplicate values
val_unique = len(pd.unique(df_start['Sentence']))
print(f'Unique values in the dataset XSS_dataset.csv: {val_unique}')
val_duplicate = len(df_start) - val_unique
print(f'Total of duplicate values to be removed from the dataset: {val_duplicate}')
# Removing duplicates
df = pd.DataFrame.drop_duplicates(df_start)

# Showing the number of benign and malignant instances after cleaning the dataset
print(f'Benign instances (0): {np.sum(df["Label"] == 0)}')
print(f'Malign instances (1): {np.sum(df["Label"] == 1)}')

# Creating a numpy array with the values in the ‘Sentence’ column
s = df['Sentence'].values

# Initialising TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(use_idf=True)

# Vectoring ‘Sentence’ using TF-IDF vectorisation
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(s)

# Creating a new data frame with the vectors resulting from the TF-IDF vectorisation
X = pd.DataFrame(tfidf_vectorizer_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Creating a numpy array with the values in the ‘Label’ column
y = kr.utils.to_categorical(df['Label'].values)

# Adding a bias that is used to shift the result of the activation function
X = np.concatenate((np.ones((len(df), 1)), X.values), axis=1)

# Saving the tfidf vectoriser
joblib.dump(tfidf_vectorizer, 'vectorizer.pkl')

# Splitting the dataset into train set (70%) and test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)

# Defining the neural network using keras
modello = Sequential()
# A 1-dimensional convolutional layer with 32 output channels, a kernel of 5 and a ReLU activation function.
modello.add(Conv1D(32, 5, activation='relu', input_shape=(X.shape[1], 1)))
# A dropout layer with dropout rate 0.4 (40% of input units), which allows some neurons to be switched off during training in order to avoid overfitting
modello.add(Dropout(0.4))
# A Max Pooling layer that reduces the spatial dimensions of the input by selecting the maximum value from each dimension patch (1, 4) for each feature map.
modello.add(MaxPooling1D(pool_size=4))
# A 1-dimensional convolutional layer with 16 output channels, a kernel of 3 and a ReLU activation function.
modello.add(Conv1D(16, 3, activation='relu'))
# Another dropout layer
modello.add(Dropout(0.4))
# This layer flattens the output of the previous layer into a 1-dimensional array.
modello.add(Flatten())
# This layer defines a fully connected layer with 2 units (binary classification) and a softmax activation function.
modello.add(Dense(2, activation='softmax'))

# Compiling the model using the Adam optimiser, the binary cross entropy loss function, which is appropriate for classification problems with 2 output classes and the metric that is used to evaluate performance.
modello.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Showing the model summary
modello.summary()

# Training the model in which 10% of the training data will be used for validation. The model is trained 5 times on the training set, on 128 samples at a time
history = modello.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=128)

# Evaluating the performance of the model
test_loss, test_accuracy = modello.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
print(f'Test loss: {test_loss}')

# Visualising the accuracy of the model in each epoch for both the training set and the validation set
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Visualising the model loss in each epoch for both the training set and the validation set
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Test set predictions
y_pred = modello.predict(X_test)

# Converting predictions into class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Converting real labels to class labels
y_test_labels = np.argmax(y_test, axis=1)

# Calculating the confusion matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

# Calculating accuracy, recall and f-measure
class_report = classification_report(y_test_labels, y_pred_labels)

# Printing the report
print(class_report)


# Representing the confusion matrix graphically
plt.figure(figsize=(10, 10))
plt.imshow(conf_matrix)
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(range(2))
plt.yticks(range(2))
plt.colorbar()

# Adding in each cell of the array the number of ranked instances with a number colour that is in good contrast to the background
for i in range(2):
    for j in range(2):
      if(i==j):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="black",
                 size=14)
      else: plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="white",
                 size=14)



plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()


# Saving the model 
modello.save("modelCNN.h5")