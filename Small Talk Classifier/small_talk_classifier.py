# Imports
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('SmallTalkData.csv')
df = pd.DataFrame(df)

# Initialize hyperparameters
vocab_size = 1000 # Number of words the model recognizes
max_len = 30 # Number of characters included in the input sequence
batch_size = 64
dimensions = 32

# Add specific parts of the dataset to x and y lists
x = []
y = []
for row in range(df.shape[0]):
  rows = []
  for point in range(len(df.loc[0]) - 1): # "- 1" because we don't want to add the last column (the labels) to the inputs section
    rows.append(df.iloc[row][point])
  x.append(rows)
  y.append(df.loc[row][-1])

# Categorize x-values
string = []
for phrase in x:
  for word in phrase[0].split():
    string.append(word) # Remove all words from their phrases

unique_words = list(dict.fromkeys(string)) # Put all unique words into a single list

x_categorized = []
for phrase in x:
  input = []
  for word in phrase[0].split():
    input.append(unique_words.index(word))
  x_categorized.append(input)

x_categorized = np.array(x_categorized)

# Categorize y-values
unique_classes = list(dict.fromkeys(y))
y_categorized = []
for val in y:
  categorized = [0 for i in unique_classes]
  categorized[unique_classes.index(val)] = 1
  y_categorized.append(categorized)
y_categorized = np.array(y_categorized)

output_shape = len(unique_classes) # Initialize output shape

# Split the data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(x_categorized, y_categorized, random_state = 1)

# Pad sequences to max_len with 0's to standardize input shape for the neural network
x_train = np.array(sequence.pad_sequences(x_train, max_len))
x_test = np.array(sequence.pad_sequences(x_test, max_len))

# Create model
model = Sequential()

# Input layer
model.add(Embedding(vocab_size, dimensions))

# Hidden layer
model.add(LSTM(dimensions))

# Output layer
model.add(Dense(output_shape, activation = 'softmax')) # Softmax activation function because there are multiple classes

# Compile model
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Train model and store training history
epochs = 100
history = model.fit(x_train, y_train, epochs = epochs, validation_data = (x_test, y_test), callbacks = [early_stopping])

# Visualize loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plt.plot(loss, label = 'Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['categorical_accuracy']
val_accuracy = history_dict['val_categorical_accuracy']

plt.plot(accuracy, label = 'Training Accuracy')
plt.plot(val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0) # Change verbose to 1 or 2 for more information
print(f'\nTest accuracy: {test_acc * 100}%')

# Make predictions based on inputs
word_index = {i : unique_words.index(i) for i in unique_words} # Dictionary that maps words to indexes
input_index = {v : k for (k, v) in word_index.items()} # Dictionary that maps indexes to words

# Function to encode text inputs
def encode_text(text):
  token = tf.keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in token]
  return sequence.pad_sequences([tokens], max_len)[0]

# Function to decode inputs into text
def decode_text(input):
  pad = 0
  text = ""
  for val in input:
    if val != pad:
      text += input_index[val] + " "
  return text[:-1]

# Function to determine the model's certainty in predictions
def find_certainty(prediction):
  pred = np.argmax(prediction)
  certainty = prediction[pred] * 100
  return certainty, pred

# Function to make predictions
def predict(text):
  encoded = encode_text(text)
  prediction_input = np.zeros((1, max_len)) # (1, 30) template because that is the length of inputs 
  prediction_input[0] = encoded # Insert input into template of zeros

  # Get prediction
  prediction = model.predict(prediction_input, verbose = 0)[0]
  certainty, pred_class = find_certainty(prediction)

  output_text = f"Model's Prediction ({certainty}% certainty): {pred_class} ({unique_classes[pred_class]})"
  return output_text

# Prediction vs. actual value (change the index to view a different input and output set in the test data)
index = 10
prediction = model.predict(x_test, verbose = 0)[index]
cert, predicted_class = find_certainty(prediction) # Get certainty and class predictions from the model
actual_class = np.argmax(y_test[index]) # Get the actual class
text_input = decode_text(x_test[index])

print("\nTest Data Sample:")
print("   - Input: " + text_input)
print(f"   - Model's Prediction ({cert}% certainty): {predicted_class} ({unique_classes[predicted_class]}) | Actual Class: {actual_class} ({unique_classes[actual_class]})")


# Sample input data predictions
sample_text = "How are you?"
print("\nUser Input Sample:")
print("   - Input: " + sample_text)
print("   - " + predict(sample_text))
