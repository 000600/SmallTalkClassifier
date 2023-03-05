# Small Talk Classifier

## The Neural Network
This neural network classifies the kind of small talk of a short phrase, greeting, or saying. The model will predict a list of 84 elements (indices zero through 83), where each value in the list represents the probability that the inputted text is a representation of the small-talk classification associated with that index. In other words, given a text input, the model outputs a list [*probability_index_is_zero*, *probability_index_is_one*, *probability_index_is_two*, ... *probability_index_is_eighty-two*, *probability_index_is_eighty-three*]. The element with the highest probability means that the index of that element (a kind of small talk mapped zero through 83) is the model's prediction. Since the model is a multi-label classifier (it classifies which kind of small talk a text classifies as), it uses a categorical crossentropy loss function and has 83 output neurons (one for each class). The model uses a standard RMSprop optimizer and uses early stopping to prevent overfitting. The model has an architecture consisting of:
- 1 Embedding layer (with an input size of 88584 and an output size of 32) 
- 1 LSTM layer (with 32 input neurons)
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/salmanfaroz/small-talk-intent-classification-data. Credit for the dataset collection goes to **Onurdyar**  and others on *Kaggle*. It describes what type of small talk a piece of text is. To view the various kinds of small talk classified by the dataset, look at the dataset attached at the link or in the repository.

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
