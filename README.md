# Small Talk Classifier

## The Neural Network
This neural network classifies the kind of small talk of a short phrase, greeting, or saying. The model outputs a list of 84 elements (indices zero through 83), where each value in the list represents the probability that the inputted text is a representation of the small-talk classification associated with that index. In other words, given a text input, the model outputs a list [*probability_index_is_zero*, *probability_index_is_one*, *probability_index_is_two*, ... *probability_index_is_eighty-two*, *probability_index_is_eighty-three*]. Each index corresponds to a type of small talk in the dataset, such as "acquaintance" or "annoying," and the index with the highest probability is the model's predicted classification. Since the model is a multi-label classifier (it classifies which kind of small talk a text classifies as), it uses a categorical crossentropy loss function and has 84 output neurons (one for each class). The model uses a standard RMSprop optimizer and uses early stopping to prevent overfitting. The model has an architecture consisting of:
- 1 Embedding layer (with an input size of 1000 and an output size of 32) 
- 1 LSTM layer (with 32 input neurons)
- 1 Output layer (with 84 output neurons and a softmax activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/salmanfaroz/small-talk-intent-classification-data. Credit for the dataset collection goes to **Onurdyar** and others on *Kaggle*. It describes what type of small talk a piece of text is. To view the various kinds of small talk classified by the dataset, look at the dataset attached at the link or in the repository.

## Libraries
This neural network was created with the help of the Tensorflow and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
