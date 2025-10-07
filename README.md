This project is for CS 4375 – Coding Assignment 1. It implements two neural network architectures in PyTorch: a Feedforward Neural Network (FFNN) and a Recurrent Neural Network (RNN). Both models are trained to perform 5-class sentiment analysis on Yelp reviews, predicting ratings from 1 to 5 based on review text.
The dataset provided includes training.json, validation.json, and test.json. The RNN also uses pretrained word embeddings loaded from word_embedding.pkl, which should be extracted from the provided Data_Embedding.zip file.
To run the models, use Python 3.8 and install the required dependencies listed in requirements.txt. You can train each model by running the following commands in your terminal:
For the Feedforward Neural Network (FFNN):
python ffnn.py --hidden_dim 128 --epochs 5 --train_data ./training.json --val_data ./validation.json
python ffnn.py --hidden_dim 256 --epochs 5 --train_data ./training.json --val_data ./validation.json
For the Recurrent Neural Network (RNN):
python rnn.py --hidden_dim 128 --epochs 10 --train_data ./training.json --val_data ./validation.json
python rnn.py --hidden_dim 256 --epochs 10 --train_data ./training.json --val_data ./validation.json
Both models train and validate on the provided data, and results (training and validation accuracies) are printed after each epoch. The FFNN uses bag-of-words vectorization, while the RNN processes sequential word embeddings.
Make sure that all files, including the JSON datasets and the embedding file, are located in the same directory as the Python scripts before running
