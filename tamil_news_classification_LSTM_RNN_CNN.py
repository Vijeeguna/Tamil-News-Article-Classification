import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

train = pd.read_csv('tamil_news_train.csv')
test = pd.read_csv('tamil_news_test.csv')

# Dropping duplicates
train.drop_duplicates().reset_index(inplace=True, drop=True)

# Label Encoding
label_encoder = preprocessing.LabelEncoder()
train['CategoryInTamil'] = label_encoder.fit_transform(train['CategoryInTamil'])
Y = train['CategoryInTamil'].astype('category')
Y = pd.get_dummies(Y).values

# Tokenize words

num_words = 22000
tokenizer = Tokenizer(num_words=num_words,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=False)
tokenizer.fit_on_texts(train['NewsInTamil'])
print('Unique word count:', len(tokenizer.word_index))
x_train = tokenizer.texts_to_sequences(train['NewsInTamil'])
X = pad_sequences(x_train, maxlen=250, padding='post')


# Performance plot
def plt_performance(history):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'b', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'r', label='Validation accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'b', label='Training accuracy')
    plt.plot(history.history['val_loss'], 'r', label='Validation accuracy')
    plt.title('Loss')
    plt.legend()
    plt.show()


# LSTM Model

def build_lstm(num_filters, vocab_size, dropout, embedding_dim, maxlen, optimizer):
    sequential_lstm = Sequential()
    sequential_lstm.add(layers.Embedding(input_dim=vocab_size,
                                         output_dim=embedding_dim,
                                         input_length=maxlen))
    sequential_lstm.add(layers.SpatialDropout1D(dropout))
    sequential_lstm.add(layers.LSTM(num_filters, dropout=dropout, recurrent_dropout=dropout, return_sequences=True))
    sequential_lstm.add(layers.LSTM(num_filters, dropout=dropout, recurrent_dropout=dropout))
    sequential_lstm.add(layers.Dense(6, activation='softmax'))
    sequential_lstm.compile(loss='categorical_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])
    return sequential_lstm


# Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Model Building

sequential_cnn = Sequential()
vocab_size = len(tokenizer.word_index)+1
output_dim = 50
print(vocab_size)
print(X_train.shape[1])
input_length = X_train.shape[1]
sequential_cnn.add(layers.Embedding(input_dim=vocab_size,
                                output_dim=output_dim,
                                input_length=input_length))
sequential_cnn.add(layers.Conv1D(100, 5, activation='relu'))
sequential_cnn.add(layers.GlobalMaxPool1D())
sequential_cnn.add(layers.Dense(10, activation='relu'))
sequential_cnn.add(layers.Dense(6, activation='softmax'))
print(sequential_cnn.summary())
sequential_cnn.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
#history_cnn = sequential_cnn.fit(X_train, Y_train,
#                         epochs=10,
#                         batch_size=32,
#                         validation_data=(X_test, Y_test))

#plt_performance(history_cnn)
# accuracy - 1
# validation accuracy ~ 0.75

# Hyper-parameter Tuning
param_grid = dict(num_filters=[50, 64, 100, 128],
                  dropout =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  vocab_size=[vocab_size],
                  embedding_dim=[50],
                  maxlen=[100],
                  optimizer= ['adam', 'rmsprop'])
sequential_lstm = KerasClassifier(build_fn=build_lstm,
                                  epochs= 10,
                                  batch_size=32,
                                  verbose=False)
grid = RandomizedSearchCV(estimator= sequential_lstm,
                          param_distributions=param_grid,
                          cv=10,
                          n_iter=5)
result = grid.fit(X_train, Y_train)
score = grid.score(X_test, Y_test)
print('Optimal parameter values are: ', result.best_estimator_)
print('Accuracy of the fit is: ', score)

