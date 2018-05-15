import numpy as np
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed
from keras.callbacks import ModelCheckpoint

def create_dictionaries(s):
    unique_chars = list(set(s))
    dictionary = dict([(c, n) for n, c in enumerate(unique_chars)])
    inverse_dictionary = dict([(n, c) for n, c in enumerate(unique_chars)])
    return (dictionary, inverse_dictionary)

# The next thing is to learn how to convert letters to one-hot vectors
def convert_to_one_hots(s, d):
    indices = np.array([d[c] for c in s])
    one_hots = np.zeros((len(indices), len(d)))
    one_hots[np.arange(len(indices)), indices] = 1
    return one_hots

def convert_to_sequences(data, batch_size, d):
    one_hots = convert_to_one_hots(data, d)
    x = one_hots
    y = np.roll(one_hots, -1, axis=0)
    X = []
    Y = []
    for n in xrange(len(data)/batch_size):
        X.append(x[n*batch_size:(n+1)*batch_size])
        Y.append(y[n*batch_size:(n+1)*batch_size])
    return (np.array(X), np.array(Y))

def train_model(filenames):
    contents_arr = []
    for filename in filenames:
        with open(filename, 'r') as f:
            contents_arr.append(f.read())
    contents = "\n".join(contents_arr)

    # Now we need to create a dictionary
    d, inverse_d = create_dictionaries(contents)
    data_size = len(contents)

    print "Dictionary size", len(d)
    print "File size", data_size

    train_size = int(data_size*0.8)
    validation_size = int(data_size*0.1)

    train_data = contents[:train_size]
    validation_data = contents[train_size:train_size+validation_size]
    test_data = contents[train_size+validation_size:]
    test_size = len(test_data)

    # Time to create LSTM
    space_dim = len(d)
    batch_size = 50
    minibatch_size = 50

    model = Sequential()
    model.add(LSTM(128, input_shape=(None, space_dim), return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(space_dim, activation="softmax")))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_X, train_Y = convert_to_sequences(train_data, batch_size, d)
    validation_X, validation_Y = convert_to_sequences(validation_data, batch_size, d)

    model.fit(train_X, train_Y, minibatch_size, 100,
              shuffle=False,
              validation_data=(validation_X, validation_Y),
              callbacks=[ModelCheckpoint("checkpoint.{epoch:02d}.hdf5")])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RNN model to generate strings of text")
    parser.add_argument("--train-files", type=str, help="Path to train files", nargs='*', required=True)

    args = parser.parse_args()
    print "Specified filenames", args.train_files
    train_model(args.train_files)
