import random

import numpy as np

from train import create_dictionaries, convert_to_one_hots
from keras.models import load_model

def sample_from_distribution(dist):
    r = random.random()
    bound = 0
    for n, d in enumerate(dist):
        bound += d
        if r <= bound:
            return n

def generate_text_impl(model, d, inverse_d, length):
    batch_size = 50
    c = random.choice(d.keys())
    x = None
    text = []
    for i in xrange(length):
        text.append(c)
        if x is None:
            x = convert_to_one_hots([c], d)
        else:
            x = np.append(x, convert_to_one_hots([c], d), axis=0)
            if len(x) > batch_size:
                x = x[1:]
        distribution = model.predict(np.array([x]))[0][-1]
        c = inverse_d[sample_from_distribution(distribution)]
    return "".join(text)

def generate_text(filename, model_path, length):
    with open(filename, 'r') as f:
        contents = f.read()

    # Now we need to create a dictionary
    d, inverse_d = create_dictionaries(contents)
    model = load_model(model_path)
    return generate_text_impl(model, d, inverse_d, length)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print "Usage: sample.py <filename> <model checkpoint path> <length of text to generate>"
        exit(1)

    print generate_text(sys.argv[1], sys.argv[2], int(sys.argv[3]))
