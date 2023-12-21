import pickle

def load_mnist():
    with open('data/mnist.pkl', 'rb') as f:
        mnist = pickle.load(f)
    return mnist['training_images'], mnist['training_labels'], mnist['test_images'], mnist['test_labels']

def load_alphabet():
    with open('data/alphabet.pkl', 'rb') as f:
        alphabet = pickle.load(f)
        training_samples = alphabet['training_images'], alphabet['training_labels']
        validation_samples = alphabet['validation_images'], alphabet['validation_labels']
        test_samples = alphabet['test_images'], alphabet['test_labels']
    return training_samples[0], training_samples[1], validation_samples[0], validation_samples[1], test_samples[0], test_samples[1]

def save_pickle(data, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(input_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data
