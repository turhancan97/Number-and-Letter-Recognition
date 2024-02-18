import pandas as pd
import numpy as np
from recognition.load_and_save_files import load_mnist, load_alphabet, save_pickle

# Load MNIST and Alphanet datasets
train_x_mnist, train_y_mnist, test_x_mnist, test_y_mnist = load_mnist()
train_x_alphabet, train_y_alphabet, valid_x_alphabet, valid_y_alphabet, test_x_alphabet, test_y_alphabet = load_alphabet()

# Merge training and validation data in the Alphabet dataset because they will be used as training data
train_x_alphabet = np.concatenate((train_x_alphabet, valid_x_alphabet))
train_y_alphabet = np.concatenate((train_y_alphabet, valid_y_alphabet))

# Convert to Pandas DataFrames
train_x_mnist, train_y_mnist, test_x_mnist, test_y_mnist = [
    pd.DataFrame(x) for x in [train_x_mnist, train_y_mnist, test_x_mnist, test_y_mnist]
]
train_x_alphabet, train_y_alphabet, test_x_alphabet, test_y_alphabet = [
    pd.DataFrame(x)
    for x in [train_x_alphabet, train_y_alphabet, test_x_alphabet, test_y_alphabet]
]

# Shuffle the data in the MNIST dataset as the data is stored in a sequential manner
train_x_mnist, train_y_mnist = train_x_mnist.sample(
    frac=1, random_state=42
), train_y_mnist.sample(frac=1, random_state=42)
test_x_mnist, test_y_mnist = test_x_mnist.sample(
    frac=1, random_state=42
), test_y_mnist.sample(frac=1, random_state=42)
train_x_alphabet, train_y_alphabet = train_x_alphabet.sample(
    frac=1, random_state=42
), train_y_alphabet.sample(frac=1, random_state=42)
test_x_alphabet, test_y_alphabet = test_x_alphabet.sample(
    frac=1, random_state=42
), test_y_alphabet.sample(frac=1, random_state=42)

# There are 60000 training data in the MNIST dataset. There are approximately 30000 training data in the Alphanet dataset.
# If we train like this, the MNIST dataset will have much more weight.
# And this will cause the model to better predict the numbers in the MNIST dataset.
# Therefore, we will use 40% of the data in the MNIST dataset.
# We will do the same for the test data as well.
# This way, the training and test data for numbers and letters will have a similar distribution.
train_x_mnist = train_x_mnist.iloc[: int(len(train_x_mnist) * 0.4)]
train_y_mnist = train_y_mnist.iloc[: int(len(train_y_mnist) * 0.4)]
train_x = pd.concat([train_x_alphabet, train_x_mnist])
train_y = pd.concat([train_y_alphabet, train_y_mnist])

test_x_mnist = test_x_mnist.iloc[: int(len(test_x_mnist) * 0.4)]
test_y_mnist = test_y_mnist.iloc[: int(len(test_y_mnist) * 0.4)]
test_x = pd.concat([test_x_alphabet, test_x_mnist])
test_y = pd.concat([test_y_alphabet, test_y_mnist])

print("*************************************")
print("Shape of train and test data:\n")
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
print("*************************************")

# Shuffle the data again to avoid any issues
train_x, train_y = train_x.sample(frac=1, random_state=42), train_y.sample(
    frac=1, random_state=42
)
test_x, test_y = test_x.sample(frac=1, random_state=42), test_y.sample(
    frac=1, random_state=42
)

# Normalize the data.
# This allows the model to train faster and produce better results.
train_x = train_x / 255.0
test_x = test_x / 255.0

# Save as pickle file named Preprocessed_data.pkl
data = {
    "train_x": train_x,
    "train_y": train_y,
    "test_x": test_x,
    "test_y": test_y,
}
save_pickle(data, "data/preprocessed_data.pkl")
