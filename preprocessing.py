import pandas as pd
import numpy as np
from load_and_save_files import load_mnist, load_alphabet, save_pickle

train_x_mnist, train_y_mnist, test_x_mnist, test_y_mnist = load_mnist()

train_x_alphabet, train_y_alphabet, valid_x_alphabet, valid_y_alphabet, test_x_alphabet, test_y_alphabet = load_alphabet()

# add validation to the end of training set
train_x_alphabet = np.concatenate((train_x_alphabet, valid_x_alphabet))
train_y_alphabet = np.concatenate((train_y_alphabet, valid_y_alphabet))

# Convert to pandas dataframe
train_x_mnist, train_y_mnist, test_x_mnist, test_y_mnist = [
    pd.DataFrame(x) for x in [train_x_mnist, train_y_mnist, test_x_mnist, test_y_mnist]
]
train_x_alphabet, train_y_alphabet, test_x_alphabet, test_y_alphabet = [
    pd.DataFrame(x)
    for x in [train_x_alphabet, train_y_alphabet, test_x_alphabet, test_y_alphabet]
]

# suffle
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

# Take %10 of mnsit data and add to alphabet data
train_x_mnist = train_x_mnist.iloc[: int(len(train_x_mnist) * 0.1)]
train_y_mnist = train_y_mnist.iloc[: int(len(train_y_mnist) * 0.1)]
train_x = pd.concat([train_x_alphabet, train_x_mnist])
train_y = pd.concat([train_y_alphabet, train_y_mnist])

# same for test data
test_x_mnist = test_x_mnist.iloc[: int(len(test_x_mnist) * 0.075)]
test_y_mnist = test_y_mnist.iloc[: int(len(test_y_mnist) * 0.075)]
test_x = pd.concat([test_x_alphabet, test_x_mnist])
test_y = pd.concat([test_y_alphabet, test_y_mnist])

print("*************************************")
print("Shape of train and test data:\n")
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
print("*************************************")

# suffle again
train_x, train_y = train_x.sample(frac=1, random_state=42), train_y.sample(
    frac=1, random_state=42
)
test_x, test_y = test_x.sample(frac=1, random_state=42), test_y.sample(
    frac=1, random_state=42
)

# Normalize
train_x = train_x / 255.0
test_x = test_x / 255.0

# Save as pickle with name preprocessed_data.pkl
data = {
    "train_x": train_x,
    "train_y": train_y,
    "test_x": test_x,
    "test_y": test_y,
}
save_pickle(data, "data/preprocessed_data.pkl")
