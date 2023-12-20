import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_mnist():
    with open('alphabet.pkl', 'rb') as f:
        mnist = pickle.load(f)
    return mnist['training_images'], mnist['training_labels'], mnist['test_images'], mnist['test_labels']

train_x, train_y, test_x, test_y = load_mnist()
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# Convert to pandas dataframe
train_x, train_y, test_x, test_y = [pd.DataFrame(x) for x in [train_x, train_y, test_x, test_y]]

# suffle
train_x, train_y = train_x.sample(frac=1, random_state=42), train_y.sample(frac=1, random_state=42)
test_x, test_y = test_x.sample(frac=1, random_state=42), test_y.sample(frac=1, random_state=42)

# show the first 16 images with matplotlib and write the labels on the plot
fig, ax = plt.subplots(4, 4)
for i, axi in enumerate(ax.flat):
    axi.imshow(train_x.iloc[i].values.reshape(28, 28), cmap='binary')
    letter = chr(train_y.iloc[i].values[0] + ord('a'))
    axi.set(xticks=[], yticks=[], xlabel=letter)
ax[0, 0].set(title='Examples of digits\n');
plt.show()

# Normalize
train_x = train_x/255.0
test_x = test_x/255.0

svc = SVC(verbose=True, random_state=42)

svc.fit(train_x, train_y.values.flatten())

filename = "svm_model.pkl"
pickle.dump(svc, open(filename, 'wb'))

y_pred = svc.predict(test_x)
print(classification_report(test_y, y_pred))