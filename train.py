import pandas as pd
from recognition.utils import train_model
from recognition.load_and_save_files import load_pickle, save_pickle

# Load preprocessed data.
data = load_pickle("data/preprocessed_data.pkl")

# Split the data.
train_x, train_y, test_x, test_y = (
    data["train_x"],
    data["train_y"],
    data["test_x"],
    data["test_y"],
)

# Train with 4 different models.
# Store them in a list.
# Create a dictionary to store model names and accuracies.
# Then, train th model, calculate the accuracy, and add it to the dictionary in a loop.
model_names = [
    "Logistic Regression",
    "K Nearest Neighbors",
    "Multi Layer Perceptron",
    "Decision Tree",
    "Random Forest",
]
model_accuracy = {}
for model_name in model_names:
    model, accuracy = train_model(model_name, train_x, train_y, test_x, test_y)
    model_accuracy[model_name] = accuracy
    # Change model name to lowercase and remove spaces with underscores.
    model_name = model_name.lower().replace(" ", "_")
    filename = f"model/{model_name}.pkl"
    save_pickle(model, filename)

# Print model accuracy as a dataframe and sort by accuracy.
model_accuracy = pd.DataFrame(
    model_accuracy.items(), columns=["Model", "Accuracy"]
).sort_values(by="Accuracy", ascending=False)
print("Model accuracy:")
print(model_accuracy)
# Print the best model.
print("Best model:")
print(model_accuracy.head(1))

# The best model is Artificial Neural Networks.
# In the best_model_train.py file, hyperparameters of this model were optimized to obtain a better model.
