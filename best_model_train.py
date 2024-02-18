from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
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

# Random Forest model with 83% accuracy is selected as the best model.
# Now, the goal is to optimize the hyperparameters of this model to obtain a better model.
# Increase the accuracy and save this model as best_model.pkl.

# Initialize the model.
model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
)

model.fit(train_x, train_y.values.flatten())  # Train the model.
y_pred = model.predict(test_x)  # Test the model on the test set.
test_acc = accuracy_score(test_y, y_pred) * 100  # Calculate test accuracy.
print("Test accuracy: ", test_acc)  # Print test accuracy.
print(classification_report(test_y, y_pred))  # Print important metrics.

# Save the model.
filename = "model/best_model.pkl"
save_pickle(model, filename)

# The new accuracy is 85%.
# As a result, the model has improved.
