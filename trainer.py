from utils import train_model
from load_and_save_files import load_pickle, save_pickle

data = load_pickle("preprocessed_data.pkl")

train_x, train_y, test_x, test_y = data["train_x"], data["train_y"], data["test_x"], data["test_y"]

# # Train
# model_name = 'Support Vector Machine'
# model_name = 'K Nearest Neighbors'
# model_name = 'Random Forest'
model_name = "Multi Layer Perceptron"

model = train_model(model_name, train_x, train_y, test_x, test_y)

# change model name small letter and remove spaces with underscore
model_name = model_name.lower().replace(" ", "_")
filename = f"{model_name}.pkl"
save_pickle(model, filename)
