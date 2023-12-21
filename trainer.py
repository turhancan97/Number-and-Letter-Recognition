import pandas as pd
from utils import train_model
from load_and_save_files import load_pickle, save_pickle

data = load_pickle("data/preprocessed_data.pkl")

train_x, train_y, test_x, test_y = data["train_x"], data["train_y"], data["test_x"], data["test_y"]

# # Train
model_names = ['Logistic Regression', 'K Nearest Neighbors', 'Multi Layer Perceptron', 'Support Vector Machine']
model_accuracy = {}
for model_name in model_names:
    model, accuracy = train_model(model_name, train_x, train_y, test_x, test_y)
    model_accuracy[model_name] = accuracy
    # change model name small letter and remove spaces with underscore
    model_name = model_name.lower().replace(" ", "_")
    filename = f"model/{model_name}.pkl"
    save_pickle(model, filename)

# model doğruluğunu dataframe olarak yazdırın ve doğruluğa göre sıralayın
model_accuracy = pd.DataFrame(model_accuracy.items(), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
print("Model accuracy:")
print(model_accuracy)
# En iyi modeli yazdır.
print("Best model:")
print(model_accuracy.head(1))

# En iyi Model Yapay Sinir Ağları çıktı.
# best_model_train.py dosyasında bu modelin hyperparameterları optimize edilerek daha iyi bir model elde edilmeye çalışıldı.