from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from load_and_save_files import load_pickle, save_pickle

# Ön işleme yapılmış verileri yüklüyoruz.
data = load_pickle("data/preprocessed_data.pkl")

# Verileri ayırıyoruz.
train_x, train_y, test_x, test_y = data["train_x"], data["train_y"], data["test_x"], data["test_y"]

# %73 accuracy ile yapay sinir ağları modeli en iyi model olarak seçildi.
# Şimdi amaç bu modelin hyperparameterlarını optimize ederek daha iyi bir model elde etmek.
# ve accuracy'i arttırarak bu modeli best_model.pkl olarak kaydetmek.

# modeli çağırıyoruz.
model = MLPClassifier(
    hidden_layer_sizes=(200, 200, 200, 200, 200),
    max_iter=500,
    n_iter_no_change=50,
    verbose=True,
    random_state=42,
)

model.fit(train_x, train_y.values.flatten()) # modeli eğitiyoruz.
y_pred = model.predict(test_x) # modeli test seti üzerinde test ediyoruz.
test_acc = accuracy_score(test_y, y_pred) * 100 # test accuracy hesaplıyoruz.
print("Test accuracy: ", test_acc) # test accuracy'i yazdırıyoruz.
print(classification_report(test_y, y_pred)) # önemli metrikleri yazdırıyoruz.

# modeli kaydediyoruz.
filename = "model/best_model.pkl"
save_pickle(model, filename)

# Yeni accuracy %89 oldu.
# Sonuç olarak model daha iyi hale gelmiş oldu.
