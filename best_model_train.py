from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from recognition.load_and_save_files import load_pickle, save_pickle

# Ön işleme yapılmış verileri yüklüyoruz.
data = load_pickle("data/preprocessed_data.pkl")

# Verileri ayırıyoruz.
train_x, train_y, test_x, test_y = data["train_x"], data["train_y"], data["test_x"], data["test_y"]

# %83 accuracy ile Random Forest modeli en iyi model olarak seçildi.
# Şimdi amaç bu modelin hyperparameterlarını optimize ederek daha iyi bir model elde etmek.
# ve accuracy'i arttırarak bu modeli best_model.pkl olarak kaydetmek.

# modeli çağırıyoruz.
model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
)

model.fit(train_x, train_y.values.flatten())  # modeli eğitiyoruz.
y_pred = model.predict(test_x)  # modeli test seti üzerinde test ediyoruz.
test_acc = accuracy_score(test_y, y_pred) * 100  # test accuracy hesaplıyoruz.
print("Test accuracy: ", test_acc)  # test accuracy'i yazdırıyoruz.
print(classification_report(test_y, y_pred))  # önemli metrikleri yazdırıyoruz.

# modeli kaydediyoruz.
filename = "model/best_model.pkl"
save_pickle(model, filename)

# Yeni accuracy %85 oldu.
# Sonuç olarak model daha iyi hale gelmiş oldu.
