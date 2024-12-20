import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore

# 1. Membuat data contoh sederhana (ganti dengan data Anda sendiri)
# Misalnya data harian selama 2 tahun
dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
data = np.sin(np.arange(len(dates)) * 0.1) + np.random.normal(0, 0.1, len(dates))
df = pd.DataFrame(data, index=dates, columns=["value"])


# 2. Persiapkan data
def prepare_data(data, look_back=7):
    # Normalisasi data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i : i + look_back])
        y.append(scaled_data[i + look_back])

    return np.array(X), np.array(y), scaler


# 3. Buat dan latih model
def create_model(look_back):
    model = Sequential(
        [LSTM(50, activation="relu", input_shape=(look_back, 1)), Dense(1)]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# 4. Jalankan proses training dan prediksi
def main():
    # Siapkan data
    look_back = 7  # menggunakan 7 hari sebelumnya untuk prediksi
    X, y, scaler = prepare_data(df, look_back)

    # Bagi data menjadi train dan test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Buat dan latih model
    model = create_model(look_back)
    print("Mulai training model...")
    model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1
    )

    # Lakukan prediksi
    print("\nMembuat prediksi...")
    predictions = model.predict(X_test)

    # Transform kembali ke skala asli
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test)

    # Tampilkan beberapa hasil prediksi
    print("\nPerbandingan hasil prediksi dengan nilai sebenarnya:")
    print("Prediksi\t\tAktual")
    for pred, act in zip(predictions[:5], actual[:5]):
        print(f"{pred[0]:.2f}\t\t{act[0]:.2f}")

    # Plot hasil prediksi vs aktual
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Aktual", color="blue")
    plt.plot(predictions, label="Prediksi", color="red")
    plt.title("Perbandingan Nilai Aktual vs Prediksi")
    plt.xlabel("Timestep")
    plt.ylabel("Nilai")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_plot.png")
    plt.close()

    # Plot scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predictions)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", lw=2)
    plt.title("Scatter Plot Aktual vs Prediksi")
    plt.xlabel("Aktual")
    plt.ylabel("Prediksi")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_plot.png")
    plt.close()

    print("\nPlot telah disimpan sebagai 'prediction_plot.png' dan 'scatter_plot.png'")


if __name__ == "__main__":
    main()
