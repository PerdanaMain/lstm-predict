import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore


def prepare_data(data, look_back=7):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i : i + look_back])
        y.append(scaled_data[i + look_back])

    return np.array(X), np.array(y), scaler


def create_model(look_back):
    model = Sequential(
        [LSTM(50, activation="relu", input_shape=(look_back, 1)), Dense(1)]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def predict_future(model, last_sequence, scaler, n_future=30):
    """
    Memprediksi n_future hari ke depan
    """
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(n_future):
        # Prediksi nilai berikutnya
        current_prediction = model.predict(current_sequence.reshape(1, -1, 1))
        future_predictions.append(current_prediction[0, 0])

        # Update sequence untuk prediksi berikutnya
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = current_prediction

    # Transform kembali ke skala asli
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)

    return future_predictions


def main():
    # 1. Membuat data contoh sederhana
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
    data = np.sin(np.arange(len(dates)) * 0.1) + np.random.normal(0, 0.1, len(dates))
    df = pd.DataFrame(data, index=dates, columns=["value"])

    # Siapkan data
    look_back = 7
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

    # Prediksi pada data test
    print("\nMembuat prediksi...")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test)

    # Prediksi 30 hari ke depan
    print("\nMemprediksi 30 hari ke depan...")
    last_sequence = X[-1]  # Mengambil sequence terakhir
    future_pred = predict_future(model, last_sequence, scaler, n_future=30)

    # Buat tanggal untuk prediksi masa depan
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=30, freq="D"
    )

    # Plot hasil
    plt.figure(figsize=(15, 7))

    # Plot data historis
    plt.plot(df.index[-100:], df["value"][-100:], label="Data Historis", color="blue")

    # Plot prediksi masa depan
    plt.plot(
        future_dates,
        future_pred,
        label="Prediksi 30 Hari Kedepan",
        color="red",
        linestyle="--",
    )

    plt.title("Prediksi 30 Hari Kedepan")
    plt.xlabel("Tanggal")
    plt.ylabel("Nilai")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("future_prediction_plot.png")
    plt.close()

    # Tampilkan prediksi dalam format tabel
    future_df = pd.DataFrame(future_pred, index=future_dates, columns=["Prediksi"])
    print("\nPrediksi 30 hari ke depan:")
    print(future_df)
    print("\nPlot telah disimpan sebagai 'future_prediction_plot.png'")


if __name__ == "__main__":

    main()
