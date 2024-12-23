import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from model import get_values  # Import fungsi get_values dari model.py


def prepare_data(data, look_back=7):
    """
    Mempersiapkan data untuk model LSTM
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i : (i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])

    return np.array(X), np.array(y), scaler


def create_model(look_back):
    """
    Membuat model LSTM
    """
    model = Sequential(
        [
            LSTM(
                50, activation="relu", input_shape=(look_back, 1), return_sequences=True
            ),
            LSTM(50, activation="relu"),
            Dense(25, activation="relu"),
            Dense(1),
        ]
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
        # Reshape sequence untuk prediksi
        current_input = current_sequence.reshape((1, len(current_sequence), 1))
        # Prediksi nilai berikutnya
        next_pred = model.predict(current_input, verbose=0)
        # Tambahkan ke list prediksi
        future_predictions.append(next_pred[0, 0])
        # Update sequence untuk prediksi berikutnya
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred

    # Transform kembali ke skala asli
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    return scaler.inverse_transform(future_predictions)


def main():
    # Ambil data dari database menggunakan get_values
    data_pairs = get_values(
        "a586c9e4-dfa6-44fa-bc05-cba092eef33d", "88a07a75-1f84-4436-bcf0-12739900bf4a"
    )

    # Konversi data ke format yang sesuai
    dates = [pair[0] for pair in data_pairs]
    values = np.array([float(pair[1]) for pair in data_pairs])

    # Parameter
    look_back = 7  # Window size untuk sequence
    future_days = 30  # Jumlah hari untuk diprediksi

    # Persiapkan data
    X, y, scaler = prepare_data(values, look_back)

    # Reshape data untuk LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((len(y), 1))

    # Bagi data menjadi train dan test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Buat dan latih model
    model = create_model(look_back)
    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=150,  # Meningkatkan epochs untuk training lebih lama
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    # Prediksi pada data train dan test
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform kembali ke skala asli
    train_predict = scaler.inverse_transform(train_predict)
    y_train_orig = scaler.inverse_transform(y_train)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_orig = scaler.inverse_transform(y_test)

    # Prediksi 30 hari ke depan
    last_sequence = X[-1]
    future_pred = predict_future(model, last_sequence, scaler, n_future=future_days)

    # Buat tanggal untuk prediksi masa depan
    last_date = dates[-1]
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(future_days)]

    # Plot hasil
    plt.figure(figsize=(15, 8))

    # Plot data aktual
    plt.plot(dates, values, label="Data Aktual", color="blue", linewidth=2)

    # Plot hasil training
    train_dates = dates[look_back : train_size + look_back]
    plt.plot(
        train_dates,
        train_predict,
        label="Training Predictions",
        color="green",
        alpha=0.7,
    )

    # Plot hasil testing
    test_dates = dates[train_size + look_back : len(dates)]
    plt.plot(
        test_dates, test_predict, label="Testing Predictions", color="red", alpha=0.7
    )

    # Plot prediksi masa depan
    plt.plot(
        future_dates,
        future_pred,
        label="Prediksi 30 Hari Kedepan",
        color="purple",
        linestyle="--",
        linewidth=2,
    )

    plt.title("Prediksi Time Series dengan LSTM - 30 Hari Kedepan", pad=20)
    plt.xlabel("Tanggal")
    plt.ylabel("Nilai")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Tambahkan area yang diarsir untuk prediksi masa depan
    plt.axvspan(
        future_dates[0],
        future_dates[-1],
        color="gray",
        alpha=0.1,
        label="Area Prediksi",
    )

    plt.tight_layout()

    # Simpan plot
    plt.savefig("lstm_predictions_30days.png", dpi=300, bbox_inches="tight")
    print("\nPlot telah disimpan sebagai 'lstm_predictions_30days.png'")

    # Tampilkan prediksi masa depan
    print("\nPrediksi untuk 30 hari ke depan:")
    future_df = pd.DataFrame(
        {"Tanggal": future_dates, "Prediksi": future_pred.flatten()}
    )
    print(future_df)

    # Evaluasi model
    train_rmse = np.sqrt(np.mean((train_predict - y_train_orig) ** 2))
    test_rmse = np.sqrt(np.mean((test_predict - y_test_orig) ** 2))
    print(f"\nRMSE Training: {train_rmse:.2f}")
    print(f"RMSE Testing: {test_rmse:.2f}")

    # Simpan hasil prediksi ke file
    results_df = pd.DataFrame(
        {
            "Tanggal": dates[look_back:],
            "Nilai_Aktual": values[look_back:],
            "Prediksi": np.concatenate(
                [train_predict.flatten(), test_predict.flatten()]
            ),
        }
    )
    results_df.to_csv("hasil_prediksi_historis.csv", index=False)

    # Simpan prediksi masa depan ke file terpisah
    future_df.to_csv("prediksi_30hari.csv", index=False)
    print("\nHasil prediksi telah disimpan ke:")
    print("1. 'hasil_prediksi_historis.csv' (data historis)")
    print("2. 'prediksi_30hari.csv' (prediksi 30 hari kedepan)")


if __name__ == "__main__":
    main()
