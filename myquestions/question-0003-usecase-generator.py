import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def generar_caso_de_uso_predecir_victoria_naive_bayes():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función predecir_victoria_naive_bayes(X, y, test_size, random_state).
    """

    # --- Configuración aleatoria ---
    random_state = random.randint(0, 999)
    n_samples    = random.randint(150, 500)
    n_features   = random.randint(3, 8)
    test_size    = round(random.uniform(0.15, 0.40), 2)

    # Datos binarios que simulan estadísticas de jugadores
    # (kda_promedio, tasa_headshots, horas_semana, nivel_cuenta, racha_victorias...)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 1),
        n_redundant=1,
        n_clusters_per_class=1,
        flip_y=0.05,         # algo de ruido para no tener accuracy=1.0
        random_state=random_state
    )

    # Llevar a rango positivo (simula características de videojuegos)
    X = X - X.min(axis=0)

    # --- Input ---
    input_data = {
        'X':            X,
        'y':            y,
        'test_size':    test_size,
        'random_state': random_state
    }

    # --- Output esperado (ground truth) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    output_data = {
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score':  float(f1_score(y_test, y_pred, zero_division=0))
    }

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_predecir_victoria_naive_bayes()

    print("=== INPUT ===")
    print(f"X shape:       {entrada['X'].shape}")
    print(f"y shape:       {entrada['y'].shape}")
    print(f"test_size:     {entrada['test_size']}")
    print(f"random_state:  {entrada['random_state']}")

    print("\n=== OUTPUT ESPERADO ===")
    for k, v in salida_esperada.items():
        print(f"  {k}: {v:.4f}")
