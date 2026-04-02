import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def generar_caso_de_uso_predecir_popularidad_juego():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función predecir_popularidad_juego(X, y, test_size, random_state).
    """

    # --- Configuración aleatoria ---
    random_state = random.randint(0, 999)
    n_samples    = random.randint(150, 400)
    n_features   = random.randint(3, 8)
    test_size    = round(random.uniform(0.15, 0.40), 2)

    # Generar datos de clasificación binaria
    # (simula features de videojuegos: metacritic, horas, precio, reseñas, edad_jugadores)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 1),
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=random_state
    )

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

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=random_state)
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
    entrada, salida_esperada = generar_caso_de_uso_predecir_popularidad_juego()

    print("=== INPUT ===")
    print(f"X shape:       {entrada['X'].shape}")
    print(f"y shape:       {entrada['y'].shape}")
    print(f"test_size:     {entrada['test_size']}")
    print(f"random_state:  {entrada['random_state']}")

    print("\n=== OUTPUT ESPERADO ===")
    for k, v in salida_esperada.items():
        print(f"  {k}: {v:.4f}")
