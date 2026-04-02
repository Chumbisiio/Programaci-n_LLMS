import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


def generar_caso_de_uso_clasificar_resultado_con_mlp():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función clasificar_resultado_con_mlp(X, y, capas_ocultas, test_size, random_state).
    """

    # --- Configuración aleatoria ---
    random_state = random.randint(0, 999)
    n_classes    = random.randint(2, 3)        # 2 o 3 clases (victoria/derrota o + empate)
    n_samples    = random.randint(250, 500)
    n_features   = random.randint(4, 7)
    test_size    = round(random.uniform(0.15, 0.35), 2)

    # Arquitectura de red aleatoria
    n_capas   = random.randint(1, 3)
    capas_ocultas = tuple(
        random.choice([32, 64, 128]) for _ in range(n_capas)
    )

    # Datos que simulan estadísticas de equipo en partida
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=max(n_classes, n_features - 1),
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=random_state
    )

    # --- Input ---
    input_data = {
        'X':             X,
        'y':             y,
        'capas_ocultas': capas_ocultas,
        'test_size':     test_size,
        'random_state':  random_state
    }

    # --- Output esperado (ground truth) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=capas_ocultas,
        max_iter=500,
        random_state=random_state
    )
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    output_data = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    }

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_clasificar_resultado_con_mlp()

    print("=== INPUT ===")
    print(f"X shape:        {entrada['X'].shape}")
    print(f"y shape:        {entrada['y'].shape}")
    print(f"capas_ocultas:  {entrada['capas_ocultas']}")
    print(f"test_size:      {entrada['test_size']}")
    print(f"random_state:   {entrada['random_state']}")

    print("\n=== OUTPUT ESPERADO ===")
    for k, v in salida_esperada.items():
        print(f"  {k}: {v:.4f}")
