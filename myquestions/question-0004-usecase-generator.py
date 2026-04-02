import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def generar_caso_de_uso_agrupar_jugadores_por_habilidad():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función agrupar_jugadores_por_habilidad(X, n_clusters).
    """

    # --- Configuración aleatoria ---
    n_clusters  = random.randint(2, 5)
    n_samples   = random.randint(n_clusters * 10, n_clusters * 30)
    n_features  = random.randint(3, 7)

    # Simular estadísticas de jugadores con grupos naturales
    # (kda, win_rate, horas, nivel, avg_score, headshot_rate, partidas...)
    X_parts = []
    samples_per_cluster = n_samples // n_clusters
    for k in range(n_clusters):
        center = np.random.uniform(low=k * 2.0, high=k * 2.0 + 1.5, size=n_features)
        chunk  = center + np.random.randn(samples_per_cluster, n_features) * 0.5
        X_parts.append(chunk)

    # Completar con el residuo
    remainder = n_samples - samples_per_cluster * n_clusters
    if remainder > 0:
        extra = np.random.randn(remainder, n_features)
        X_parts.append(extra)

    X = np.vstack(X_parts)
    np.random.shuffle(X)  # mezclar filas

    # Introducir NaNs (~12% de los datos)
    nan_mask = np.random.choice([True, False], size=X.shape, p=[0.12, 0.88])
    X_with_nan = X.copy()
    X_with_nan[nan_mask] = np.nan

    # --- Input ---
    input_data = {
        'X':          X_with_nan,
        'n_clusters': n_clusters
    }

    # --- Output esperado (ground truth) ---
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_with_nan)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    output_data = labels  # np.ndarray de forma (n_samples,)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_agrupar_jugadores_por_habilidad()

    print("=== INPUT ===")
    print(f"X shape:     {entrada['X'].shape}")
    print(f"n_clusters:  {entrada['n_clusters']}")
    print(f"NaNs en X:   {np.isnan(entrada['X']).sum()}")

    print("\n=== OUTPUT ESPERADO ===")
    print(f"Labels shape: {salida_esperada.shape}")
    print(f"Clusters únicos: {np.unique(salida_esperada)}")
    print(f"Primeras 10 etiquetas: {salida_esperada[:10]}")
