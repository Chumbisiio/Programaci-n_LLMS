import pandas as pd
import numpy as np
import random


def generar_caso_de_uso_limpiar_y_rankear_jugadores():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función limpiar_y_rankear_jugadores(df, top_n).
    """

    # --- Configuración aleatoria ---
    n_jugadores = random.randint(6, 15)
    nombres = [f'Jugador_{chr(65+i)}' for i in range(n_jugadores)]

    # Generar stats base
    kills    = np.random.randint(0, 30, size=n_jugadores).astype(float)
    deaths   = np.random.randint(0, 15, size=n_jugadores).astype(float)
    assists  = np.random.randint(0, 20, size=n_jugadores).astype(float)
    partidas = np.random.randint(1, 30, size=n_jugadores).astype(float)

    df = pd.DataFrame({
        'jugador':          nombres,
        'kills':            kills,
        'deaths':           deaths,
        'assists':          assists,
        'partidas_jugadas': partidas
    })

    # Introducir NaNs en columnas numéricas (~15% de los datos)
    for col in ['kills', 'deaths', 'assists', 'partidas_jugadas']:
        mask = np.random.choice([True, False], size=n_jugadores, p=[0.15, 0.85])
        df.loc[mask, col] = np.nan

    # Elegir top_n aleatoriamente
    top_n = random.randint(2, max(2, n_jugadores - 2))

    # --- Input ---
    input_data = {
        'df':    df.copy(),
        'top_n': top_n
    }

    # --- Output esperado (ground truth) ---
    df_clean = df.dropna().copy()

    df_clean['kda'] = (df_clean['kills'] + df_clean['assists']) / df_clean['deaths'].clip(lower=1)

    df_result = (
        df_clean[['jugador', 'kda', 'partidas_jugadas']]
        .sort_values('kda', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    output_data = df_result

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_limpiar_y_rankear_jugadores()

    print("=== INPUT ===")
    print(f"top_n: {entrada['top_n']}")
    print("DataFrame:")
    print(entrada['df'])

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
