import os
import json
import glob


def main():
    metrics_dir = "metrics"

    if not os.path.exists(metrics_dir):
        print(
            f"Katalog '{metrics_dir}' nie istnieje. Uruchom najpierw trening, aby go wygenerować."
        )
        return

    json_files = glob.glob(os.path.join(metrics_dir, "*.json"))

    if not json_files:
        print(f"Nie znaleziono plików JSON w katalogu '{metrics_dir}'.")
        return

    # Nagłówek tabeli
    print(
        f"{'Eksperyment':<30} | {'Najlepsza średnia (Best)':<25} | {'Ostatnia średnia (Last)':<25}"
    )
    print("-" * 86)

    for file_path in json_files:
        experiment_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            moving_avg_history = data.get("moving_avg_history", [])

            if moving_avg_history:
                best_avg = max(moving_avg_history)
                last_avg = moving_avg_history[-1]
                print(f"{experiment_name:<30} | {best_avg:<25.2f} | {last_avg:<25.2f}")
            else:
                print(
                    f"{experiment_name:<30} | {'Brak danych':<25} | {'Brak danych':<25}"
                )
        except Exception as e:
            print(f"Błąd podczas czytania pliku {file_path}: {e}")


if __name__ == "__main__":
    main()
