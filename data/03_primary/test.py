import pickle
import json

# 1. Wczytaj plik
with open("circuit_lap_data.pickle", "rb") as f:
    data_dict = pickle.load(f)

# 2. Wybierz tor (np. pierwszy ze słownika)
circuit_name = list(data_dict.keys())[0]
df = data_dict[circuit_name]

# 3. Konwersja na JSON (orient='records' tworzy listę obiektów/wierszy)
# Używamy .head(3), żeby nie wyświetlać tysięcy wierszy
json_data = df.head(3).to_json(orient="records", date_format="iso")

# 4. Ładne formatowanie (indentacja) przy użyciu modułu json
parsed = json.loads(json_data)
formatted_json = json.dumps(parsed, indent=4)

print(f"Podgląd dla toru: {circuit_name}")
print(formatted_json)
