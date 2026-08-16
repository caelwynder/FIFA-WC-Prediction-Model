import pandas as pd
from pathlib import Path

# Paths

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_FILE = DATA_DIR / "00_merged_dataset.csv"

# === DICTIONARY ===
tournament_weights = {

    # 20 — Friendlies
    "Friendly": 20,

    # 60 — World Cup Finals
    "FIFA World Cup": 60,

    # 50 — Continental Finals & Major Intercontinental
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "UEFA Euro": 50,
    "Copa América": 50,
    "Gold Cup": 50,
    "Oceania Nations Cup": 50,
    "AFF Championship": 50,
    "ASEAN Championship": 50,
    "EAFF Championship": 50,
    "SAFF Cup": 50,
    "Gulf Cup": 50,
    "CAFA Nations Cup": 50,
    "Confederations Cup": 50,
    "UEFA Nations League": 50,
    "CONCACAF Nations League": 50,
    "CONMEBOL–UEFA Cup of Champions": 50,

    # 40 — Qualifiers & Major Tournaments
    "FIFA World Cup qualification": 40,
    "African Cup of Nations qualification": 40,
    "AFC Asian Cup qualification": 40,
    "UEFA Euro qualification": 40,
    "Copa América qualification": 40,
    "Gold Cup qualification": 40,
    "Oceania Nations Cup qualification": 40,
    "AFF Championship qualification": 40,
    "ASEAN Championship qualification": 40,
    "EAFF Championship qualification": 40,
    "Arab Cup qualification": 40,
    "AFC Challenge Cup qualification": 40,
    "COSAFA Cup qualification": 40,
    "CFU Caribbean Cup qualification": 40,
    "CONCACAF Nations League qualification": 40,

    # 30 — All Other Tournaments
    "Nordic Championship": 30,
    "Cyprus International Tournament": 30,
    "Lunar New Year Cup": 30,
    "Malta International Tournament": 30,
    "King's Cup": 30,
    "COSAFA Cup": 30,
    "Melanesia Cup": 30,
    "Baltic Cup": 30,
    "Amílcar Cabral Cup": 30,
    "WAFF Championship": 30,
    "USA Cup": 30,
    "King Hassan II Tournament": 30,
    "Merdeka Tournament": 30,
    "United Arab Emirates Friendship Tournament": 30,
    "CECAFA Cup": 30,
    "Millennium Cup": 30,
    "Cup of Ancient Civilizations": 30,
    "Windward Islands Tournament": 30,
    "CFU Caribbean Cup": 30,
    "UNCAF Cup": 30,
    "Island Games": 30,
    "SKN Football Festival": 30,
    "Prime Minister's Cup": 30,
    "Unity Cup": 30,
    "The Other Final": 30,
    "Arab Cup": 30,
    "TIFOCO Tournament": 30,
    "South Pacific Games": 30,
    "Indian Ocean Island Games": 30,
    "Afro-Asian Games": 30,
    "AFC Challenge Cup": 30,
    "FIFI Wild Cup": 30,
    "Copa del Pacífico": 30,
    "Nehru Cup": 30,
    "Coupe de l'Outre-Mer": 30,
    "VFF Cup": 30,
    "Corsica Cup": 30,
    "Dragon Cup": 30,
    "ABCS Tournament": 30,
    "Nile Basin Tournament": 30,
    "Nations Cup": 30,
    "Copa Paz del Chaco": 30,
    "Copa Confraternidad": 30,
    "Pacific Games": 30,
    "Superclásico de las Américas": 30,
    "Viva World Cup": 30,
    "Kirin Cup": 30,
    "Kirin Challenge Cup": 30,
    "OSN Cup": 30,
    "Pacific Mini Games": 30,
    "Intercontinental Cup": 30,
    "Three Nations Cup": 30,
    "Mahinda Rajapaksa Cup": 30,
    "Navruz Cup": 30,
    "MSG Prime Minister's Cup": 30,
    "Jordan International Tournament": 30,
    "Tri Nation Tournament": 30,
    "Mauritius Four Nations Cup": 30,
    "Soccer Ashes": 30,
    "FIFA Series": 30,
    "Marianas Cup": 30,
    "Tri-Nations Series": 30,
    "Canadian Shield": 30,
    "Outrigger Challenge Cup": 30,
    "South Asian Super Cup": 30,
    "CONCACAF Series": 30,
    "Al Ain International Cup": 30
}

# === USER INPUT ===
column_name = "tournament"   # exact column header

# === SCRIPT ===
df = pd.read_csv(OUT_FILE)

'''

if column_name in df.columns:
    unique_values = df[column_name].dropna().astype(str).unique().tolist()
    print("Unique values:")
    print(unique_values)
else:
    print("Column not found.")

'''

df["tournament"] = df["tournament"].map(tournament_weights)

df["tournament"] = df["tournament"].fillna(30)

# Save new file
df.to_csv(OUT_FILE, index=False)

print("Tournament column successfully converted to weights.")