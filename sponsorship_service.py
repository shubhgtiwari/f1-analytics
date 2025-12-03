import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st

# --- CONFIGURATION ---
DB_URI = st.secrets["DB_CONNECTION_URI"]

engine = create_engine(DB_URI)

# 1. Define Sponsor Mappings
SPONSOR_MAP = {
    "Red Bull Racing": ["Oracle", "Bybit"],
    "Ferrari": ["Santander", "Shell"],
    "Mercedes": ["Petronas", "Ineos"],
    "McLaren": ["Google Chrome", "OKX"],
    "Aston Martin": ["Aramco", "Cognizant"],
    "Alpine": ["BWT", "Castrol"],
    "Williams": ["Komatsu", "Gulf"],
    "RB": ["Visa", "Cash App"],
    "Kick Sauber": ["Stake", "Kick"],
    "Haas F1 Team": ["MoneyGram", "Chipotle"]
}

# 2. Define Media Value Weights
VISIBILITY_MULTIPLIERS = {
    1: 10.0, 
    2: 6.0,
    3: 4.0,
    4: 3.0,
    5: 2.5,
}

def calculate_media_value(race_id=1):
    """
    Sponsorship ROI Analytics
    Calculates the 'Earned Media Value' based on lap positions.
    """
    print(f"--- Calculating Sponsorship ROI for Race {race_id} ---")
    
    # 1. Get Lap Data
    query = text(f"""
        SELECT l.driver_code, d.team, l.position, l.lap_number
        FROM lap_times l
        JOIN drivers d ON l.driver_code = d.code
        WHERE l.race_id = {race_id}
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
    # 2. Calculate Value Per Lap
    BASE_LAP_VALUE = 5000 
    
    roi_data = []
    
    for team, sponsors in SPONSOR_MAP.items():
        team_laps = df[df['team'].str.contains(team, case=False, na=False)]
        
        total_value = 0
        
        for _, row in team_laps.iterrows():
            pos = row['position']
            multiplier = VISIBILITY_MULTIPLIERS.get(pos, 1.5 if pos <= 10 else 0.5)
            
            # ROI Formula: Base * Multiplier
            lap_value = BASE_LAP_VALUE * multiplier
            total_value += lap_value
            
        for sponsor in sponsors:
            roi_data.append({
                "Sponsor": sponsor,
                "Team": team,
                "Estimated Media Value ($)": round(total_value, 2)
            })
            
    return pd.DataFrame(roi_data).sort_values(by="Estimated Media Value ($)", ascending=False)

if __name__ == "__main__":
    df = calculate_media_value(1)
    print(df.head())