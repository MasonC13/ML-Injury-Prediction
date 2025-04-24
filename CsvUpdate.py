import pandas as pd
import numpy as np
import random

# Load your existing data
try:
    original_injury_df = pd.read_csv("InjuryRecord.csv")
    original_playlist_df = pd.read_csv("PlayList.csv")
    
    print("Original counts:")
    print(f"InjuryRecord.csv: {len(original_injury_df)} records")
    print(f"PlayList.csv: {len(original_playlist_df)} records")
    print(f"Injury count: {original_injury_df['DM_M1'].sum()} injuries")
    
    # How many synthetic injuries to add
    num_synthetic = 200
    
    # Generate synthetic injury data
    synthetic_injuries = []
    synthetic_playlists = []
    
    # Body parts that commonly get injured
    body_parts = ["Knee", "Ankle", "Hamstring", "Shoulder", "Head", "Foot", "Hip", "Elbow", "Hand", "Back"]
    surfaces = ["Turf", "Grass"]
    
    # Position groups with contextual awareness
    high_contact_positions = ["OL", "DL", "RB", "LB", "TE", "FB"]
    high_speed_positions = ["WR", "DB", "CB", "S", "FS", "SS"]
    other_positions = ["QB", "K", "P", "LS"]
    
    # Position groups mapping
    position_groups = {
        "OL": "Offensive Line", "DL": "Defensive Line",
        "RB": "Running Back", "LB": "Linebacker",
        "TE": "Tight End", "FB": "Fullback", 
        "WR": "Wide Receiver", "DB": "Defensive Back",
        "CB": "Cornerback", "S": "Safety", "FS": "Free Safety", "SS": "Strong Safety",
        "QB": "Quarterback", "K": "Kicker", "P": "Punter", "LS": "Long Snapper"
    }
    
    # Stadium types and weather conditions
    stadium_types = ["Indoor", "Outdoor"]
    weather_conditions = ["Clear", "Rain", "Snow", "Cloudy", "Windy", "Hot", "Cold"]
    play_types = ["Run", "Pass", "Punt", "Field Goal", "Kickoff"]
    
    # Generate unique PlayKeys that don't exist in original data
    existing_playkeys = set(original_injury_df["PlayKey"].tolist())
    next_player_key = original_injury_df["PlayerKey"].max() + 1 if len(original_injury_df) > 0 else 10000
    
    for i in range(num_synthetic):
        # Create a unique play key
        play_key = f"SYN_{i+1000}"
        while play_key in existing_playkeys:
            play_key = f"SYN_{random.randint(1000, 9999)}"
        
        existing_playkeys.add(play_key)
        
        # Create player key and game ID
        player_key = next_player_key + i
        game_id = f"G{random.randint(100, 999)}"
        
        # Create injury record
        body_part = random.choice(body_parts)
        surface = random.choice(surfaces)
        
        # All synthetic data will be injuries
        dm_m1 = 1
        dm_m7 = 1 if random.random() < 0.7 else 0
        dm_m28 = 1 if random.random() < 0.4 and dm_m7 == 1 else 0
        dm_m42 = 1 if random.random() < 0.2 and dm_m28 == 1 else 0
        
        # Add injury record
        synthetic_injuries.append({
            "PlayerKey": player_key,
            "GameID": game_id,
            "PlayKey": play_key,
            "BodyPart": body_part,
            "Surface": surface,
            "DM_M1": dm_m1,
            "DM_M7": dm_m7,
            "DM_M28": dm_m28,
            "DM_M42": dm_m42
        })
        
        # Create corresponding playlist record with strategic position-play type correlations
        if random.random() < 0.6:  # 60% chance to use injury patterns
            if random.random() < 0.5:  # Run plays with high contact positions
                position = random.choice(high_contact_positions)
                play_type = "Run"
            else:  # Pass plays with high speed positions
                position = random.choice(high_speed_positions)
                play_type = "Pass"
        else:
            # Random combinations for the rest
            position = random.choice(high_contact_positions + high_speed_positions + other_positions)
            play_type = random.choice(play_types)
        
        # Get position group
        position_group = position_groups.get(position, "Other")
        
        # Temperature depends on whether it's indoor/outdoor
        stadium_type = random.choice(stadium_types)
        if stadium_type == "Indoor":
            temperature = random.randint(68, 75)  # Controlled environment
            weather = "Clear"
        else:
            temperature = random.randint(25, 95)  # More variation
            weather = random.choice(weather_conditions)
        
        # Field type matches surface in injury data
        field_type = surface
        
        # Player day, game, and play numbers
        player_day = random.randint(1, 20)
        player_game = random.randint(1, 17)
        player_game_play = random.randint(1, 70)
        
        # Add playlist record
        synthetic_playlists.append({
            "PlayerKey": player_key,
            "GameID": game_id,
            "PlayKey": play_key,
            "RosterPosition": position,
            "PlayerDay": player_day,
            "PlayerGame": player_game,
            "StadiumType": stadium_type,
            "FieldType": field_type,
            "Temperature": temperature,
            "Weather": weather,
            "PlayType": play_type,
            "PlayerGamePlay": player_game_play,
            "Position": position,
            "PositionGroup": position_group
        })
    
    # Convert to DataFrames
    synthetic_injury_df = pd.DataFrame(synthetic_injuries)
    synthetic_playlist_df = pd.DataFrame(synthetic_playlists)
    
    # Combine with original data
    combined_injury_df = pd.concat([original_injury_df, synthetic_injury_df], ignore_index=True)
    combined_playlist_df = pd.concat([original_playlist_df, synthetic_playlist_df], ignore_index=True)
    
    # Save to NEW files to avoid permission issues
    combined_injury_df.to_csv("InjuryRecord_Enhanced.csv", index=False)
    combined_playlist_df.to_csv("PlayList_Enhanced.csv", index=False)
    
    print("\nCreated new files with enhanced data:")
    print(f"InjuryRecord_Enhanced.csv: {len(combined_injury_df)} records")
    print(f"PlayList_Enhanced.csv: {len(combined_playlist_df)} records")
    print(f"Injury count: {combined_injury_df['DM_M1'].sum()} injuries")
    print(f"Added {num_synthetic} synthetic injuries")
    print("\nIMPORTANT: Update your model code to use these '_Enhanced.csv' files")
    
except Exception as e:
    print(f"Error: {e}")
    print("Please check that your CSV files exist and are accessible.")