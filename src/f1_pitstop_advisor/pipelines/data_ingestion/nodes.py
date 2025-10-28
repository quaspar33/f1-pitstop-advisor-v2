import warnings
from datetime import datetime
from typing import List
import pandas as pd
import fastf1 as f1


def get_sessions(cutoff_date: str, start_year: int = 2022) -> List:
    warnings.filterwarnings('ignore')
    
    cutoff = datetime.strptime(cutoff_date, '%Y-%m-%d')
    sessions = []
    
    end_year = cutoff.year
    years = range(start_year, end_year + 1)
    
    for year in years:
        print(f"\nAnalizuję rok {year}...")
        
        try:
            race_calendar = f1.get_event_schedule(year)
            races = race_calendar[race_calendar['EventFormat'] == 'conventional']
            
            cutoff_timestamp = pd.Timestamp(cutoff)
            races = races[races['EventDate'] < cutoff_timestamp]
            
            if races.empty:
                print(f"Brak wyścigów dla roku {year}")
                continue
            
            print(f"Znaleziono {len(races)} wyścigów dla roku {year}")
            
            for idx, race in races.iterrows():
                race_name = race['EventName']
                race_round = race['RoundNumber']
                
                try:
                    print(f"  Pobieram dane dla wyścigu: {race_name} (Runda {race_round})")
                    session = f1.get_session(year, race_round, 'R')
                    sessions.append(session)
                except Exception as e:
                    print(f"    Błąd podczas pobierania danych dla {race_name}: {e}")
        except Exception as e:
            print(f"Błąd podczas pobierania kalendarza dla roku {year}: {e}")
    
    return sessions


def load_sessions_data(sessions: List) -> List:
    loaded_sessions = []
    total = len(sessions)
    
    for i, session in enumerate(sessions, 1):
        try:
            session.load()
            print(f"Załadowano sesję {i} z {total}")
            loaded_sessions.append(session)
        except RuntimeError as e:
            print(f"Błąd ładowania sesji {i} z {total}: {e}")
            loaded_sessions.append(None)
    
    return loaded_sessions
