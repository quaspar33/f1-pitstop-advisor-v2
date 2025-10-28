from typing import List, Dict
import pandas as pd
import numpy as np
from fastf1.core import Session


def get_lap_data_with_weather(session: Session) -> pd.DataFrame:
    weather_data: pd.DataFrame = session.weather_data.copy()
    laps: pd.DataFrame = session.laps.copy()

    laps.dropna(subset=["LapTime"], ignore_index=True, inplace=True)

    weather_for_intervals = weather_data.loc[:, ["Time"]].copy()
    weather_for_intervals["EndTime"] = weather_for_intervals["Time"].shift(-1)
    weather_for_intervals.loc[weather_for_intervals.last_valid_index(), "EndTime"] = (
        weather_for_intervals.loc[weather_for_intervals.last_valid_index(), "Time"]
        + np.timedelta64(1, "m")
    )
    weather_interval_index = pd.IntervalIndex.from_arrays(
        weather_for_intervals["Time"], weather_for_intervals["EndTime"], closed="both"
    )
    weather_indexer, _ = weather_interval_index.get_indexer_non_unique(laps["Time"])
    weather_data["TmpJoinIndex"] = weather_data.index
    laps["TmpJoinIndex"] = pd.Series(weather_indexer)

    data = laps.merge(weather_data, on="TmpJoinIndex", suffixes=("", "_y"))

    data.drop(["TmpJoinIndex", "Time_y"], axis="columns", inplace=True)
    return data


def add_lap_time_seconds(data: pd.DataFrame, inplace: bool) -> pd.DataFrame | None:
    if not inplace:
        data = data.copy()
    data["LapTimeSeconds"] = data["LapTime"].apply(lambda t: t.total_seconds())
    if not inplace:
        return data
    else:
        return None


def add_z_score_for_laps(data: pd.DataFrame, inplace: bool) -> pd.DataFrame | None:
    if not inplace:
        data = data.copy()
    if "LapTimeSeconds" not in data.keys():
        add_lap_time_seconds(data, inplace=True)
        contained_lap_time_seconds = True
    else:
        contained_lap_time_seconds = False

    driver_codes = data["Driver"].unique()
    driver_df = pd.DataFrame(index=driver_codes)
    driver_df["MeanTime"] = None
    driver_df["StandardDeviation"] = None

    for driver in driver_codes:
        driver_times = data[data["Driver"] == driver]["LapTimeSeconds"]
        driver_df.loc[driver, "MeanTime"] = driver_times.mean()
        driver_df.loc[driver, "StandardDeviation"] = driver_times.std()

    data["LapTimeZScore"] = None
    average_stdev = driver_df["StandardDeviation"].mean()
    for idx in data.index:
        subset = data.loc[idx, ["LapTimeSeconds", "Driver"]]
        lap_time = subset["LapTimeSeconds"]
        driver = subset["Driver"]
        mean_time = driver_df.loc[driver, "MeanTime"]
        data.loc[idx, "LapTimeZScore"] = (lap_time - mean_time) / average_stdev

    if not contained_lap_time_seconds:
        data.drop("LapTimeSeconds", axis="columns", inplace=True)
    if not inplace:
        return data
    else:
        return None


def get_refined_lap_data_with_z_score(sessions: List[Session]) -> pd.DataFrame:
    if not sessions:
        raise ValueError("Parameter 'sessions' may not be an empty list.")

    data_list = []
    for session in sessions:
        session_data = get_lap_data_with_weather(session)
        add_z_score_for_laps(session_data, inplace=True)
        session_data = session_data.convert_dtypes()
        data_list.append(session_data)

    data = pd.concat(data_list, ignore_index=True)

    data["IsPitLap"] = ~np.isnat(data["PitInTime"])

    selected_columns = [
        "LapTimeZScore",
        "IsPitLap",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "LapNumber",
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "TrackTemp",
        "WindDirection",
        "WindSpeed",
    ]
    filtered_data = data.loc[:, selected_columns]

    final_data = pd.get_dummies(filtered_data)
    return final_data


def get_refined_lap_data_with_z_score_for_circuit(
    sessions: List[Session], circuit: str
) -> pd.DataFrame:
    selected_sessions = []
    for session in sessions:
        if session is not None:
            circuit_info = session.session_info["Meeting"]["Circuit"]
            if circuit == circuit_info["ShortName"]:
                selected_sessions.append(session)

    if not selected_sessions:
        raise KeyError(f"No sessions found for circuit: {circuit}")

    return get_refined_lap_data_with_z_score(selected_sessions)


def aggregate_laps_by_circuit(
    loaded_sessions: List[Session],
) -> Dict[str, pd.DataFrame]:
    print("Reaktywuję sesje po deserializacji...")

    for i, session in enumerate(loaded_sessions, 1):
        if session is not None:
            try:
                session.load()
                print(f"  ✓ Reaktywowano sesję {i}/{len(loaded_sessions)}")
            except Exception as e:
                print(f"  ✗ Błąd reaktywacji sesji {i}: {e}")

    print("=" * 60)

    circuits = set()
    for session in loaded_sessions:
        if session is not None:
            circuits.add(session.session_info["Meeting"]["Circuit"]["ShortName"])

    print(f"\nZnaleziono {len(circuits)} unikalnych torów")

    dfs = {}
    for circuit in circuits:
        print(f"\nPrzetwarzam dane dla toru: {circuit}")
        try:
            dfs[circuit] = get_refined_lap_data_with_z_score_for_circuit(
                loaded_sessions, circuit
            )
            print(
                f"  ✓ {circuit}: {dfs[circuit].shape[0]} okrążeń, {dfs[circuit].shape[1]} cech"
            )
        except Exception as e:
            print(f"  ✗ Błąd dla {circuit}: {e}")

    print(f"Przetworzono {len(dfs)} torów")

    return dfs
