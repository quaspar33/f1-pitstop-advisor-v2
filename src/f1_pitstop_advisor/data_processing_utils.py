from typing import Iterable, List
import pandas as pd
import numpy as np
import pickle

from fastf1.core import Session


def get_lap_data_with_weather(session: Session) -> pd.DataFrame:
    # Prepare raw data
    weather_data: pd.DataFrame = session.weather_data.copy()  # type: ignore
    laps: pd.DataFrame = session.laps.copy()

    # Drop laps with missing lap time
    laps.dropna(subset=["LapTime"], ignore_index=True, inplace=True)

    # Prepare an indexer that indexes weather data for every lap
    weather_for_intervals = weather_data.loc[:, ["Time"]].copy()
    weather_for_intervals["EndTime"] = weather_for_intervals["Time"].shift(-1)
    weather_for_intervals.loc[weather_for_intervals.last_valid_index(), "EndTime"] = (
            weather_for_intervals.loc[weather_for_intervals.last_valid_index(), "Time"] + np.timedelta64(1, "m")
    # type: ignore
    )
    weather_interval_index = pd.IntervalIndex.from_arrays(weather_for_intervals["Time"],
                                                          weather_for_intervals["EndTime"], closed="both")
    weather_indexer, _ = weather_interval_index.get_indexer_non_unique(laps["Time"])  # type: ignore
    weather_data["TmpJoinIndex"] = weather_data.index
    laps["TmpJoinIndex"] = pd.Series(weather_indexer)

    # Merge laps with weather data
    data = laps.merge(weather_data, on="TmpJoinIndex", suffixes=("", "_y"))

    data.drop(["TmpJoinIndex", "Time_y"], axis="columns", inplace=True)
    return data


def add_lap_time_seconds(data: pd.DataFrame, inplace: bool) -> pd.DataFrame | None:
    if not inplace:
        data = data.copy()
    # Add a column representing lap time in seconds
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

    # Create a DataFrame with drivers as rows and mean/standard deviation of their lap times for columns
    driver_codes = data["Driver"].unique()
    driver_df = pd.DataFrame(index=driver_codes)
    driver_df["MeanTime"] = None
    driver_df["StandardDeviation"] = None

    for driver in driver_codes:
        driver_times = data[data["Driver"] == driver]["LapTimeSeconds"]
        driver_df.loc[driver, "MeanTime"] = driver_times.mean()
        driver_df.loc[driver, "StandardDeviation"] = driver_times.std()
        # Calculate Z-score for each lap. This score will tell us how good or bad a lap is 
        # relative to the rest of the same driver's laps. Lower is better.

    data["LapTimeZScore"] = None
    average_stdev = driver_df["StandardDeviation"].mean()
    for idx in data.index:
        subset = data.loc[idx, ["LapTimeSeconds", "Driver"]]
        # Lap time for the current lap
        lap_time = subset["LapTimeSeconds"]
        driver = subset["Driver"]
        # Mean lap time for the driver who took the current lap
        mean_time = driver_df.loc[driver, "MeanTime"]
        # Add Z-score
        data.loc[idx, "LapTimeZScore"] = (lap_time - mean_time) / average_stdev

    if not contained_lap_time_seconds:
        data.drop("LapTimeSeconds", axis="columns", inplace=True)
    if not inplace:
        return data
    else:
        return None


def get_refined_lap_data_with_z_score(sessions: List[Session]) -> pd.DataFrame:
    if not sessions:
        raise ValueError(f"Parameter \"sessions\" may not be an empty list.")
    data_list = []
    for session in sessions:
        session_data = get_lap_data_with_weather(session)
        add_z_score_for_laps(session_data, inplace=True)
        session_data = session_data.convert_dtypes()
        data_list.append(session_data)

    data = pd.concat(data_list, ignore_index=True)

    # Add a feature determining whether there was a pit stop performed during each lap
    data["IsPitLap"] = ~np.isnat(data["PitInTime"])

    # Select only relevant columns for further processing
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
        "WindSpeed"
    ]
    filtered_data = data.loc[:, selected_columns]

    # Convert categorical data to boolean values
    final_data = pd.get_dummies(filtered_data)
    return final_data


def get_refined_lap_data_with_z_score_for_circuit(sessions: List[Session], circuit: int | str) -> pd.DataFrame:
    selected_sessions = []
    for session in sessions:
        circuit_info = session.session_info["Meeting"]["Circuit"]
        if isinstance(circuit, int) and circuit == circuit_info["Key"]:
            session_matches_circuit = True
        elif isinstance(circuit, str) and circuit == circuit_info["ShortName"]:
            session_matches_circuit = True
        else:
            session_matches_circuit = False
        if session_matches_circuit:
            selected_sessions.append(session)

    try:
        return get_refined_lap_data_with_z_score(selected_sessions)
    except ValueError:
        raise KeyError("No sessions found for given circuit.")
