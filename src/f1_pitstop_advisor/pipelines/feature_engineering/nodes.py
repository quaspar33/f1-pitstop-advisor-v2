from typing import List, Dict, Optional
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


def load_compound_translation_map(compounds_map: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["year", "gp", "hard", "medium", "soft"]
    missing_cols = set(required_cols) - set(compounds_map.columns)
    if missing_cols:
        raise ValueError(f"Brakuje kolumn w compounds_map: {missing_cols}")
    return compounds_map


def convert_compounds_and_filter_rain(
    session_data: pd.DataFrame,
    translation_map: pd.DataFrame,
    session: Session,
    remove_if_no_mapping: bool = True,
) -> Optional[pd.DataFrame]:
    event_name = session.event["EventName"]
    year = session.event["EventDate"].year

    print(f"  🔹 Przetwarzanie sesji {event_name} {year}")

    if "Rainfall" in session_data.columns:
        rain_filtered = session_data[~session_data["Rainfall"]].copy()
        if len(rain_filtered) != len(session_data):
            print(
                f"    ⚠️ Usunięto {len(session_data) - len(rain_filtered)} wierszy z powodu deszczu"
            )
        session_data = rain_filtered
        if len(session_data) == 0:
            print("    ⚠️ Usunięto sesję - padało przez cały wyścig")
            return None

    if "Compound" not in session_data.columns:
        print(f"    ⚠️ Brak kolumny 'Compound' w sesji {event_name} {year}")
        return session_data

    rain_compounds = ["INTERMEDIATE", "WET"]
    non_rain_data = session_data[~session_data["Compound"].isin(rain_compounds)].copy()
    if len(non_rain_data) != len(session_data):
        print(
            f"    ⚠️ Usunięto {len(session_data) - len(non_rain_data)} wierszy opon deszczowych"
        )
    session_data = non_rain_data
    if len(session_data) == 0:
        print("    ⚠️ Usunięto sesję - tylko opony deszczowe")
        return None

    mapping_row = translation_map[
        (translation_map["year"] == year) & (translation_map["gp"] == event_name)
    ]
    if mapping_row.empty:
        if remove_if_no_mapping:
            print(f"    ⚠️ Brak mapowania dla: {event_name} {year} - usuwam sesję")
            return None
        else:
            print(
                f"    ⚠️ Brak mapowania dla: {event_name} {year} - pozostawiam bez zmiany"
            )
            return session_data

    mapping_dict = {
        "HARD": mapping_row.iloc[0]["hard"],
        "MEDIUM": mapping_row.iloc[0]["medium"],
        "SOFT": mapping_row.iloc[0]["soft"],
    }

    session_data["RealCompound"] = session_data["Compound"].map(mapping_dict)
    unmapped = session_data["RealCompound"].isna().sum()
    if unmapped > 0:
        print(f"    ⚠️ {unmapped} wierszy nie udało się zmapować na RealCompound")

    session_data = session_data.dropna(subset=["RealCompound"]).copy()
    if len(session_data) == 0:
        print("    ⚠️ Usunięto sesję - brak zmapowanych compound")
        return None

    session_data["CompoundNumeric"] = (
        session_data["RealCompound"].str.extract(r"C(\d+)").astype(int)
    )
    session_data["Compound"] = session_data["RealCompound"]
    session_data.drop(["RealCompound"], axis=1, inplace=True)

    print(f"    ✅ Sesja przetworzona: {len(session_data)} wierszy pozostało")
    return session_data


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

    return filtered_data


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
    compounds_map: pd.DataFrame,
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

            for s in loaded_sessions:
                if s and s.session_info["Meeting"]["Circuit"]["ShortName"] == circuit:
                    dfs[circuit] = convert_compounds_and_filter_rain(
                        dfs[circuit],
                        translation_map=compounds_map,
                        session=s,
                        remove_if_no_mapping=True,
                    )
                    break

            if dfs[circuit] is not None:
                print(
                    f"  ✓ {circuit}: {dfs[circuit].shape[0]} okrążeń, {dfs[circuit].shape[1]} cech"
                )
        except Exception as e:
            print(f"  ✗ Błąd dla {circuit}: {e}")

    print(f"Przetworzono {len(dfs)} torów")
    return dfs
