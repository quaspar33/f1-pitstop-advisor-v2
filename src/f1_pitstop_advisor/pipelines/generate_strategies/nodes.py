import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from itertools import combinations_with_replacement, product
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_compound_combinations(num_stints: int, compound_mode: str = "numeric") -> List:
    if compound_mode == "numeric":
        compounds = [1, 2, 3, 4, 5]
    elif compound_mode == "categorical":
        compounds = ["SOFT", "MEDIUM", "HARD"]
    else:
        raise ValueError(f"Unknown compound_mode: {compound_mode}")
    return list(product(compounds, repeat=num_stints))


def get_race_laps(circuit: str, race_laps: dict) -> int:
    if circuit not in race_laps:
        logger.warning(
            f"Brak danych o liczbie okrążeń dla {circuit}, używam domyślnie 56"
        )
        return 56
    return race_laps[circuit]


def generate_strategies(
    total_laps: int, min_stops: int = 1, max_stops: int = 3
) -> List[List[int]]:
    strategies = []
    min_stint_length = 8
    max_stint_length = 35
    earliest_stop = 8
    latest_stop = total_laps - 8
    possible_laps = list(range(earliest_stop, latest_stop + 1))

    for num_stops in range(min_stops, max_stops + 1):
        for strategy in combinations_with_replacement(possible_laps, num_stops):
            strategy_list = sorted(list(strategy))
            valid = True
            prev_stop = 0
            for stop in strategy_list:
                if (
                    stop - prev_stop < min_stint_length
                    or stop - prev_stop > max_stint_length
                ):
                    valid = False
                    break
                prev_stop = stop
            if valid and (
                total_laps - strategy_list[-1] < min_stint_length
                or total_laps - strategy_list[-1] > max_stint_length
            ):
                valid = False
            if valid and strategy_list not in strategies:
                strategies.append(strategy_list)

    logger.info(f"Generated {len(strategies)} valid strategies for {total_laps} laps")
    return strategies


def prepare_lap_features(
    lap_number: int,
    stint_lap: int,
    compound: Any,
    circuit_data: pd.DataFrame,
    compound_mode: str = "numeric",
    air_temp: float = None,
    track_temp: float = None,
    humidity: float = None,
    rainfall: bool = False,
    pressure: float = None,
    wind_speed: float = None,
    wind_direction: float = None,
) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "LapNumber": [lap_number],
            "TyreLife": [stint_lap],
            "FreshTyre": [1 if stint_lap == 1 else 0],
            "IsPitLap": [0],
        }
    )

    features["AirTemp"] = (
        air_temp if air_temp is not None else circuit_data["AirTemp"].median()
    )
    features["TrackTemp"] = (
        track_temp if track_temp is not None else circuit_data["TrackTemp"].median()
    )
    features["Humidity"] = (
        humidity if humidity is not None else circuit_data["Humidity"].median()
    )
    features["Pressure"] = (
        pressure if pressure is not None else circuit_data["Pressure"].median()
    )
    features["Rainfall"] = 1 if rainfall else 0
    features["WindSpeed"] = (
        wind_speed if wind_speed is not None else circuit_data["WindSpeed"].median()
    )
    features["WindDirection"] = (
        wind_direction
        if wind_direction is not None
        else circuit_data["WindDirection"].median()
    )

    if compound_mode == "numeric":
        features["CompoundNumeric"] = compound
    elif compound_mode == "categorical":
        features["Compound_HARD"] = 1 if compound == "HARD" else 0
        features["Compound_MEDIUM"] = 1 if compound == "MEDIUM" else 0
        features["Compound_SOFT"] = 1 if compound == "SOFT" else 0
    else:
        raise ValueError(f"Unknown compound_mode: {compound_mode}")

    excluded_columns = ["LapTimeZScore", "Compound"]
    required_cols = [col for col in circuit_data.columns if col not in excluded_columns]

    for col in required_cols:
        if col not in features.columns:
            if circuit_data[col].dtype in ["int64", "float64"]:
                features[col] = circuit_data[col].median()
            else:
                features[col] = (
                    circuit_data[col].mode()[0]
                    if len(circuit_data[col].mode()) > 0
                    else 0
                )

    result_features = features[required_cols].copy()

    for col in result_features.columns:
        if result_features[col].dtype == "object":
            try:
                result_features[col] = pd.to_numeric(
                    result_features[col], errors="coerce"
                )
            except Exception:
                logger.warning(
                    f"Nie można przekonwertować kolumny {col} na numeryczną, usuwam ją"
                )
                result_features = result_features.drop(columns=[col])

    return result_features


def simulate_strategy(
    strategy: List[int],
    total_laps: int,
    circuit_data: pd.DataFrame,
    model: Any,
    circuit: str = None,
    mean_pit_stop_dict: Dict[str, float] = None,
    pit_stop_std_dict: Dict[str, float] = None,
    compounds: List[Any] = None,
    compound_mode: str = "numeric",
    base_pitstop_time: float = 20.0,
    pitstop_std: float = 2.0,
    air_temp: float = None,
    track_temp: float = None,
    humidity: float = None,
    rainfall: bool = False,
    pressure: float = None,
    wind_speed: float = None,
    wind_direction: float = None,
) -> Tuple[float, pd.DataFrame]:
    num_stints = len(strategy) + 1
    if compounds is None:
        compounds = (
            [3] * num_stints if compound_mode == "numeric" else ["MEDIUM"] * num_stints
        )
    if len(compounds) != num_stints:
        raise ValueError(
            f"Liczba mieszanek ({len(compounds)}) != liczba stintów ({num_stints})"
        )

    if circuit and mean_pit_stop_dict and circuit in mean_pit_stop_dict:
        circuit_pit_time = mean_pit_stop_dict[circuit]
    else:
        circuit_pit_time = base_pitstop_time
        if circuit and mean_pit_stop_dict:
            logger.warning(
                f"Brak danych pit stop dla {circuit}, używam domyślnego: {circuit_pit_time:.2f}s"
            )

    if circuit and pit_stop_std_dict and circuit in pit_stop_std_dict:
        circuit_pit_std = pit_stop_std_dict[circuit]
    else:
        circuit_pit_std = pitstop_std

    pit_stops = set(strategy)
    results = []

    current_stint = 0
    stint_lap = 1

    for lap in range(1, total_laps + 1):
        pit_time = 0
        if lap in pit_stops:
            pit_time = max(
                0, np.random.normal(loc=circuit_pit_time, scale=circuit_pit_std)
            )
            current_stint += 1
            stint_lap = 1

        lap_features = prepare_lap_features(
            lap_number=lap,
            stint_lap=stint_lap,
            compound=compounds[current_stint],
            circuit_data=circuit_data,
            compound_mode=compound_mode,
            air_temp=air_temp,
            track_temp=track_temp,
            humidity=humidity,
            rainfall=rainfall,
            pressure=pressure,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
        )

        predicted_zscore = model.predict(lap_features)[0]

        results.append(
            {
                "LapNumber": lap,
                "Stint": current_stint + 1,
                "StintLap": stint_lap,
                "TyreLife": stint_lap,
                "Compound": compounds[current_stint],
                "PredictedZScore": predicted_zscore,
                "IsPitStop": lap in pit_stops,
                "PitStopTime": pit_time,
            }
        )

        stint_lap += 1

    results_df = pd.DataFrame(results)
    mean_zscore = results_df["PredictedZScore"].mean()
    return mean_zscore, results_df


def optimize_strategy_for_circuit(
    circuit: str,
    circuit_data: pd.DataFrame,
    model: Any,
    race_laps: int,
    mean_pit_stop_dict: Dict[str, float] = None,
    pit_stop_std_dict: Dict[str, float] = None,
    min_stops: int = 1,
    max_stops: int = 3,
    compound_mode: str = "numeric",
    base_pitstop_time: float = 20.0,
    pitstop_std: float = 2.0,
) -> pd.DataFrame:
    logger.info(f"Przetwarzam {circuit}...")

    strategies = generate_strategies(race_laps, min_stops, max_stops)

    mean_pit = mean_pit_stop_dict.get(circuit, base_pitstop_time)
    pit_std = pit_stop_std_dict.get(circuit, pitstop_std)

    logger.info(
        f"Generuję strategie dla {circuit} – Liczba okrążeń: {race_laps}, "
        f"Średnia strata na pitstop: {mean_pit:.2f}s, Odchylenie standardowe: {pit_std:.2f}s"
    )

    strategy_results = []

    for strategy in strategies:
        mean_zscore, sim_df = simulate_strategy(
            strategy=strategy,
            total_laps=race_laps,
            circuit_data=circuit_data,
            model=model,
            circuit=circuit,
            mean_pit_stop_dict=mean_pit_stop_dict,
            pit_stop_std_dict=pit_stop_std_dict,
            compounds=None,
            compound_mode=compound_mode,
            base_pitstop_time=base_pitstop_time,
            pitstop_std=pitstop_std,
        )
        strategy_results.append(
            {
                "Circuit": circuit,
                "Strategy": strategy,
                "MeanZScore": mean_zscore,
                "Simulation": sim_df,
            }
        )

    results_df = pd.DataFrame(strategy_results)
    results_df = results_df.sort_values("MeanZScore").reset_index(drop=True)
    return results_df


def optimize_all_circuits(
    models_dict: Dict,
    data_dict: Dict,
    params: Dict,
) -> pd.DataFrame:
    race_laps = params.get("race_laps", {})
    mean_pit_stop_dict = params.get("mean_pit_stop", {})
    pit_stop_std_dict = params.get("pit_stop_std", {})

    compound_mode = params.get("compound_mode", "numeric")
    min_stops = params.get("min_stops", 1)
    max_stops = params.get("max_stops", 3)
    base_pitstop_time = params.get("base_pitstop_time", 20.0)
    pitstop_std = params.get("pitstop_std", 2.0)

    all_results = []

    for circuit in models_dict.keys():
        total_laps = race_laps.get(circuit, 56)
        circuit_results = optimize_strategy_for_circuit(
            circuit=circuit,
            circuit_data=data_dict[circuit],
            model=models_dict[circuit],
            race_laps=total_laps,
            mean_pit_stop_dict=mean_pit_stop_dict,
            pit_stop_std_dict=pit_stop_std_dict,
            min_stops=min_stops,
            max_stops=max_stops,
            compound_mode=compound_mode,
            base_pitstop_time=base_pitstop_time,
            pitstop_std=pitstop_std,
        )
        all_results.append(circuit_results)

    return pd.concat(all_results, ignore_index=True)


def select_best_strategies(
    optimization_results: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    top_n = params.get("top_n", 1)
    best_strategies = []

    for circuit in optimization_results["Circuit"].unique():
        circuit_results = optimization_results[
            optimization_results["Circuit"] == circuit
        ]
        circuit_best = circuit_results.nsmallest(top_n, "MeanZScore")
        best_strategies.append(circuit_best)

    return pd.concat(best_strategies, ignore_index=True)


def save_optimization_results_csv(optimization_results: pd.DataFrame) -> pd.DataFrame:
    results_csv = optimization_results.copy()
    results_csv["Strategy"] = results_csv["Strategy"].apply(lambda x: str(x))
    if "Simulation" in results_csv.columns:
        results_csv = results_csv.drop(columns=["Simulation"])
    return results_csv


def save_best_strategies_csv(best_strategies: pd.DataFrame) -> pd.DataFrame:
    results_csv = best_strategies.copy()
    results_csv["Strategy"] = results_csv["Strategy"].apply(lambda x: str(x))
    if "Simulation" in results_csv.columns:
        results_csv = results_csv.drop(columns=["Simulation"])
    results_csv["NumStops"] = results_csv["Strategy"].apply(lambda x: len(eval(x)))
    return results_csv


def generate_detailed_simulation(
    best_strategies: pd.DataFrame, models_dict: Dict, data_dict: Dict, params: Dict
) -> Dict[str, pd.DataFrame]:
    detailed_simulations = {}
    compound_mode = params.get("compound_mode", "numeric")

    mean_pit_stop_dict = params.get("mean_pit_stop", {})
    pit_stop_std_dict = params.get("pit_stop_std", {})

    for circuit in best_strategies["Circuit"].unique():
        circuit_best = best_strategies[best_strategies["Circuit"] == circuit].iloc[0]
        best_strategy = circuit_best["Strategy"]
        compounds = circuit_best.get("Compounds", None)
        if compounds is not None:
            compounds = list(compounds)

        total_laps = (
            params["race_laps"].get(circuit, 56) if "race_laps" in params else 56
        )

        mean_pit = mean_pit_stop_dict.get(
            circuit, params.get("base_pitstop_time", 20.0)
        )
        pit_std = pit_stop_std_dict.get(circuit, params.get("pitstop_std", 2.0))

        logger.info(
            f"Generuję szczegółową symulację dla {circuit} – Średnia strata na pitstop: {mean_pit:.2f}s, "
            f"Odchylenie standardowe: {pit_std:.2f}s, Strategia: {best_strategy}"
        )

        mean_zscore, detailed_sim = simulate_strategy(
            strategy=best_strategy,
            total_laps=total_laps,
            circuit_data=data_dict[circuit],
            model=models_dict[circuit],
            circuit=circuit,
            mean_pit_stop_dict=mean_pit_stop_dict,
            pit_stop_std_dict=pit_stop_std_dict,
            compounds=compounds,
            compound_mode=compound_mode,
            air_temp=params.get("air_temp"),
            track_temp=params.get("track_temp"),
            humidity=params.get("humidity"),
            rainfall=params.get("rainfall", False),
            pressure=params.get("pressure"),
            wind_speed=params.get("wind_speed"),
            wind_direction=params.get("wind_direction"),
        )

        detailed_sim["Circuit"] = circuit
        detailed_simulations[circuit] = detailed_sim

        logger.info(f"  Średni z-score: {mean_zscore:.6f}")

    return detailed_simulations


def save_detailed_simulations_csv(
    detailed_simulations: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    all_sims = []
    for circuit, sim_df in detailed_simulations.items():
        sim_copy = sim_df.copy()
        sim_copy["Circuit"] = circuit
        all_sims.append(sim_copy)
    return pd.concat(all_sims, ignore_index=True)


def visualize_strategies(
    detailed_simulations: Dict[str, pd.DataFrame]
) -> Dict[str, plt.Figure]:
    figures = {}

    for circuit, simulation_df in detailed_simulations.items():
        if simulation_df.empty:
            logger.warning(f"Brak danych symulacji dla toru: {circuit}")
            continue

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            simulation_df["LapNumber"],
            simulation_df["PredictedZScore"],
            label="Przewidywany Z-score",
            color="blue",
            linewidth=2,
        )

        pit_laps = simulation_df[simulation_df["IsPitStop"]]["LapNumber"]
        for lap in pit_laps:
            ax.axvline(
                lap,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Pit stop" if lap == pit_laps.iloc[0] else "",
            )

        ax.set_xlabel("Okrążenie", fontsize=12)
        ax.set_ylabel("Z-score", fontsize=12)
        ax.set_title(
            f"{circuit} – Przebieg strategii pit stopów", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        figures[f"{circuit}.png"] = fig
        logger.info(f"📊 Wykres utworzony dla: {circuit}")

    return figures
