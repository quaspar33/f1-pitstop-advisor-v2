import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from itertools import product
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_compound_combinations_with_f1_rules(
    num_stints: int, circuit: str, compound_mapping: Dict
) -> List[List[int]]:
    """
    Generuje kombinacje mieszanek zgodnie z zasadami F1:
    - Nie można używać tej samej mieszanki przez cały wyścig
    - Używa wartości numerycznych specyficznych dla danego toru
    """
    if circuit in compound_mapping:
        compounds = sorted(list(compound_mapping[circuit].values()))
    else:
        compounds = [3, 2, 1]
        logger.warning(
            f"Brak mapowania mieszanek dla {circuit}, używam domyślnych: {compounds}"
        )

    all_combinations = list(product(compounds, repeat=num_stints))

    valid_combinations = [
        list(combo) for combo in all_combinations if len(set(combo)) > 1
    ]

    return valid_combinations


def translate_compound(
    compound_numeric: int, circuit: str, compound_mapping: Dict
) -> str:
    """
    Tłumaczy numeryczną wartość mieszanki (1, 2, 3) na nazwę kardynalną (SOFT, MEDIUM, HARD)
    na podstawie mapowania dla danego toru.
    """
    if circuit not in compound_mapping:
        default_map = {1: "SOFT", 2: "MEDIUM", 3: "HARD"}
        return default_map.get(compound_numeric, "UNKNOWN")

    circuit_map = compound_mapping[circuit]
    for compound_name, numeric_value in circuit_map.items():
        if numeric_value == compound_numeric:
            return compound_name.upper()

    return "UNKNOWN"


def reverse_translate_compounds(circuit: str, compound_mapping: Dict) -> Dict[str, int]:
    """
    Tworzy odwrotne mapowanie: nazwa kardynalna -> wartość numeryczna dla toru.
    """
    if circuit not in compound_mapping:
        return {"SOFT": 3, "MEDIUM": 2, "HARD": 1}

    return {k.upper(): v for k, v in compound_mapping[circuit].items()}


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

    if latest_stop < earliest_stop:
        logger.warning(f"Too few laps ({total_laps}) for valid strategies")
        return []

    possible_laps = list(range(earliest_stop, latest_stop + 1))

    for num_stops in range(min_stops, max_stops + 1):
        from itertools import combinations_with_replacement

        for strategy in combinations_with_replacement(possible_laps, num_stops):
            strategy_list = sorted(list(strategy))
            valid = True
            prev_stop = 0

            for stop in strategy_list:
                stint_length = stop - prev_stop
                if stint_length < min_stint_length or stint_length > max_stint_length:
                    valid = False
                    break
                prev_stop = stop

            if valid:
                last_stint_length = total_laps - strategy_list[-1]
                if (
                    last_stint_length < min_stint_length
                    or last_stint_length > max_stint_length
                ):
                    valid = False

            if valid and strategy_list not in strategies:
                strategies.append(strategy_list)

    logger.info(
        f"Generated {len(strategies)} valid pit stop strategies for {total_laps} laps"
    )
    return strategies


def prepare_lap_features(
    lap_number: int,
    stint_lap: int,
    compound: int,
    circuit_data: pd.DataFrame,
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
            "CompoundNumeric": [compound],
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
    compounds: List[int],
    total_laps: int,
    circuit_data: pd.DataFrame,
    model: Any,
    circuit: str = None,
    mean_pit_stop_dict: Dict[str, float] = None,
    pit_stop_std_dict: Dict[str, float] = None,
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
                "CompoundNumeric": compounds[current_stint],
                "PredictedZScore": predicted_zscore,
                "IsPitStop": lap in pit_stops,
                "PitStopTime": pit_time,
            }
        )

        stint_lap += 1

    results_df = pd.DataFrame(results)
    mean_zscore = results_df["PredictedZScore"].mean()
    total_pit_time = results_df["PitStopTime"].sum()

    return mean_zscore, results_df, total_pit_time


def optimize_strategy_for_circuit(
    circuit: str,
    circuit_data: pd.DataFrame,
    model: Any,
    race_laps: int,
    mean_pit_stop_dict: Dict[str, float] = None,
    pit_stop_std_dict: Dict[str, float] = None,
    compound_mapping: Dict = None,
    min_stops: int = 1,
    max_stops: int = 3,
    base_pitstop_time: float = 20.0,
    pitstop_std: float = 2.0,
) -> pd.DataFrame:
    logger.info(f"Przetwarzam {circuit}...")

    strategies = generate_strategies(race_laps, min_stops, max_stops)

    if not strategies:
        logger.warning(f"No valid strategies generated for {circuit}")
        return pd.DataFrame()

    mean_pit = mean_pit_stop_dict.get(circuit, base_pitstop_time)
    pit_std = pit_stop_std_dict.get(circuit, pitstop_std)

    logger.info(
        f"Generuję strategie dla {circuit} – Liczba okrążeń: {race_laps}, "
        f"Średnia strata na pitstop: {mean_pit:.2f}s, Odchylenie standardowe: {pit_std:.2f}s, "
        f"Mieszanki: {compound_mapping[circuit]}"
    )

    if compound_mapping is None:
        compound_mapping = {}

    strategy_results = []

    total_combinations = 0
    for strategy in strategies:
        num_stints = len(strategy) + 1
        compound_combinations = get_compound_combinations_with_f1_rules(
            num_stints=num_stints, circuit=circuit, compound_mapping=compound_mapping
        )
        total_combinations += len(compound_combinations)

    logger.info(
        f"Testowanie {total_combinations} kombinacji strategii i mieszanek dla {circuit}"
    )

    with tqdm(
        total=total_combinations,
        desc=f"  {circuit}",
        unit=" komb",
        ncols=100,
        leave=True,
    ) as pbar:
        for strategy in strategies:
            num_stints = len(strategy) + 1
            compound_combinations = get_compound_combinations_with_f1_rules(
                num_stints=num_stints,
                circuit=circuit,
                compound_mapping=compound_mapping,
            )

            for compounds in compound_combinations:
                try:
                    mean_zscore, sim_df, total_pit_time = simulate_strategy(
                        strategy=strategy,
                        compounds=compounds,
                        total_laps=race_laps,
                        circuit_data=circuit_data,
                        model=model,
                        circuit=circuit,
                        mean_pit_stop_dict=mean_pit_stop_dict,
                        pit_stop_std_dict=pit_stop_std_dict,
                        base_pitstop_time=base_pitstop_time,
                        pitstop_std=pitstop_std,
                    )

                    strategy_results.append(
                        {
                            "Circuit": circuit,
                            "Strategy": strategy,
                            "Compounds": compounds,
                            "MeanZScore": mean_zscore,
                            "TotalPitTime": total_pit_time,
                            "NumStops": len(strategy),
                            "Simulation": sim_df,
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Error simulating strategy {strategy} with compounds {compounds}: {e}"
                    )
                finally:
                    pbar.update(1)

    if not strategy_results:
        logger.warning(f"No successful simulations for {circuit}")
        return pd.DataFrame()

    results_df = pd.DataFrame(strategy_results)
    results_df = results_df.sort_values("MeanZScore").reset_index(drop=True)

    logger.info(
        f"Completed optimization for {circuit}: {len(results_df)} strategies evaluated"
    )
    return results_df


def optimize_all_circuits(
    models_dict: Dict,
    data_dict: Dict,
    params: Dict,
) -> pd.DataFrame:
    race_laps = params.get("race_laps", {})
    mean_pit_stop_dict = params.get("mean_pit_stop", {})
    pit_stop_std_dict = params.get("pit_stop_std", {})
    compound_mapping = params.get("compounds", {})

    min_stops = params.get("min_stops", 1)
    max_stops = params.get("max_stops", 3)
    base_pitstop_time = params.get("base_pitstop_time", 20.0)
    pitstop_std = params.get("pitstop_std", 2.0)

    logger.info(
        f"""Rozpoczynam optymalizację dla {len(models_dict)} torów z parametrami:
    min_stops={min_stops}
    max_stops={max_stops}
    top_n={params.get('top_n', 1)}
    air_temp={params.get('air_temp')}
    track_temp={params.get('track_temp')}
    humidity={params.get('humidity')}
    rainfall={params.get('rainfall')}
    pressure={params.get('pressure')}
    wind_speed={params.get('wind_speed')}
    wind_direction={params.get('wind_direction')}"""
    )

    joined = ",\n    ".join(models_dict.keys())
    logger.info(f"Tory do przetworzenia:\n    {joined}")

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
            compound_mapping=compound_mapping,
            min_stops=min_stops,
            max_stops=max_stops,
            base_pitstop_time=base_pitstop_time,
            pitstop_std=pitstop_std,
        )

        if not circuit_results.empty:
            all_results.append(circuit_results)

    if not all_results:
        logger.warning("No results generated for any circuit")
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def select_best_strategies(
    optimization_results: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    if optimization_results.empty:
        return pd.DataFrame()

    top_n = params.get("top_n", 1)
    best_strategies = []

    for circuit in optimization_results["Circuit"].unique():
        circuit_results = optimization_results[
            optimization_results["Circuit"] == circuit
        ]
        circuit_best = circuit_results.nsmallest(top_n, "MeanZScore")
        best_strategies.append(circuit_best)

    return pd.concat(best_strategies, ignore_index=True)


def save_optimization_results_csv(
    optimization_results: pd.DataFrame, params: Dict = None
) -> pd.DataFrame:
    if optimization_results.empty:
        return pd.DataFrame()

    results_csv = optimization_results.copy()

    compound_mapping = params.get("compounds", {}) if params else {}

    def format_compounds(row):
        circuit = row["Circuit"]
        compounds_numeric = row["Compounds"]
        if isinstance(compounds_numeric, list):
            compound_names = [
                translate_compound(c, circuit, compound_mapping)
                for c in compounds_numeric
            ]
            return str(compound_names)
        return str(compounds_numeric)

    results_csv["Strategy"] = results_csv["Strategy"].apply(lambda x: str(x))
    results_csv["CompoundsCardinal"] = results_csv.apply(format_compounds, axis=1)
    results_csv["CompoundsNumeric"] = results_csv["Compounds"].apply(lambda x: str(x))

    if "Simulation" in results_csv.columns:
        results_csv = results_csv.drop(columns=["Simulation", "Compounds"])

    return results_csv


def save_best_strategies_csv(
    best_strategies: pd.DataFrame, params: Dict = None
) -> pd.DataFrame:
    if best_strategies.empty:
        return pd.DataFrame()

    results_csv = best_strategies.copy()

    compound_mapping = params.get("compounds", {}) if params else {}

    def format_compounds(row):
        circuit = row["Circuit"]
        compounds_numeric = row["Compounds"]
        if isinstance(compounds_numeric, list):
            compound_names = [
                translate_compound(c, circuit, compound_mapping)
                for c in compounds_numeric
            ]
            return str(compound_names)
        return str(compounds_numeric)

    results_csv["Strategy"] = results_csv["Strategy"].apply(lambda x: str(x))
    results_csv["CompoundsCardinal"] = results_csv.apply(format_compounds, axis=1)
    results_csv["CompoundsNumeric"] = results_csv["Compounds"].apply(lambda x: str(x))

    if "Simulation" in results_csv.columns:
        results_csv = results_csv.drop(columns=["Simulation", "Compounds"])

    return results_csv


def generate_detailed_simulation(
    best_strategies: pd.DataFrame, models_dict: Dict, data_dict: Dict, params: Dict
) -> Dict[str, pd.DataFrame]:
    if best_strategies.empty:
        return {}

    detailed_simulations = {}

    mean_pit_stop_dict = params.get("mean_pit_stop", {})
    pit_stop_std_dict = params.get("pit_stop_std", {})
    compound_mapping = params.get("compounds", {})

    for circuit in best_strategies["Circuit"].unique():
        circuit_best = best_strategies[best_strategies["Circuit"] == circuit].iloc[0]
        best_strategy = circuit_best["Strategy"]
        compounds = circuit_best.get("Compounds", None)

        if compounds is not None:
            compounds = list(compounds)
        else:
            num_stints = len(best_strategy) + 1
            compounds = [2] * num_stints

        total_laps = (
            params["race_laps"].get(circuit, 56) if "race_laps" in params else 56
        )

        mean_pit = mean_pit_stop_dict.get(
            circuit, params.get("base_pitstop_time", 20.0)
        )
        pit_std = pit_stop_std_dict.get(circuit, params.get("pitstop_std", 2.0))

        logger.info(
            f"Generuję szczegółową symulację dla {circuit} – Średnia strata na pitstop: {mean_pit:.2f}s, "
            f"Odchylenie standardowe: {pit_std:.2f}s, Strategia: {best_strategy}, Mieszanki: {compounds}"
        )

        try:
            mean_zscore, detailed_sim, total_pit_time = simulate_strategy(
                strategy=best_strategy,
                compounds=compounds,
                total_laps=total_laps,
                circuit_data=data_dict[circuit],
                model=models_dict[circuit],
                circuit=circuit,
                mean_pit_stop_dict=mean_pit_stop_dict,
                pit_stop_std_dict=pit_stop_std_dict,
                air_temp=params.get("air_temp"),
                track_temp=params.get("track_temp"),
                humidity=params.get("humidity"),
                rainfall=params.get("rainfall", False),
                pressure=params.get("pressure"),
                wind_speed=params.get("wind_speed"),
                wind_direction=params.get("wind_direction"),
            )

            detailed_sim["CompoundCardinal"] = detailed_sim["CompoundNumeric"].apply(
                lambda x: translate_compound(x, circuit, compound_mapping)
            )
            detailed_sim["Circuit"] = circuit
            detailed_simulations[circuit] = detailed_sim

            logger.info(
                f"  Średni z-score: {mean_zscore:.6f}, Łączny czas pit stopów: {total_pit_time:.2f}s"
            )
        except Exception as e:
            logger.error(f"Error generating detailed simulation for {circuit}: {e}")
            continue

    return detailed_simulations


def save_detailed_simulations_csv(
    detailed_simulations: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    if not detailed_simulations:
        return pd.DataFrame()

    all_sims = []
    for circuit, sim_df in detailed_simulations.items():
        sim_copy = sim_df.copy()
        sim_copy["Circuit"] = circuit
        all_sims.append(sim_copy)

    return pd.concat(all_sims, ignore_index=True)


def visualize_strategies(
    detailed_simulations: Dict[str, pd.DataFrame], params: Dict = None
) -> Dict[str, plt.Figure]:
    figures = {}

    for circuit, simulation_df in detailed_simulations.items():
        if simulation_df.empty:
            logger.warning(f"Brak danych symulacji dla toru: {circuit}")
            continue

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

        ax1.plot(
            simulation_df["LapNumber"],
            simulation_df["PredictedZScore"],
            label="Przewidywany Z-score",
            color="blue",
            linewidth=2,
        )

        pit_laps = simulation_df[simulation_df["IsPitStop"]]["LapNumber"]
        for lap in pit_laps:
            ax1.axvline(
                lap,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Pit stop" if lap == pit_laps.iloc[0] else "",
            )

        ax1.set_xlabel("Okrążenie", fontsize=12)
        ax1.set_ylabel("Z-score", fontsize=12)
        ax1.set_title(
            f"{circuit} – Przebieg strategii pit stopów", fontsize=14, fontweight="bold"
        )
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        stints = (
            simulation_df.groupby("Stint")
            .agg({"LapNumber": ["min", "max"], "CompoundCardinal": "first"})
            .reset_index()
        )

        colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "white"}

        for _, stint in stints.iterrows():
            start_lap = stint[("LapNumber", "min")]
            end_lap = stint[("LapNumber", "max")]
            compound = stint[("CompoundCardinal", "first")]
            color = colors.get(compound, "gray")

            ax2.barh(
                0,
                end_lap - start_lap + 1,
                left=start_lap - 1,
                height=0.5,
                color=color,
                edgecolor="black",
                linewidth=1.5,
                label=(
                    compound
                    if compound
                    not in [
                        s[("CompoundCardinal", "first")]
                        for _, s in stints.iloc[: stint.name].iterrows()
                    ]
                    else ""
                ),
            )

            mid_lap = (start_lap + end_lap) / 2
            ax2.text(
                mid_lap,
                0,
                f"Stint {stint['Stint']}\n{compound}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        ax2.set_xlabel("Okrążenie", fontsize=12)
        ax2.set_ylabel("")
        ax2.set_yticks([])
        ax2.set_xlim(0, simulation_df["LapNumber"].max())
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3, axis="x")

        fig.tight_layout()
        figures[f"{circuit}.png"] = fig
        logger.info(f"📊 Wykres utworzony dla: {circuit}")

    return figures
