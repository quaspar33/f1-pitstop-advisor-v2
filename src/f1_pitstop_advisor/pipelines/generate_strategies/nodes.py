import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product, combinations_with_replacement

logger = logging.getLogger(__name__)


def get_compound_combinations_with_f1_rules(
    num_stints: int, circuit: str, compound_mapping: Dict
) -> List[List[int]]:
    if circuit in compound_mapping:
        compounds = sorted(list(compound_mapping[circuit].values()))
    else:
        compounds = [3, 2, 1]

    all_combinations = list(product(compounds, repeat=num_stints))
    return [list(combo) for combo in all_combinations if len(set(combo)) > 1]


def translate_compound(
    compound_numeric: int, circuit: str, compound_mapping: Dict
) -> str:
    if circuit not in compound_mapping:
        return {1: "SOFT", 2: "MEDIUM", 3: "HARD"}.get(compound_numeric, "UNKNOWN")
    for name, val in compound_mapping[circuit].items():
        if val == compound_numeric:
            return name.upper()
    return "UNKNOWN"


def generate_strategies(
    total_laps: int, min_stops: int = 1, max_stops: int = 3, step: int = 2
) -> List[List[int]]:
    """
    Generuje możliwe momenty zjazdów.
    step=2 oznacza sprawdzanie co drugiego okrążenia dla drastycznego przyspieszenia obliczeń.
    """
    strategies = []
    min_stint, max_stint = 8, 35
    earliest, latest = 8, total_laps - 8

    if latest < earliest:
        return []

    possible_laps = list(range(earliest, latest + 1, step))

    for num_stops in range(min_stops, max_stops + 1):
        for strategy in combinations_with_replacement(possible_laps, num_stops):
            strategy_list = sorted(list(strategy))
            valid = True
            prev_stop = 0

            for stop in strategy_list:
                if not (min_stint <= stop - prev_stop <= max_stint):
                    valid = False
                    break
                prev_stop = stop

            if valid and not (min_stint <= total_laps - strategy_list[-1] <= max_stint):
                valid = False

            if valid and strategy_list not in strategies:
                strategies.append(strategy_list)
    return strategies


def prepare_lap_features(
    lap_number: int,
    stint_lap: int,
    compound: int,
    is_pit_lap: int,
    circuit_data: pd.DataFrame,
    params: Dict,
) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "LapNumber": [lap_number],
            "TyreLife": [stint_lap],
            "FreshTyre": [1 if stint_lap == 1 else 0],
            "IsPitLap": [is_pit_lap],
            "CompoundNumeric": [compound],
        }
    )

    env_cols = [
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "WindSpeed",
        "WindDirection",
    ]
    for col in env_cols:
        features[col] = (
            params.get(col.lower())
            if params.get(col.lower()) is not None
            else circuit_data[col].median()
        )

    features["Rainfall"] = 1 if params.get("rainfall") else 0

    excluded = ["LapTimeZScore", "Compound"]
    required_cols = [col for col in circuit_data.columns if col not in excluded]

    for col in required_cols:
        if col not in features.columns:
            if circuit_data[col].dtype in [np.float64, np.int64]:
                features[col] = circuit_data[col].median()
            else:
                features[col] = 0

    return features[required_cols]


def simulate_strategy(
    strategy: List[int],
    compounds: List[int],
    total_laps: int,
    circuit_data: pd.DataFrame,
    model: Any,
    params: Dict,
) -> Tuple[float, pd.DataFrame]:
    pit_stops = set(strategy)
    results = []
    current_stint, stint_lap = 0, 1

    for lap in range(1, total_laps + 1):
        is_pit_lap = 1 if lap in pit_stops else 0

        lap_features = prepare_lap_features(
            lap, stint_lap, compounds[current_stint], is_pit_lap, circuit_data, params
        )
        predicted_zscore = model.predict(lap_features)[0]

        results.append(
            {
                "LapNumber": lap,
                "Stint": current_stint + 1,
                "CompoundNumeric": compounds[current_stint],
                "PredictedZScore": predicted_zscore,
                "IsPitStop": bool(is_pit_lap),
            }
        )

        if is_pit_lap:
            current_stint += 1
            stint_lap = 1
        else:
            stint_lap += 1

    df = pd.DataFrame(results)
    return df["PredictedZScore"].mean(), df


def optimize_all_circuits(
    models_dict: Dict, data_dict: Dict, params: Dict
) -> pd.DataFrame:
    all_results = []
    race_laps_map = params.get("race_laps", {})
    compound_mapping = params.get("compounds", {})

    for circuit, model in models_dict.items():
        laps = race_laps_map.get(circuit, 56)
        strategies = generate_strategies(
            laps, params.get("min_stops", 1), params.get("max_stops", 3), step=2
        )

        total_combinations = 0
        for s in strategies:
            total_combinations += len(
                get_compound_combinations_with_f1_rules(
                    len(s) + 1, circuit, compound_mapping
                )
            )

        logger.info(
            f"--- Optymalizacja {circuit} | Kombinacji: {total_combinations} ---"
        )

        circuit_results = []
        with tqdm(
            total=total_combinations, desc=f"Tor: {circuit}", unit="komb"
        ) as pbar:
            for strategy in strategies:
                combinations = get_compound_combinations_with_f1_rules(
                    len(strategy) + 1, circuit, compound_mapping
                )
                for compounds in combinations:
                    mean_z, sim_df = simulate_strategy(
                        strategy, compounds, laps, data_dict[circuit], model, params
                    )
                    circuit_results.append(
                        {
                            "Circuit": circuit,
                            "Strategy": strategy,
                            "Compounds": compounds,
                            "MeanZScore": mean_z,
                            "Simulation": sim_df,
                        }
                    )
                    pbar.update(1)

        if circuit_results:
            all_results.append(pd.DataFrame(circuit_results))

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def select_best_strategies(
    optimization_results: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    if optimization_results.empty:
        return pd.DataFrame()
    top_n = params.get("top_n", 1)
    return (
        optimization_results.sort_values("MeanZScore")
        .groupby("Circuit")
        .head(top_n)
        .reset_index(drop=True)
    )


def save_optimization_results_csv(
    optimization_results: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    if optimization_results.empty:
        return pd.DataFrame()
    df = optimization_results.copy()
    compound_mapping = params.get("compounds", {})

    df["Strategy"] = df["Strategy"].apply(str)
    df["CompoundsCardinal"] = df.apply(
        lambda r: str(
            [
                translate_compound(c, r["Circuit"], compound_mapping)
                for c in r["Compounds"]
            ]
        ),
        axis=1,
    )

    cols_to_drop = ["Simulation", "Compounds"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def save_best_strategies_csv(
    best_strategies: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    return save_optimization_results_csv(best_strategies, params)


def generate_detailed_simulation(
    best_strategies: pd.DataFrame, models_dict: Dict, data_dict: Dict, params: Dict
) -> Dict[str, pd.DataFrame]:
    detailed = {}
    compound_mapping = params.get("compounds", {})

    for _, row in best_strategies.iterrows():
        circuit = row["Circuit"]
        _, sim_df = simulate_strategy(
            row["Strategy"],
            row["Compounds"],
            params["race_laps"].get(circuit, 56),
            data_dict[circuit],
            models_dict[circuit],
            params,
        )

        sim_df["CompoundCardinal"] = sim_df["CompoundNumeric"].apply(
            lambda x: translate_compound(x, circuit, compound_mapping)
        )
        sim_df["Circuit"] = circuit
        detailed[circuit] = sim_df
    return detailed


def save_detailed_simulations_csv(
    detailed_simulations: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    if not detailed_simulations:
        return pd.DataFrame()
    return pd.concat(detailed_simulations.values(), ignore_index=True)


def visualize_strategies(
    detailed_simulations: Dict[str, pd.DataFrame], params: Dict = None
) -> Dict[str, plt.Figure]:
    figures = {}
    for circuit, df in detailed_simulations.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

        ax1.plot(
            df["LapNumber"],
            df["PredictedZScore"],
            color="blue",
            label="Predicted Z-Score",
        )
        ax1.scatter(
            df[df["IsPitStop"]]["LapNumber"],
            df[df["IsPitStop"]]["PredictedZScore"],
            color="red",
            zorder=5,
            label="Pit Stop",
        )
        ax1.set_title(f"Strategy for {circuit} (Pit-loss embedded in Z-score)")
        ax1.set_ylabel("Z-Score")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        df["stint_change"] = df["IsPitStop"].shift(fill_value=False)
        df["stint_id"] = df["stint_change"].cumsum()

        stint_groups = df.groupby("stint_id")
        for stint_id, group in stint_groups:
            compound = group["CompoundCardinal"].iloc[0]
            color = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "white"}.get(
                compound, "grey"
            )
            ax2.barh(
                0,
                len(group),
                left=group["LapNumber"].min() - 1,
                color=color,
                edgecolor="black",
            )

            mid_lap = group["LapNumber"].min() + (len(group) / 2) - 1
            ax2.text(
                mid_lap,
                0,
                compound,
                ha="center",
                va="center",
                fontweight="bold",
                color="black" if color != "white" else "black",
            )

        ax2.set_yticks([])
        ax2.set_xlabel("Lap")
        ax2.set_xlim(0, df["LapNumber"].max())
        fig.tight_layout()
        figures[f"{circuit}.png"] = fig
    return figures
