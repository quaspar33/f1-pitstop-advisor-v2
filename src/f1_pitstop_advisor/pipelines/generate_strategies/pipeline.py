from .nodes import (
    optimize_all_circuits,
    select_best_strategies,
    generate_detailed_simulation,
    visualize_strategies,
    save_optimization_results_csv,
    save_best_strategies_csv,
    save_detailed_simulations_csv,
)
from kedro.pipeline import node, Pipeline


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=optimize_all_circuits,
                inputs=[
                    "best_parameters",
                    "circuit_lap_data",
                    "params:strategy_optimization",
                ],
                outputs="optimization_results",
                name="optimize_all_circuits_node",
            ),
            node(
                func=select_best_strategies,
                inputs=["optimization_results", "params:strategy_optimization"],
                outputs="best_strategies",
                name="select_best_strategies_node",
            ),
            node(
                func=save_optimization_results_csv,
                inputs=["optimization_results", "params:strategy_optimization"],
                outputs="optimization_results_csv",
                name="save_optimization_results_csv_node",
            ),
            node(
                func=save_best_strategies_csv,
                inputs=["best_strategies", "params:strategy_optimization"],
                outputs="best_strategies_csv",
                name="save_best_strategies_csv_node",
            ),
            node(
                func=generate_detailed_simulation,
                inputs=[
                    "best_strategies",
                    "best_parameters",
                    "circuit_lap_data",
                    "params:strategy_optimization",
                ],
                outputs="detailed_simulations",
                name="generate_detailed_simulation_node",
            ),
            node(
                func=save_detailed_simulations_csv,
                inputs=["detailed_simulations"],
                outputs="detailed_simulations_csv",
                name="save_detailed_simulations_csv_node",
            ),
            node(
                func=visualize_strategies,
                inputs=["detailed_simulations", "params:strategy_optimization"],
                outputs="strategy_plots",
                name="visualize_strategies_node",
            ),
        ]
    )
