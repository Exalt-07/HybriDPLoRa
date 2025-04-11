# train.py
import flwr as fl
from dp_engine import AdaptiveDPE

def main():
    dp_engine = AdaptiveDPE(target_epsilon=4.0)
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=dp_engine.update_config
    )
    fl.simulation.start_simulation(...)
