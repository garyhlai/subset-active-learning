from src.active_learner import ActiveLearner, ActiveLearnerConfig
import pickle
from pathlib import Path


def test_uncertainty_sampling():
    with open(Path(__file__).parent / "test_active_learner_data.pkl", "rb") as f:
        expected_selected_data = pickle.load(f)

    config = ActiveLearnerConfig(debug=True, sampling_sizes=(1000, 2000), strategy="uncertainty_sampling")
    active_learner = ActiveLearner(config)
    actual_selected_data = active_learner.uncertainty_sampling(1000)

    assert actual_selected_data == expected_selected_data
