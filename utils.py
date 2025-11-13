import os, pandas as pd


DEFAULT_CONFIG = {
    "target_layer": None, # NEEDS TO BE DEFINED BASED ON EXPERIMENT
    "intervention_pos": None, # NEEDS TO BE DEFINED BASED ON EXPERIMENT
    "epochs": 1,
    "batch_size": 8,
    "total_samples": 5000,
    "low_rank_dimension": 1,
    "learning_rate": 1e-3,
    "gradient_accumulation_steps": 4,
    "cache_dir": "../../../shared/hf_cache",
    "baseline_accuracy": None,
}

neural_interventions = ['realnvp', 'revnet']