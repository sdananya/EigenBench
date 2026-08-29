"""
EigenBench run spec — stoic-arm comparison (sycophancy constitution, airisk scenarios).
Generated for the stoic-arm project: base + three character adapters + frontier models.
Collection/training parameters mirror runs/matrix/sycophancy/spec.py (upstream protocol).
"""

RUN_SPEC = {
    "verbose": True,
    "models": {
        "qwen3-base": "hf_local:Qwen/Qwen3-8B",
        "stoic": "hf_local:/workspace-vast/sdananya/stoic-arm/eigen_adapters/stoic",
        "mixed": "hf_local:/workspace-vast/sdananya/stoic-arm/eigen_adapters/mixed",
        "parrhesia": "hf_local:/workspace-vast/sdananya/stoic-arm/eigen_adapters/parrhesia",    },
    "dataset": {
        "path": "data/scenarios/airiskdilemmas.json",
        "start": 100,
        "count": 100,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/oct_sycophancy.json",
        "num_criteria": 10,
    },
    "collection": {
        "enabled": True,
        "evaluations_path": "evaluations.jsonl",
        "cached_responses_path": None,
        "allow_ties": True,
        "group_size": 4,
        "groups": 1,
        "sampler_mode": "random_judge_group",
    },
    "training": {
        "enabled": True,
        "model": "btd_ties",
        "dims": [2],
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1000,
        "batch_size": 32,
        "device": "cpu",
        "test_size": 0.2,
        "group_split": False,
        "separate_criteria": False,
        "bootstrap": {
            "enabled": True,
            "n_bootstraps": 100,
            "random_seed": 42,
            "save_models": False,
            "save_trust_matrices": True,
        },
    },
}
