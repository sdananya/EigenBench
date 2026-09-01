"""
EigenBench run spec — canonical decomposition cell: 11-model pool (all arms + base_prompted + frontier) (sycophancy constitution, flipped-AITA scenarios).
Generated for the stoic-arm project: base + three character adapters + frontier models.
Collection/training parameters mirror runs/matrix/sycophancy/spec.py (upstream protocol).
"""

RUN_SPEC = {
    "verbose": True,
    "models": {
        "qwen3-base": "hf_local:Qwen/Qwen3-8B",
        "base_prompted": "hf_local:/workspace-vast/sdananya/stoic-arm/eigen_adapters/qwen3-8b-prompted",
        "stoic": "hf_local:/workspace-vast/sdananya/stoic-arm/eigen_adapters/stoic",
        "mixed": "hf_local:/workspace-vast/sdananya/stoic-arm/eigen_adapters/mixed",
        "stoic_noD7": "hf_local:/workspace-vast/sdananya/stoic-arm/eigen_adapters/stoic_noD7",
        "stoic4": "hf_local:/workspace-vast/sdananya/stoic-arm/eigen_adapters/stoic4",
        "stoic4n": "hf_local:/workspace-vast/sdananya/stoic-arm/eigen_adapters/stoic4n",
        "parrhesia": "hf_local:/workspace-vast/sdananya/stoic-arm/eigen_adapters/parrhesia",
        "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
        "gpt-5": "openai/gpt-5",
        "gemini-2.5-pro": "google/gemini-2.5-pro",
    },
    "dataset": {
        "path": "data/scenarios/elephant_aita_flip.json",
        "start": 0,
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
        "max_tokens": 8192,
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
