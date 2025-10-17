"""
Build reasoning gym composite enviornments. 

Assumes youll want to train on some subset, and test on another. 

Could be identical, or could be different to test out some transfer learning. 


"""

import reasoning_gym
from reasoning_gym.composite import DatasetSpec


def make_specs(names, weights, configs=None):
    """
    Create DatasetSpec list from names, weights, and optional configs.

    names and weights should be the same length. configs is optional; if you
    leave it out, each dataset just uses an empty config
    """
    if configs is None:
        configs = [{} for _ in names]
    if not (len(names) == len(weights) == len(configs)):
        raise ValueError("names, weights, and configs must have the same length")
    specs = []
    for n, w, cfg in zip(names, weights, configs):
        weight_value = float(w)
        if weight_value <= 0:
            raise ValueError("all weights must be positive")
        specs.append(DatasetSpec(name=n, weight=weight_value, config=cfg or {}))
    return specs


def build_composite_set(size: int, seed: int, names, weights, configs=None):
    """
    Build a composite dataset and return the reasoning_gym dataset object.

    You pass a list of task names and matching weights (and optional configs),
    and this creates one dataset where sampling follows those weights.
    """
    specs = make_specs(names, weights, configs)
    return reasoning_gym.create_dataset('composite', size=size, seed=seed, datasets=specs)

def build_reasoning_envs(train_names, train_weights, train_size: int, seed: int = 42, train_configs=None, eval_names=None, eval_weights=None, eval_size: int | None = None, eval_configs=None):
    """
    Build a composite train dataset, and optionally a composite eval dataset.

    Returns a tuple: (train_dataset, eval_dataset_or_None).
    Both are the raw reasoning_gym dataset objects, so you can iterate them and
    call methods like dataset.score_answer(...).
    """

    if eval_names is None:
        eval_names = []
    if eval_weights is None:
        eval_weights = []

    # Validate train
    if train_configs is not None and len(train_configs) != len(train_names):
        raise ValueError("train_configs must match train_names length if provided")
    if not (len(train_names) == len(train_weights)):
        raise ValueError("train_names and train_weights must have the same length")

    # Validate eval
    if eval_names or eval_weights or eval_configs or eval_size:
        if eval_configs is not None and len(eval_configs) != len(eval_names):
            raise ValueError("eval_configs must match eval_names length if provided")
        if not (len(eval_names) == len(eval_weights)):
            raise ValueError("eval_names and eval_weights must have the same length")

    # Build training composite set
    train_dataset = build_composite_set(size=train_size, seed=seed, names=train_names, weights=train_weights, configs=train_configs)

    # Optional eval-only composite set
    if eval_names and eval_weights and (eval_size is not None):
        eval_dataset = build_composite_set(size=eval_size, seed=seed, names=eval_names, weights=eval_weights, configs=eval_configs)
    else:
        eval_dataset = None

    return train_dataset, eval_dataset


if __name__ == "__main__":
    train_names = ["leg_counting", "figlet_font"]
    train_weights = [0.7, 0.3]
    train_size = 1000
    seed = 123

    train_configs = None  # Or e.g., [{"length": 5}, {"maze_size": 6}]
    eval_names = ["leg_counting"]
    eval_weights = [1.0]
    eval_size = 200
    eval_configs = None

    train_dataset, eval_dataset = build_reasoning_envs(
        train_names,
        train_weights,
        train_size,
        seed=seed,
        train_configs=train_configs,
        eval_names=eval_names,
        eval_weights=eval_weights,
        eval_size=eval_size,
        eval_configs=eval_configs
    )

    print("Train dataset size:", len(train_dataset))
    for i, ex in enumerate(train_dataset):
        if i < 2:
            print("Train example:", ex)
    if eval_dataset is not None:
        print("Eval dataset size:", len(eval_dataset))
        for i, ex in enumerate(eval_dataset):
            if i < 2:
                print("Eval example:", ex)

