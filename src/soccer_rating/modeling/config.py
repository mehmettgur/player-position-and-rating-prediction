from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    # Which label column to use for stratification (M1..M8)
    label_col: str
    # If None -> use all numeric features; otherwise use rule map for that label
    use_all_features: bool
    # tag string for reporting
    tag: str
