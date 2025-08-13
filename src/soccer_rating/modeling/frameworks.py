from .config import ModelConfig

MODEL_PARAMS = {
    'M1': ModelConfig(label_col='kmeans_position_label',     use_all_features=False, tag='General/KMeans'),
    'M2': ModelConfig(label_col='kmeans_position_label',     use_all_features=True,  tag='General/KMeans'),
    'M3': ModelConfig(label_col='kmeans_detailed_position',  use_all_features=False, tag='Detailed/KMeans'),
    'M4': ModelConfig(label_col='kmeans_detailed_position',  use_all_features=True,  tag='Detailed/KMeans'),
    'M5': ModelConfig(label_col='rule_position_label',       use_all_features=False, tag='General/Rules'),
    'M6': ModelConfig(label_col='rule_position_label',       use_all_features=True,  tag='General/Rules'),
    'M7': ModelConfig(label_col='rule_detailed_position',    use_all_features=False, tag='Detailed/Rules'),
    'M8': ModelConfig(label_col='rule_detailed_position',    use_all_features=True,  tag='Detailed/Rules'),
}
