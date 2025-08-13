from typing import Dict, List

GENERAL_ORDER = ['GK', 'Defense', 'Midfield', 'Forward']
DETAILED_ORDER = [
    'GK',
    'Center Defense',
    'Side Defense',
    'Center Midfield',
    'Side Midfield',
    'Center Forward',
    'Side Forward'
]

RULE_FEATURES_GENERAL: Dict[str, List[str]] = {
    'GK': ['gk_reflexes','gk_positioning','gk_diving','gk_handling','gk_kicking'],
    'Defense': ['marking','standing_tackle','sliding_tackle','interceptions','strength','aggression'],
    'Midfield': ['vision','short_passing','long_passing','ball_control','dribbling','agility'],
    'Forward': ['finishing','volleys','positioning','shot_power','heading_accuracy','dribbling'],
}

RULE_FEATURES_DETAILED: Dict[str, List[str]] = {
    'GK': ['gk_reflexes','gk_positioning','gk_diving','gk_handling','gk_kicking'],
    'Center Defense': ['marking','standing_tackle','sliding_tackle','interceptions','strength','heading_accuracy'],
    'Side Defense': ['crossing','acceleration','sprint_speed','agility','standing_tackle'],
    'Center Midfield': ['vision','short_passing','long_passing','interceptions','standing_tackle','aggression'],
    'Side Midfield': ['acceleration','sprint_speed','agility','dribbling','crossing'],
    'Center Forward': ['finishing','positioning','volleys','shot_power','heading_accuracy','strength'],
    'Side Forward': ['acceleration','dribbling','agility','crossing','finishing','curve'],
}
