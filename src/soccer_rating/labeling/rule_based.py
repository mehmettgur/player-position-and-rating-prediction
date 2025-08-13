import numpy as np
import pandas as pd

def rule_general(row: pd.Series) -> int:
    # returns argmax index for [GK, DEF, MID, FOR]
    gk = 0.45*row['gk_reflexes'] + 0.25*row['gk_positioning'] + 0.15*row['gk_diving'] + 0.10*row['gk_handling'] + 0.05*row['gk_kicking']
    defense = 0.30*row['marking'] + 0.20*row['standing_tackle'] + 0.15*row['sliding_tackle'] + 0.15*row['interceptions'] + 0.10*row['strength'] + 0.10*row['aggression']
    midfield = 0.25*row['vision'] + 0.20*row['short_passing'] + 0.15*row['long_passing'] + 0.15*row['ball_control'] + 0.15*row['dribbling'] + 0.10*row['agility']
    forward = 0.30*row['finishing'] + 0.20*row['volleys'] + 0.20*row['positioning'] + 0.15*row['shot_power'] + 0.15*row['heading_accuracy']
    return int(np.argmax([gk, defense, midfield, forward]))

RULE_MAP_GENERAL = {0:'GK',1:'Defense',2:'Midfield',3:'Forward'}

def rule_detailed(row: pd.Series) -> str:
    scores = {}
    scores['GK'] = 0.45*row['gk_reflexes'] + 0.25*row['gk_positioning'] + 0.15*row['gk_diving'] + 0.10*row['gk_handling'] + 0.05*row['gk_kicking']
    scores['Center Defense'] = 0.25*row['marking'] + 0.20*row['standing_tackle'] + 0.20*row['sliding_tackle'] + 0.15*row['interceptions'] + 0.10*row['strength'] + 0.10*row['heading_accuracy']
    scores['Side Defense'] = 0.25*row['crossing'] + 0.20*row['acceleration'] + 0.20*row['sprint_speed'] + 0.15*row['agility'] + 0.20*row['standing_tackle']
    scores['Center Midfield'] = 0.25*row['vision'] + 0.20*row['short_passing'] + 0.15*row['long_passing'] + 0.15*row['interceptions'] + 0.15*row['standing_tackle'] + 0.10*row['aggression']
    scores['Side Midfield'] = 0.20*row['crossing'] + 0.20*row['sprint_speed'] + 0.15*row['agility'] + 0.15*row['dribbling'] + 0.10*row['curve'] + 0.10*row['acceleration'] + 0.10*row['shot_power']
    scores['Center Forward'] = 0.30*row['finishing'] + 0.20*row['positioning'] + 0.15*row['volleys'] + 0.10*row['shot_power'] + 0.15*row['heading_accuracy'] + 0.10*row['strength']
    scores['Side Forward'] = 0.25*row['acceleration'] + 0.20*row['dribbling'] + 0.20*row['agility'] + 0.15*row['crossing'] + 0.10*row['finishing'] + 0.10*row['curve']
    return max(scores, key=scores.get)
