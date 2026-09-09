# iadc_mapper.py
# IADC Fixed Cutter Classification System mapping tables and decoder.

IADC_BODY = {
    'M': {'name': 'Matrix', 'density': 14.5, 'color': '#7f8c8d', 'wear_resistance': 'high', 'key': 'M'},
    'S': {'name': 'Steel', 'density': 7.85, 'color': '#95a5a6', 'wear_resistance': 'medium', 'key': 'S'},
    'D': {'name': 'Diamond-Enhanced', 'density': 3.5, 'color': '#f1c40f', 'wear_resistance': 'extreme', 'key': 'D'},
}

IADC_FORMATION = {
    1: {'name': 'Very Soft', 'ucs_min': 0, 'ucs_max': 20, 'wear_factor': 0.1, 'key': 1},
    2: {'name': 'Soft', 'ucs_min': 20, 'ucs_max': 40, 'wear_factor': 0.3, 'key': 2},
    3: {'name': 'Soft-Medium', 'ucs_min': 40, 'ucs_max': 60, 'wear_factor': 0.5, 'key': 3},
    4: {'name': 'Medium', 'ucs_min': 60, 'ucs_max': 80, 'wear_factor': 0.7, 'key': 4},
    5: {'name': 'Medium-Hard', 'ucs_min': 80, 'ucs_max': 120, 'wear_factor': 1.2, 'key': 5},
    6: {'name': 'Hard', 'ucs_min': 120, 'ucs_max': 180, 'wear_factor': 1.6, 'key': 6},
    7: {'name': 'Extremely Hard', 'ucs_min': 180, 'ucs_max': 250, 'wear_factor': 2.0, 'key': 7},
}

IADC_CUTTER = {
    1: {'size_mm': 19.0, 'count_range': (30, 40), 'label': 'Heavy/Aggressive', 'key': 1},
    2: {'size_mm': 16.0, 'count_range': (25, 35), 'label': 'Medium', 'key': 2},
    3: {'size_mm': 13.0, 'count_range': (40, 50), 'label': 'Dense', 'key': 3},
    4: {'size_mm': 8.0, 'count_range': (50, 70), 'label': 'Light/High-Density', 'key': 4},
}

IADC_PROFILE = {
    1: {'name': 'Short Fishtail', 'cone_angle': 180, 'stability': 'low', 'key': 1},
    2: {'name': 'Short Profile', 'cone_angle': 160, 'stability': 'medium', 'key': 2},
    3: {'name': 'Medium Profile', 'cone_angle': 140, 'stability': 'high', 'key': 3},
    4: {'name': 'Long Profile', 'cone_angle': 120, 'stability': 'very high', 'key': 4},
}

def decode_iadc(code: str) -> dict:
    """Convert a 4‑character IADC code to design parameters."""
    if len(code) != 4:
        raise ValueError("IADC code must be 4 characters.")
    return {
        'code': code,
        'body': IADC_BODY[code[0]],
        'formation': IADC_FORMATION[int(code[1])],
        'cutter': IADC_CUTTER[int(code[2])],
        'profile': IADC_PROFILE[int(code[3])],
    }