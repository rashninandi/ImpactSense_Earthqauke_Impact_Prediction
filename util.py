def severity_label(magnitude, sig):
    if magnitude >= 8:
        return "Catastrophic"
    elif magnitude >= 7:
        return "Extreme"
    elif magnitude >= 6:
        return "Major"
    else:
        return "Moderate"


def log_transform1(x):
    import numpy as np
    return np.log1p(x)