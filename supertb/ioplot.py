"""
Classes and functions associated with default matplotlib rc settings
"""

def setup_matplotlib(kind='default'):

    import matplotlib as mplt
    import matplotlib.pyplot as plt
    mplt.rcParams['axes.linewidth'] = 2.
    mplt.rcParams['xtick.major.width'] = 2.
    mplt.rcParams['ytick.major.width'] = 2.

