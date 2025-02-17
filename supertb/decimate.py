"""
"""

import time
import numpy as np
import pickle
from math import exp, log, sin, cos, pi, sqrt, trunc, ceil
import scipy.constants as const
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as spalg 
from scipy.sparse.linalg import eigsh as sparse_eigh
import scipy
from scipy.interpolate import splev, splrep

from copy import copy, deepcopy
from supertb import NumericalAtom, AtomicTable, AtomicBasisSet, Eigenset, BlochPhase, Spinor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings


def decimate_periodic_orbitals(isite, iorb, graph, egraph, table, arg, eshift=0., iener=1.e-6, etol=1.e-8):



