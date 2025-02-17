"""
This module defines the classes and functions that enable to handle tight-binding
models
"""

import numpy as np
from math import exp, log, sin, cos, pi, sqrt, trunc, ceil
import scipy.constants as const
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as spalg 
import scipy
from copy import deepcopy
from pymatgen.core import Lattice, Structure
import time
import itertools
import sys
import os
sys.path.append("/Users/simondubois/Work/Programs/CoverTree-master")
from .covertree import CoverTree
import networkx as nx
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import inspect


class NumericalAtom(dict):
    """
    Chemical element with associated set of atomic orbitals.

    .. attribute:: label

        Label used internally to identify Numerical Atoms in a unique way, 
        e.g., "C", "Fe", or "C1", "C2", "Cspd"

    .. attribute:: symbol

       Element symbol, e.g. "C", "Fe"

    .. attribute:: nshells

       Maximum number of shells per orbital angular momentum

    .. attribute:: num_orbitals

       Number of orbitals

    """

    orbital_type_index = {'s':0, \
                     'x':1, 'y':2,  'z':3,  \
                     'px':1, 'py':2,  'pz':3, \
                     'xy':4,  'yz':5,  'xz':6, 'x2-y2':7, '3z2-r2':8, \
                     'dxy':4,  'dyz':5,  'dxz':6, 'dx2-y2':7, 'd3z2-r2':8}

    orbital_angmom_index = {0:{0:0}, \
                            1:{1:1,-1:2,0:3}, \
                            2:{-2:4,-1:5,1:6,2:7,0:8}}

    orbital_type =   ['s','px','py','pz','dxy','dyz','dxz','dx2-y2','d3z2-r2']
    orbital_angmom = [(0,0),(1,1),(1,-1),(1,0),(2,-2),(2,-1),(2,1),(2,2),(2,0)]

    def __init__(self, label, symbol, orbs):
        """
        Creates a Numerical Atom.

        Args:
            label: 
                String that will be used to uniquely identify the
                created numerical atom.
            symbol: 
                Symbol of the element in the periodic table 
            orbs: 
                List of orbitals. Orbitals can be specified either by
                their index, the associated strings, or the 
                (l,m) quantum numbers:
                
                0: 's'  	, (0, 0) 
                1: 'px'		, (1, 1)
                2: 'py'		, (1,-1)
                3: 'pz'		, (1, 0)
                4: 'dxy'	, (2,-2)
                5: 'dyz'	, (2,-1)
                6: 'dxz'	, (2, 1)
                7: 'dx2-y2'	, (2, 2)
                8: 'd3z2-r2'	, (2, 0) 
        """

        super(dict,self).__init__()
        self['label'] = label
        self['symbol'] = symbol
        self['nshells'] = 1

        self['orbs'] = []
        self['params'] = []
        for x in orbs:
            self.add_orbital(x)

    def add_orbital(self,orb):
        """
        Adds an orbital.

        Args:
            orb: Orbital specified either by its index, its basestring label,
                 or its (l,m) quantum numbers:

                0: 's'  	, (0, 0) 
                1: 'px'		, (1, 1)
                2: 'py'		, (1,-1)
                3: 'pz'		, (1, 0)
                4: 'dxy'	, (2,-2)
                5: 'dyz'	, (2,-1)
                6: 'dxz'	, (2, 1)
                7: 'dx2-y2'	, (2, 2)
                8: 'd3z2-r2'	, (2, 0) 

        """

        iorb = self.get_type_index(orb)         
        self['orbs'].append([iorb,self._new_orbital_shell(iorb)])
        self['params'].append({})
        for iorb, ishell in self['orbs']:
            if ishell > self['nshells']-1:
                self['nshells'] = ishell+1  

    def set_atomic_params(self, **params):

        for idx in range(len(self['params'])):
            self.set_orbital_params(params,index=idx)

    def set_orbital_params(self, params, shell=0, **kwargs):
        """
        Set orbital parameters.
        """
        if 'index' in kwargs:
            found, idx = self._is_orbital_index(kwargs['index'])

        elif 'type' in kwargs:
            iorb = self.get_type_index(kwargs['type'])            
            found, idx = self._is_orbital_id((iorb,shell))

        self['params'][idx].update((k,v) for k,v in params.items())

    def remove_orbital(self, shell=0, **kwargs):
        """
        Remove an orbital.
        """

        if 'index' in kwargs:
            found, idx = self._is_orbital_index(kwargs['index'])

        elif 'type' in kwargs:
            iorb = self.get_type_index(kwargs['type'])            
            found, idx = self._is_orbital_id((iorb,shell))

        if found:
            self._remove_orbital(idx) 
       
    def _remove_orbital(self, x): 
        ishell = self['orbs'][x][0]
        del self['orbs'][x]
        del seld['params'][x]

        for orb, shell in self['orbs']:
            if shell == ishell:
                return
        for iorb, orb in enumerate(self['orbs']):
            if orb[1]>ishell:
                self['orbs'][iorb][1] -= 1
        self['nshells'] -= 1
    
 
    def _is_orbital_index(self,x):
        if x not in range(self.num_orbitals):
            return False, 0
        else:
            return True, x
    
    def _is_orbital_id(self, x):
        
        for idx, orb in enumerate(self['orbs']):
            if orb[0] == x[0] and orb[1] == x[1]:
               return True, idx 

    def get_type_index(self, orb):

        if isinstance(orb, str) :
            self._test_orbital_string(orb)
            iorb = NumericalAtom.orbital_type_index[orb]
        elif isinstance(orb, int):
            self._test_orbital_int(orb)
            iorb = orb
        elif (isinstance(orb, tuple) or isinstance(orb, list)) and \
             len(orb) == 2:
            self._test_orbital_tuple(orb)
            iorb = NumericalAtom.orbital_angmom_index[orb[0]][orb[1]]
        else:
            raise ValueError(" The orbitals should be given either by name or by index ! \
                               Acceptable names are s, px, py, pz, dxy, dyz, dxz, dx2-y2, and d3z2-r2. \
                               Acceptable indexes  are 0, 1, 2, 3, 4, 5, 6, 7, and 8.")
         
        return iorb

 
    def _test_orbital_string(self,x):
        if x not in NumericalAtom.orbital_type_index:
            raise ValueError(" The orbitals should be given either by name, index or (l,m) quantum numbers ! \
                               Acceptable names are s, px, py, pz, dxy, dyz, dxz, dx2-y2, and d3z2-r2.")

    def _test_orbital_int(self,x):
        if x not in range(9):
            raise ValueError(" The orbitals should be given either by name, index or (l,m) quantum numbers ! \
                               Acceptable indexes  are 0, 1, 2, 3, 4, 5, 6, 7, and 8.")
 
    def _test_orbital_tuple(self,x):

        if x[0] not in NumericalAtom.orbital_angmom_index:
            raise ValueError(" The orbitals should be given either by name, index or (l,m) quantum numbers ! \
                               Acceptable names are s, px, py, pz, dxy, dyz, dxz, dx2-y2, and d3z2-r2.")
        elif x[1] not in NumericalAtom.orbital_angmom_index[x[0]]:
            raise ValueError(" The orbitals should be given either by name, index or (l,m) quantum numbers ! \
                               Acceptable names are s, px, py, pz, dxy, dyz, dxz, dx2-y2, and d3z2-r2.")

    def _new_orbital_shell(self,orb):
        
        shell = 0
        for iorb, ishell in self['orbs']:
            if iorb == orb:
                shell = max(shell,ishell+1)   
        return shell

    @property
    def label(self):
        return self['label']

    @label.setter
    def label(self,label):
        self['label'] = label

    @property
    def symbol(self):
        return self['symbol']

    @symbol.setter
    def symbol(self, symbol):
        self['symbol'] = symbol
        
    @property
    def num_shells(self):
        return self['nshells']
        
    @property
    def num_orbitals(self):
        return len(self['orbs'])
        
    @property
    def orbitals(self):
        return self['orbs']
        
    @orbitals.setter
    def orbitals(self, orbs):
        for orb in orbs:
            self.add_orbital(orb)

class AtomicTable(dict):
    """
    Dictionary whose keys and values are respectively the user defined 
    atomic labels and the associated NumericalAtoms.
    """
    def __init__(self, *args):
        """
        Creates an AtomicTable instance from a list of NumericalAtoms.

        Args:
            *args: enables to pass the NumericalAtoms either as separate 
                positional arguments or as a list of NumericalAtoms. 
        """
  
        super(dict,self).__init__()
        for arg in args:
            self.add_atoms(arg)
          
    def add_atoms(self, *args):
        """
        Adds an atom or a list of atoms to the AtomicTable

        Args:
            *args: enables to pass the NumericalAtoms either as separate 
                positional arguments or as a list of NumericalAtoms. 
        """

        for atom in args:
            if isinstance(atom, (list,tuple)):
                for item in atom:
                    if not isinstance(item,NumericalAtom):
                        raise ValueError()
                        self[item.label] = item

            elif isinstance(atom,NumericalAtom):
                self[atom.label] = atom
 
            else:
                raise ValueError()
          
class AtomicBasisSet(object):
    
    def __init__(self, sites, atoms, atomictable):

        """
        Initialises an atomic basis set from the lists of sites and atomic symbols by looking
        in the given atomic table.
        """

        self._size = 0
        self.basisset = []
        self.sites = []
        self.ptrs = {} 

        #print sites
	#print atoms
        for isite, data  in enumerate(zip(sites,atoms)):
            sitelabel, atomlabel = data

            self.sites.append({})
            self.ptrs[sitelabel] = isite
            self.sites[isite]['atom'] = atomlabel
            self.sites[isite]['first'] = self._size
            self.sites[isite]['num'] = atomictable[atomlabel].num_orbitals

            for ibasis in range(self.sites[isite]['num']):
                self.basisset.append({})
                self.basisset[-1]['site'] = sitelabel
                self.basisset[-1]['local'] = ibasis
                self.basisset[-1]['atom'] = atomlabel
                self.basisset[-1]['orb'] = atomictable[atomlabel].orbitals[ibasis]

                self._size += 1

    @property
    def size(self):
        return self._size

    def iter_basis_on_sites(self,*args):

        for arg in args:
            if not isinstance(arg,(list,tuple)):
                arg = [arg]

            for subarg in arg:
                site = self.sites[self.ptrs[subarg]]
                for ibasis in range(site['first'],site['first']+site['num']):
                    yield ibasis

    def first_basis_on_site(self, site):
 
        return self.sites[self.ptrs[site]]['first']

    def last_basis_on_site(self, site):
 
        site = self.sites[self.ptrs[site]]
        return site['first']+site['num']-1

    def num_basis_on_site(self, site):
 
        return self.sites[self.ptrs[site]]['num']

    def site_of_basis(self,ibasis):

        return self.basisset[ibasis]['site']

    def atom_of_basis(self,ibasis):

        return self.basisset[ibasis]['atom']

    def local_index_of_basis(self,ibasis):

        return self.basisset[ibasis]['local']

    def orbital_of_basis(self,ibasis):

        return self.basisset[ibasis]['orb']



