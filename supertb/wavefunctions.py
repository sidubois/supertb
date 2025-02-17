""" Python Classes and functions
Simple Pyhton classes and functions providing some geometrical functionalities
to handle atomic systems
"""

import time
import numpy as np
from supertb import Structure, Lattice, Spinor, Eigenset, PeriodicPointCollection, Wavecar

################################################################################        

class WaveFunctions(object):

    def init_from_vasp(self, file=None, verbose=False):

        # Read file and initialise wavecar object
        self.wavecar = Wavecar(file=file, verbose=verbose)
        self.colinear = self.wavecar.colinear
        self.lattice = self.wavecar.lattice

        ## Read global info from wavecar instance
        #self.kpt = wavecar.kpt
        #self.ener = wavecar.ener
        #self.colinear = wavecar.colinear
        #self.occ = wavecar.occ
        #self.nkpt, self.nband, self.nspin = self.ener.shape

        self.iwf = []
        self.wf = []
    
    def select_energies(self, klist, emin=-float("inf"), emax=float("inf"), shift_ibands=0):

        self.kpts = self.wavecar.kpt[klist]
        for ikpt_loc, ikpt_glob in enumerate(klist):
            for iband in range(self.wavecar.nband):
                for ispin in range(self.wavecar.nspin):
                    ewf = self.wavecar.ener[ikpt_glob,iband,ispin].real
                    if (emin <= ewf and ewf <= emax):
                       #self.iwf.append((ikpt_loc, iband - shift_ibands, ispin, ewf)) 
                       self.iwf.append((ikpt, iband - shift_ibands, ispin, ewf)) 

    def select_energies2(self, emin=-float("inf"), emax=float("inf"), shift_ibands=0):

        self.kpts = self.wavecar.kpt
        for ik in range(self.wavecar.kpt.shape[0]): 
            for ib in range(self.wavecar.nband):
                for ispin in range(self.wavecar.nspin):
                    ewf = self.wavecar.ener[ik,ib,ispin].real
                    if (emin <= ewf and ewf <= emax):
                       self.iwf.append((ik, ib - shift_ibands, ispin, ewf)) 


    def select_bands(self, ibands, shift_ibands=0):

        self.kpts = self.wavecar.kpt
        for ik in range(self.wavecar.kpt.shape[0]): 
            for ib in ibands:
                for ispin in range(self.wavecar.nspin):
                    ewf = self.wavecar.ener[ik,ib,ispin].real
                    self.iwf.append((ik, ib - shift_ibands, ispin, ewf)) 

    def select_from_dict(self, dic, shift_ibands=0):
        
        ikpt_loc = 0
        for ikpt in dic.keys():
            for iband in dic[ikpt].keys():
                for ispin in dic[ikpt][iband]:
                    ewf = self.wavecar.ener[ikpt,iband,ispin].real
                    #self.iwf.append((ikpt_loc, iband - shift_ibands, ispin, ewf)) 
                    self.iwf.append((ikpt, iband - shift_ibands, ispin, ewf)) 
            ikpt_loc += 1


    def to_eigenset(self, normalized=True, efermi=0.):

        spinpol = False
        spinorb = False
        if self.wavecar.nspin == 2:
            spinpol = True
        if not self.wavecar.colinear:
            spinorb = True
        spin = Spinor(spinpol=spinpol, spinorb=spinorb)

        eig = Eigenset(self.lattice, spin=spin, kpts=self.wavecar.kpt, fractional=False)
        for ik, ib, ispin, ewf in self.iwf:

            V = self.wavecar.plane_waves_coefficients(ik,ib,ispin) 
            eig.set_eigenvalue(ik, ib, ispin, E=ewf-efermi)
            eig.set_eigenvalue(ik, ib, ispin, V=V)

        return eig

    def evaluate(self, grid, normalized=True, **kwargs):

        if self.colinear:
            ndim = 1
        else:
            ndim = 2

        if 'subset' in kwargs:
            subset = kwargs['subset']
            coords = grid.subset_coords(subset)
            for ikpt, iband, ispin, ewf in self.iwf:
                wf = grid.init_field(np.complex, ikpt=ikpt, iband=iband, ispin=ispin, ener=ewf, subset=subset, ndim=ndim)
                wf.values = self.wavecar.evaluate(ikpt, iband, ispin, coords, normalized=normalized)
                self.wf.append(wf)
        else:
            coords = grid.coords
            for ikpt, iband, ispin, ewf in self.iwf:
                wf = grid.init_field(np.complex, ikpt=ikpt, iband=iband, ispin=ispin, ener=ewf, ndim=ndim)
                wf.values = self.wavecar.evaluate(ikpt, iband, ispin, coords, normalized=normalized)
                self.wf.append(wf)
       
