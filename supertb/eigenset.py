"""
"""

import numpy as np
import pickle
from math import exp, log, sin, cos, pi, sqrt, trunc, ceil
import scipy.constants as const
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as spalg 
import scipy
from scipy.interpolate import splev, splrep

from copy import copy, deepcopy
from supertb import BlochPhase, Spinor, AtomicBasisSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings

class Eigenset(dict):

    def __init__(self, lattice, **kwargs):
 
        self.lattice = lattice

        if 'spin' in kwargs:
            self.spin = kwargs['spin']
        else:
            self.spin = Spinor(**kwargs)

        if 'kpts' in kwargs:
            kpts = kwargs['kpts']
        else:
            kpts = [[0.,0.,0.]]

        if 'fractional' in kwargs:
            fractional = kwargs['fractional']
        else:
            fractional = False

        if fractional:
            self.kpts = np.array([np.dot(kpt,lattice.reciprocal_lattice.matrix) \
                                      for kpt in kpts])
        else:
            self.kpts = kpts

        if 'vertices' in kwargs:
            self.vertices = kwargs['vertices']

        if 'wkpts' in kwargs:
            self.wkpts = kwargs['wkpts']
        else:
            #self.wkpts = [1.]*len(self.kpts)
            self.wkpts = [1./len(self.kpts)]*len(self.kpts)

        for ikpt in range(len(kpts)):
            self[ikpt] = dict()
   
    def set_bands(self, ener):

        self.set_eigenvalues(E=ener)
        self.bands = np.array(E=ener)         
     
    def set_eigenvalues(self, **kwargs):

        for ikpt, values in enumerate(zip(*kwargs.values())):
            self.set_eigenvalues_k(ikpt, **dict(zip(kwargs.keys(), values)))
        
    def set_eigenvalues_k(self, ikpt, **kwargs):

        for iband, values in enumerate(zip(*kwargs.values())):
            self.set_eigenvalues_k_b(ikpt, iband, **dict(zip(kwargs.keys(), values)))

    def set_eigenvalues_k_b(self, ikpt, iband, **kwargs):
        
        for ispin, values in enumerate(zip(*kwargs.values())):
            self.set_eigenvalue(ikpt, iband, ispin, **dict(zip(kwargs.keys(), values)))

    def set_eigenvalue(self, ikpt, iband, ispin, **kwargs):

        self.check_kpoint_index(ikpt)
        self.create_entry(ikpt, iband, ispin)
        for key, value in kwargs.items():
            self[ikpt][iband][ispin][key] = value

    def get_eigenvalue(self, ikpt, iband, ispin):

        if ikpt in self:
            if iband in self[ikpt]:
                if ispin in self[ikpt][iband]:
                    return self[ikpt][iband][ispin]
        return {}

    def set_default_eigenvalue_data(self, **kwargs):
 
        for ikp in self:
            for iband in self[ikp]:
                for ispin in self[ikp][iband]:
                    for key, value in kwargs.items():
                        self[ikp][iband][ispin][key] = value

    def eigenvalues_iter(self):

        for ikp in self:
            for iband in self[ikp]:
                for ispin in self[ikp][iband]:
                    yield ikp, iband, ispin, self[ikp][iband][ispin]

    def energies_iter(self):

        for ikp in self:
            for iband in self[ikp]:
                for ispin in self[ikp][iband]:
                    yield ikp, iband, ispin, self[ikp][iband][ispin]['E']

    def energies_array(self):
        
        return np.fromiter(self.energies_iter(),dtype=[('',int),('',int),('',int),('',complex)]) 

    def evaluate_spin_direction3(self, flat=True):


        for ikp, iband, ispin, eigenval in self.eigenvalues_iter():
        
            if 'V' in eigenval:    
                if flat:
                    wf = self.spin.spinor_vector(eigenval['V'])
                else:
                    wf = eigenval['V']
               
                a = wf[:,0] 
                b = wf[:,1] 
                ac = np.conj(a)
                bc = np.conj(b)
               
                nx = (np.sum(ac*b)+np.sum(bc*a)).real
                ny = -(np.sum(bc*a)-np.sum(ac*b)).imag
                nz = (np.sum(ac*a)-np.sum(bc*b)).real
               
                s = np.array([nx, ny, nz])  
                self[ikp][iband][ispin]['S'] = s/np.linalg.norm(s)

    def evaluate_spin_direction2(self, flat=True):

        for ikp, iband, ispin, eigenval in self.eigenvalues_iter():
            
            if 'V' in eigenval:    
                if flat:
                    wf = self.spin.spinor_vector(eigenval['V'])
                else:
                    wf = eigenval['V']
                a = np.sum(wf[:,0]*np.conj(wf[:,0]))/np.sum(wf[:,1]*np.conj(wf[:,1]))
                phi = -np.arctan2(a.imag,a.real)
                theta = 2.*np.arctan(1./np.absolute(a))
                sdir = [np.sin(theta)*np.cos(phi),\
                        np.sin(theta)*np.sin(phi),\
                        np.cos(theta)]
                
                self[ikp][iband][ispin]['S2'] = np.array([theta, phi])
                self[ikp][iband][ispin]['S'] = np.array(sdir)
                #print ikp, iband, ispin, eigenval['E'], sdir

    def evaluate_spin_direction(self, flat=True):

        for ikp, iband, ispin, eigenval in self.eigenvalues_iter():
            
            if 'V' in eigenval:    
                if flat:
                    wf = self.spin.spinor_vector(eigenval['V'])
                else:
                    wf = eigenval['V']
                
                #wf = self.spin.spinor_vector(eigenval['V'])
                #a = eigenval['V'][:,0] 
                #b = eigenval['V'][:,1] 
                a = wf[:,0] 
                b = wf[:,1] 
                ac = np.conj(a)
                bc = np.conj(b)
                
                naa = np.sum(a*ac)
                nbb = np.sum(b*bc)
                nab = np.sum(a*bc)
                phi = -np.arctan2(nab.imag,nab.real)
                theta = np.arctan2(2*(nab.real*np.cos(phi)-nab.imag*np.sin(phi)),(naa-nbb).real)
                sdir = [(np.sin(theta)*np.cos(phi)).real,\
                        (np.sin(theta)*np.sin(phi)).real,\
                        (np.cos(theta)).real]
                
                self[ikp][iband][ispin]['S'] = np.array(sdir)
                self[ikp][iband][ispin]['SA'] = np.array([phi, theta])
                self[ikp][iband][ispin]['SD'] = np.array([naa, nbb, nab])
                #print ikp, iband, ispin, eigenval['E'], sdir

    def create_entry(self, ikpt, iband, ispin):
        
        if iband not in self[ikpt]:
            self[ikpt][iband] = dict()
        if ispin not in self[ikpt][iband]:
            self[ikpt][iband][ispin] = dict()

    def check_kpoint_index(self, ikpt):

        if ikpt >= len(self.kpts):
            error_msg = "Error in Eigenset object : index of k-point is out of range, " + str(ikpt)
            raise IndexError(error_msg)

    def check_spin_index(self, ispin):

        if ispin >= self.spin.nspin:
            error_msg = "Error in Eigenset object : spin dimension is out of range, " + str(ispin)
            raise IndexError(error_msg)

    def kpt_index(self, kpt, fractional=True, ktol=1.e-14):

        if not fractional:
            kpt = np.dot(kpt,lattice.reciprocal_lattice.inv_matrix)
 
        idx = None
        for ikpt, old_kpt in enumerate(self.kpts) :
            if np.linalg.norm(old_kpt-kpt) < ktol:
                idx = ikpt
                break
    
        return idx 

    def shift_bands_indexes(self, shift):

        spin = deepcopy(self.spin)
        eigenset = Eigenset(self.lattice, spin=spin, kpts=self.kpts, wkpts=self.wkpts)  

        for ikpt, iband, ispin, eig in self.eigenvalues_iter():
            eigenset.set_eigenvalue(ikpt, iband+shift, ispin, **eig) 
        
        return eigenset
    
    def add_non_colinear_spin(self):

        if self.spin.nspin == 1:
            spin_shift = [0,1]
        elif self.spin.nspin == 2:
            spin_shift = [0]

        spin = Spinor(spinorb=True)
        eigenset = Eigenset(self.lattice, spin, kpts=self.kpts, wkpts=self.wkpts)

        for ikpt, iband, ispin, eig in self.eigenvalues_iter():
            for ishift in spin_shift:
                eigenset.set_eigenval(eig, ikpt=ikpt, iband=2*iband+ispin+ishift, ispin=0)         
        
        return eigenset
  
    def eigenvalues_matrix(self, label):

        if not hasattr(self,'bands'):
            ee = []
            for ikp in self:
                ek = []
                for iband in self[ikp]:
                    espin = []
                    for ispin in self[ikp][iband]:
                        espin.append(self[ikp][iband][ispin][label])
                    ek.append(np.array(espin))
                ee.append(np.array(ek))
                #print (ikp, ek)
        
            return np.array(ee) 

        else:
            return self.bands

    def select_specific(self, targets, bands=[]):

        selected_eigvals = []
        selected_ikps = []
        selected_bands = []

        for ikp, iband, ispin in targets:

            ee = self[ikp][iband][ispin]['E']
            if ikp not in selected_ikps:
                selected_ikps.append(ikp)
            jkp = selected_ikps.index(ikp)

            selected_bands.append((jkp, iband, ispin))
            selected_eigvals.append(ee)
            
        selected_kpts = np.array(self.kpts)[selected_ikps]
        selected_wkpts = np.array(self.wkpts)[selected_ikps]
        eigenset = Eigenset(self.lattice, spin=self.spin, \
                            kpts=selected_kpts, wkpts=selected_wkpts, \
                            fractional=False)

        if len(bands) == 0:
            for idx, band in enumerate(selected_bands):
                ik, iband, ispin = band
                bands.append(iband) 

        for idx, band in enumerate(selected_bands):
            ik, iband, ispin = band
            eigenset.set_eigenvalue(ik, bands[idx], ispin, E=selected_eigvals[idx])

        return eigenset

    def select(self, klist=[], wklist=[], ksub=[], emin=None, emax=None, \
                         imin=None, imax=None, ktol=1.e-06, bshift=0, reorder=False, \
                         lproj=[], iproj=[], pnorm=False, pmin=None, pmax=None):

        # Projections
        projdic = {'a':0,'s':1,'p':2,'d':3,'f':4}
        if len(lproj) > 0:
            pflag = True
            jproj = []
            for lp in lproj:
                jproj.append(projdic[lp]) 
        elif len(iproj) > 0:
            pflag = True
            jproj = []
            for ip in iproj:
                jproj.append(ip)
        else:
            pflag = False


        if len(klist) >= 1:
            ikps = []
            wkpts = copy(self.wkpts)
            for jkp, newkpt in enumerate(klist):
                for ikp, oldkpt in enumerate(self.kpts):
                   ndiff = np.linalg.norm(np.array(newkpt)-oldkpt)
                   if ndiff <= ktol:
                       ikps.append(ikp)
                       if len(wklist) >= 1:
                           wkpts[ikp] = wklist[jkp]
                       break
        elif len(ksub) >= 1:
            ikps = ksub
            wkpts = copy(self.wkpts)
        else:
            ikps = range(len(self.kpts))
            wkpts = copy(self.wkpts)

        selected_eigvals = []
        selected_ikps = []
        selected_bands = []

        
        if emin != None and emax != None:
            if imin != None and imax != None:
                for ikp in ikps:
                    jband = 0
                    for iband in range(imin,imax+1):
                        selected = False
                        for ispin in self[ikp][iband]:
                            if pflag:
                                if pnorm:
                                    proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])/\
                                           np.sum(self[ikp][iband][ispin]['P'][:,0])  
                                else:
                                    proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])
                                if not (pmin<= proj and proj <=pmax):
                                    continue
                            #ee = self[ikp][iband][ispin]['E']
                            ee = self[ikp][iband][ispin]
                            if (emin <= ee['E'] and ee['E'] <= emax):
                                if ikp not in selected_ikps:
                                    selected_ikps.append(ikp)
                                jkp = selected_ikps.index(ikp)
                                if reorder:
                                    selected_bands.append((jkp, jband, ispin))
                                else:
                                    selected_bands.append((jkp, iband, ispin))
                                selected_eigvals.append(ee)
                                selected = True
                        if selected:
                            jband += 1

            else:
                for ikp in ikps:
                    jband = 0
                    for iband in self[ikp]:
                        selected = False
                        for ispin in self[ikp][iband]:
                            if pflag:
                                if pnorm:
                                    proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])/\
                                           np.sum(self[ikp][iband][ispin]['P'][:,0])  
                                else:
                                    proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])
                                if not (pmin<= proj and proj <=pmax):
                                    continue
                            #ee = self[ikp][iband][ispin]['E']
                            ee = self[ikp][iband][ispin]
                            if (emin <= ee['E'] and ee['E'] <= emax):
                                if ikp not in selected_ikps:
                                    selected_ikps.append(ikp)
                                jkp = selected_ikps.index(ikp)
                                if reorder:
                                    selected_bands.append((jkp, jband, ispin))
                                else:
                                    selected_bands.append((jkp, iband, ispin))
                                selected_eigvals.append(ee)
                                selected = True
                        if selected:
                            jband += 1

        else:
            for jkp, ikp in enumerate(ikps):
                selected_ikps.append(ikp)
                jband = 0
                for iband in range(imin,imax+1):
                    selected = False
                    if iband not in self[ikp]:
                        continue 
                    for ispin in self[ikp][iband]:
                        if pflag:
                            if pnorm:
                                proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])/\
                                       np.sum(self[ikp][iband][ispin]['P'][:,0])  
                            else:
                                proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])
                            if not (pmin<= proj and proj <=pmax):
                                continue
                        #ee = self[ikp][iband][ispin]['E']
                        ee = self[ikp][iband][ispin]
                        if reorder:
                            selected_bands.append((jkp,jband,ispin))
                        else:
                            selected_bands.append((jkp,iband,ispin))
                        selected_eigvals.append(ee)
                        selected = True
                    if selected:
                        jband += 1

        selected_kpts = np.array(self.kpts)[selected_ikps]
        selected_wkpts = np.array(wkpts)[selected_ikps]
        eigenset = Eigenset(self.lattice, spin=self.spin, \
                            kpts=selected_kpts, wkpts=selected_wkpts, \
                            fractional=False)

        for idx, band in enumerate(selected_bands):
            ik, iband, ispin = band
            #eigenset.set_eigenvalue(ik, iband, ispin, E=selected_eigvals[idx])
            eigenset.set_eigenvalue(ik, iband, ispin, **selected_eigvals[idx])

        return eigenset.shift_bands_indexes(bshift)


    def plot_spins_2d_ud(self, iplane, ticks, band, saxes=None, cmap='jet', \
                  pivot='mid', vmin=0., vmax=1., scale=True, log=False):

        import matplotlib.pyplot as plt 

        plane = np.zeros(3)
        plane[iplane] = 1.

        if saxes == None:
            saxes = np.identity(3)

        kpts_up = np.empty([0,2])
        spins_up = np.empty([0,3])
        kpts_dw = np.empty([0,2])
        spins_dw = np.empty([0,3])
        
        for ikp in self:
 
            ckpt = self.kpts[ikp]
            rkpt = np.dot(ckpt,self.lattice.reciprocal_lattice.inv_matrix) 
            nk = np.dot(rkpt,plane)            

            if nk >=ticks[0] and nk <= ticks[1]:

                pk = np.array([ckpt[(iplane-1)%3],ckpt[(iplane+1)%3]])            

                if band in self[ikp]:
                    spin = self[ikp][band][0]['S']
                    if spin[2] >= 0:
                        kpts_up = np.vstack((kpts_up,pk))
                        spins_up = np.vstack((spins_up,spin))
                    else:
                        kpts_dw = np.vstack((kpts_dw,pk))
                        spins_dw = np.vstack((spins_dw,spin))
         
        if scale:
            spins_up[:,:2] = (spins_up[:,:2].T/np.linalg.norm(spins_up[:,:2],axis=-1)).T
            spins_dw[:,:2] = (spins_dw[:,:2].T/np.linalg.norm(spins_dw[:,:2],axis=-1)).T

        plot = plt.quiver(kpts_up[:,0],kpts_up[:,1],spins_up[:,0],spins_up[:,1],spins_up[:,2],pivot=pivot,cmap='Reds')
        plt.quiver(kpts_dw[:,0],kpts_dw[:,1],spins_dw[:,0],spins_dw[:,1],np.abs(spins_dw[:,2]),pivot=pivot,cmap='Blues')

        if log:
           plot.set_norm(LogNorm(vmin=vmin, vmax=vmax))
        else:
           plot.set_clim(vmin=vmin,vmax=vmax)
        
        plt.colorbar()
        plt.axis('equal')


    def plot_spins_2d(self, iplane, ticks, band, saxes=None, cmap='jet', \
                  pivot='mid', vmin=0., vmax=1., scale=True, log=False):

        import matplotlib.pyplot as plt 
        import matplotlib.colors as colors

        plane = np.zeros(3)
        plane[iplane] = 1.

        if saxes == None:
            saxes = np.identity(3)

        kpts = np.empty([0,2])
        spins = np.empty([0,3])
        
        for ikp in self:
 
            ckpt = self.kpts[ikp]
            rkpt = np.dot(ckpt,self.lattice.reciprocal_lattice.inv_matrix) 
            nk = np.dot(rkpt,plane)            

            if nk >=ticks[0] and nk <= ticks[1]:

                pk = np.array([ckpt[(iplane-1)%3],ckpt[(iplane+1)%3]])            

                if band in self[ikp]:
                    kpts = np.vstack((kpts,pk))
                    spin = self[ikp][band][0]['S']
                    spins = np.vstack((spins,spin))

        c = np.linalg.norm(spins[:,:2],axis=-1)
        if scale:
            spins[:,:2] = (spins[:,:2].T/np.linalg.norm(spins[:,:2],axis=-1)).T

        plot = plt.quiver(kpts[:,0],kpts[:,1],spins[:,0],spins[:,1],c,pivot=pivot,cmap=cmap)

        if log:
           plot.set_norm(colors.LogNorm(vmin=vmin, vmax=vmax))
        else:
           plot.set_clim(vmin=vmin,vmax=vmax)
        
        plt.colorbar()
        plt.axis('equal')


    def plot_bands(self, emin=None, emax=None, xmin=None, xmax=None, \
             kshift=0, spincolors=[], eshift=0., xscale=True, kmin=None, kmax=None, \
             klabels=[], vertices=[], vertices_width=1, labelint=False, ylabel=None, \
             show_gap = None, jspin=-1, printlabels=False,\
             delta_bands=[], spin_split=False, bands=[], yfactor=1., **kwargs):

        import matplotlib as mplt
        import matplotlib.pyplot as plt
        #mplt.rcParams['axes.linewidth'] = 2. #set the value globally
        #mplt.rcParams['xtick.major.width'] = 2.
        #mplt.rcParams['ytick.major.width'] = 2.

        # K-points coordinates
        nkpts = self.kpts.shape[0]
        if xscale:
            xaxis = [0.]
            for ik in range(1,nkpts):
                d = np.linalg.norm((self.kpts[ik]-self.kpts[ik-1])*self.wkpts[ik])
                xaxis.append(d+xaxis[ik-1])
            xaxis = np.array(xaxis)-xaxis[kshift]
        else:
            xaxis = np.array(range(nkpts))


        # Plot eigenvalues
        ener = self.eigenvalues_matrix('E')
        nk, nb, ns = ener.shape
        if len(bands) != 0:
            ee = ener[:,bands,:]
        elif len(delta_bands) != 0:
            ee = np.zeros((nk,len(delta_bands),ns))
            if not spin_split:
                for idelta, delta in enumerate(delta_bands):
                    iband, jband = delta
                    ee[:,idelta,:] = ener[:,jband,:]-ener[:,iband,:]
            else:
                for idelta, delta in enumerate(delta_bands):
                    iband, jband = delta
                    for ik in range(nk):
                        if self[ik][iband][0]['S'][2] >= 0.:
                            ee[ik,idelta,:] = ener[ik,iband,:]-ener[ik,jband,:]
                        else: 
                            ee[ik,idelta,:] = ener[ik,jband,:]-ener[ik,iband,:]
        else:
            ee = ener

        if jspin >= 0:
            spinrange = [jspin]
        else:
            spinrange = range(self.spin.nspin)

        for ispin in spinrange:
            if len(spincolors) == self.spin.nspin:
                kwargs['color'] = spincolors[ispin]
            plt.plot(xaxis, ee[:,:,ispin]+eshift,**kwargs) 

        # Plotting range
        if xmin == None:
            xmin = xaxis.min()
        if xmax == None:
            xmax = xaxis.max()
        if emin == None:
            emin = ee.min()
        if emax == None:
            emax = ee.max()

        # Show gap
        if show_gap != None:
            igap0, ispin0, igap1, ispin1  = show_gap 
            gap = ener[:,igap1,ispin1] - ener[:,igap0,ispin0]
            
            #for kk, ek in enumerate(gap):
            #    print kk, ek, ener[kk,igap1,ispin1], ener[kk,igap0,ispin0]

            kgap = gap.argmin()              
            print ('Egap = ', gap[kgap], '(eV)')
            print ('Kgap = kpt[', kgap, '] = ', np.dot(self.kpts[kgap],self.lattice.reciprocal_lattice.inv_matrix))
             
            plt.plot([xaxis[kgap]]*2,[-1.e10,1.e10],'--', color='black')

        # Labelling of the k-point vertices
        if len(vertices) != 0:
            vindex = vertices
        elif hasattr(self,'vertices'):
            vindex = self.vertices
        else:
            vindex = []

        if len(klabels) != 0 and len(vindex) != len(klabels):
            error_msg = "Error in plot_bands: inconsistent number of k-points labels !"
            raise IndexError(error_msg)

        xticks = [] 
        xticks_labels = [] 
        kwargs['color'] = 'black'
        kwargs['linewidth'] = vertices_width
        for ilab, klab in enumerate(klabels):
            xticks.append(xaxis[vindex[ilab]])
            xticks_labels.append(klab) 
            plt.plot([xticks]*2,[emin*1.e10,emax*1.e10], **kwargs)
        if printlabels:
            print (xticks, xticks_labels)
        plt.xticks(xticks, xticks_labels, fontsize=24)

        # Plotting range
        if kmin != None:
            xmin = xaxis[vindex[kmin]]
        if kmax != None:
            xmax = xaxis[vindex[kmax]]

        plt.xlim(xmin, xmax)
        plt.ylim(emin, emax)

        # Label and ticks of the y-axis
        if ylabel == None:
            ylabel = 'Energy  (eV)'

        plt.ylabel(ylabel,fontsize=24)
        plt.yticks(fontsize=18)

        ax = plt.gca()  
        yticks = ax.get_yticks()
        yticklabels = []
        for ylab in yticks:
            ytick = ylab*yfactor
            if labelint:
                yticklabels.append(str(int(ytick)))
            else:
                yticklabels.append(str(ytick))
        ax.set_yticklabels(yticklabels)

    def plot_projected_dos_mask(self, npts=500, smear=0.1, kind='gaussian', \
             emin=None, emax=None, ymin=None, ymax=None,
             spincolors=[], eshift=0., labelint=False, \
             ylabel=None, xlabel=None, efactor=1.,\
             delta_bands=[], bands=[], yfactor=1., **kwargs):

        import matplotlib as mplt
        import matplotlib.pyplot as plt
        mplt.rcParams['axes.linewidth'] = 2. #set the value globally
        mplt.rcParams['xtick.major.width'] = 2.
        mplt.rcParams['ytick.major.width'] = 2.


        # Plot eigenvalues
        ww = {}
        ee = {}


        mask = kwargs.pop('mask', [True])
     
        normalize = False
        if 'rmask' in kwargs:
            rmask = kwargs.pop('rmask',[True])
            normalize = True
         
        if len(bands) != 0:
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []   
                for ikp in self:
                    w = self.wkpts[ikp]
                    for iband in bands:
                        if iband in self[ikp]:
                            etmp.append(self[ikp][iband][ispin]['E'])
                            wtmp.append(w)
                ww[ispin] = np.array(wtmp)
                ee[ispin] = np.array(etmp)
                    
        elif len(delta_bands) != 0:
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []   
                for ikp in self:
                    w = self.wkpts[ikp]
                    for iband, jband in delta_bands:
                        if iband in self[ikp] and  \
                           jband in self[ikp]:
                            etmp.append(self[ikp][jband][ispin]['E']-\
                                         self[ikp][iband][ispin]['E']) 
                            wtmp.append(w)
                ww[ispin] = np.array(wtmp)
                ee[ispin] = np.array(etmp)

        else:
            print (self[0][0][0]['P'].shape)
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []
                for ikp in self:
                    w = self.wkpts[ikp]
                    for iband in self[ikp]:
                        etmp.append(self[ikp][iband][ispin]['E'])
                        if normalize:
                            proj = np.sum(self[ikp][iband][ispin]['P'][mask])/\
                                   np.sum(self[ikp][iband][ispin]['P'][rmask])
                        else:
                            proj = np.sum(self[ikp][iband][ispin]['P'][mask])
                        wtmp.append(w*proj)
                ww[ispin] = np.array(wtmp)
                ee[ispin] = np.array(etmp)

        # Energy range
        if emin == None:
            emin = 1.1*np.array([ee[ispin].min() for ispin in range(self.spin.nspin)]).min()
        if emax == None:
            emax = 1.1*np.array([ee[ispin].max() for ispin in range(self.spin.nspin)]).max()
        epts = np.array([emin + ipt*(emax-emin)/npts for ipt in range(npts)])

        # DOS
        dos = {} 
        for ispin in range(self.spin.nspin):
            dos[ispin] = np.zeros(npts)
            for ipt in range(npts):
                if kind == 'lorentz':
                    dos[ispin][ipt] = \
                    np.sum(np.dot((smear/2.)/((epts[ipt]-ee[ispin])**2 + (smear/2.)**2),ww[ispin]))
                else:
                    dos[ispin][ipt] = \
                    np.sum(np.dot(np.exp(-((epts[ipt]-ee[ispin])**2/smear**2)),ww[ispin])) 

        # Plot
        for ispin in range(self.spin.nspin):
            if len(spincolors) == self.spin.nspin:
                kwargs['color'] = spincolors[ispin]
            plt.plot(epts+eshift, dos[ispin], **kwargs)

        # Plotting range
        if ymin == None:
            ymin = 0.
        if ymax == None:
            ymax = 1.1*np.array([dos[ispin].max() for ispin in range(self.spin.nspin)]).max()
        plt.xlim(emin, emax)
        plt.ylim(ymin, ymax)

        # Label and ticks of the y-axis
        if ylabel == None:
            ylabel = 'DOS '

        plt.ylabel(ylabel,fontsize=16)
        plt.yticks(fontsize=12)

        if xlabel == None:
            xlabel = 'Energy  (eV)'

        plt.xlabel(xlabel,fontsize=16)
        plt.xticks(fontsize=12)

        ax = plt.gca()  
        xticks = ax.get_xticks()
        xticklabels = []
        for xlab in xticks:
            xtick = xlab*efactor
            if labelint:
                xticklabels.append(str(int(xtick)))
            else:
                xticklabels.append(str(xtick))
        ax.set_xticklabels(xticklabels)

        return epts+eshift, dos


    def plot_projected_dos(self, npts=500, smear=0.1, kind='gaussian', \
             emin=None, emax=None, ymin=None, ymax=None,
             spincolors=[], eshift=0., labelint=False, \
             ylabel=None, xlabel=None, efactor=1.,\
             delta_bands=[], bands=[], yfactor=1., \
             atoms=[], iproj=[], lproj=[], sproj=None, **kwargs):

        import matplotlib.pyplot as plt

        
        # Projections
        projdic = {'a':0,'s':1,'p':2,'d':3,'f':4}
        jproj = []
        if len(lproj) > 0:
            for lp in lproj:
                jproj.append(projdic[lp]) 
        elif len(iproj) > 0:
            for ip in iproj:
                jproj.append(ip)
        else:
            jproj.append(0)



        # Plot eigenvalues
        ww = {}
        ee = {}

        if len(bands) != 0:
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []   
                for ikp in self:
                    w = self.wkpts[ikp]
                    for iband in bands:
                        if iband in self[ikp]:
                            etmp.append(self[ikp][iband][ispin]['E'])
                            wtmp.append(w)
                ww[ispin] = np.array(wtmp)
                ee[ispin] = np.array(etmp)
                    
        elif len(delta_bands) != 0:
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []   
                for ikp in self:
                    w = self.wkpts[ikp]
                    for iband, jband in delta_bands:
                        if iband in self[ikp] and  \
                           jband in self[ikp]:
                            etmp.append(self[ikp][jband][ispin]['E']-\
                                         self[ikp][iband][ispin]['E']) 
                            wtmp.append(w)
                ww[ispin] = np.array(wtmp)
                ee[ispin] = np.array(etmp)

        else:
            print (self[0][0][0]['P'].shape)
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []
                for ikp in self:
                    w = self.wkpts[ikp]
                    for iband in self[ikp]:
                        etmp.append(self[ikp][iband][ispin]['E'])
                        if len(atoms) == 0:
                            pp = np.sum(np.sum(self[ikp][iband][ispin]['P'][:,jproj],axis=-1),axis=-1)/\
                                 np.sum(self[ikp][iband][ispin]['P'][:,0])
                        else:
                            pp = np.sum(np.sum(self[ikp][iband][ispin]['P'][atoms,jproj],axis=-1),axis=-1)/\
                                 np.sum(self[ikp][iband][ispin]['P'][:,0])
                        wtmp.append(w*pp)
                ww[ispin] = np.array(wtmp)
                ee[ispin] = np.array(etmp)

        # Energy range
        if emin == None:
            emin = 1.1*np.array([ee[ispin].min() for ispin in range(self.spin.nspin)]).min()
        if emax == None:
            emax = 1.1*np.array([ee[ispin].max() for ispin in range(self.spin.nspin)]).max()
        epts = np.array([emin + ipt*(emax-emin)/npts for ipt in range(npts)])

        # DOS
        dos = {} 
        for ispin in range(self.spin.nspin):
            dos[ispin] = np.zeros(npts)
            for ipt in range(npts):
                if kind == 'lorentz':
                    dos[ispin][ipt] = \
                    np.sum(np.dot((smear/2.)/((epts[ipt]-ee[ispin])**2 + (smear/2.)**2),ww[ispin]))
                else:
                    dos[ispin][ipt] = \
                    np.sum(np.dot(np.exp(-((epts[ipt]-ee[ispin])**2/smear**2)),ww[ispin])) 

        # Plot
        for ispin in range(self.spin.nspin):
            if len(spincolors) == self.spin.nspin:
                kwargs['color'] = spincolors[ispin]
            plt.plot(epts+eshift, dos[ispin], **kwargs)

        # Plotting range
        if ymin == None:
            ymin = 0.
        if ymax == None:
            ymax = 1.1*np.array([dos[ispin].max() for ispin in range(self.spin.nspin)]).max()
        plt.xlim(emin, emax)
        plt.ylim(ymin, ymax)

        # Label and ticks of the y-axis
        if ylabel == None:
            ylabel = 'DOS '

        plt.ylabel(ylabel,fontsize=16)
        plt.yticks(fontsize=12)

        if xlabel == None:
            xlabel = 'Energy  (eV)'

        plt.xlabel(xlabel,fontsize=16)
        plt.xticks(fontsize=12)

        ax = plt.gca()  
        xticks = ax.get_xticks()
        xticklabels = []
        for xlab in xticks:
            xtick = xlab*efactor
            if labelint:
                xticklabels.append(str(int(xtick)))
            else:
                xticklabels.append(str(xtick))
        ax.set_xticklabels(xticklabels)


    def compute_dos4(self, npts=500, smear=0.1, kind='gaussian', \
             emin=None, emax=None, wmin=None, wmax=None, bands=[], kpts=[], interpol=False, w=None):

        from scipy.interpolate import interp1d

        if hasattr(self,'eigenarray'):
            etmp = self.eigenarray
        else:
            self.eigenarray = self.eigenvalues_matrix('E')
            etmp = self.eigenarray

        #print (etmp.shape)

        if len(kpts) == 0:
            if len(bands) == 0:
                ee = etmp
            else:
                ee = etmp[:,bands,:]
            ww = np.empty(ee.shape[:2])
            for ib in range(ee.shape[1]): 
                ww[:,ib] = self.wkpts

        else:
            if len(bands) == 0:
                ee = etmp[kpts,:,:]
            else:
                ee0 = etmp[kpts,:,:]
                ee = ee0[:,bands,:]
            ww = np.empty(ee.shape[:2])
            for ib in range(ee.shape[1]): 
                ww[:,ib] = self.wkpts[kpts]

        if w != None:
             ww[:,:] = w
        else:
             ww[:,:] = 1.

        # Energy range
        if emin == None:
            emin = 1.1*np.array([ee[ispin].min() for ispin in range(self.spin.nspin)]).min()
        if emax == None:
            emax = 1.1*np.array([ee[ispin].max() for ispin in range(self.spin.nspin)]).max()
        epts = np.array([emin + ipt*(emax-emin)/npts for ipt in range(npts)])

        # DOS
        dos = np.zeros((len(epts),self.spin.nspin))
        for ispin in range(self.spin.nspin):
            mask = (ee > emin) & (ee<emax)
            mask[:,:,ispin-1] = False

            for ipt in range(npts):
                dos[ipt,ispin] = np.dot(np.exp(-((epts[ipt]-ee[mask]/smear)**2/2)), ww[mask[:,:,ispin]])

        if interpol:
            f = interp1d(epts, dos.T/(smear*(2*np.pi)**0.5))
            return f
        else:
            return epts, np.array(dos)/(smear*(2*np.pi)**0.5)




    def compute_dos3(self, npts=500, smear=0.1, kind='gaussian', \
             emin=None, emax=None, wmin=None, wmax=None, bands=[], kpts=[], interpol=False, w=None):

        from scipy.interpolate import interp1d

        if hasattr(self,'eigenarray'):
            etmp = self.eigenarray
        else:
            self.eigenarray = self.eigenvalues_matrix('E')
            etmp = self.eigenarray

        print (etmp.shape)

        if len(kpts) == 0:
            if len(bands) == 0:
                ee = etmp
            else:
                ee = etmp[:,bands,:]
            ww = np.empty(ee.shape[:2])
            for ib in range(ee.shape[1]): 
                ww[:,ib] = self.wkpts

        else:
            if len(bands) == 0:
                ee = etmp[kpts,:,:]
            else:
                ee0 = etmp[kpts,:,:]
                ee = ee0[:,bands,:]
            ww = np.empty(ee.shape[:2])
            for ib in range(ee.shape[1]): 
                ww[:,ib] = self.wkpts[kpts]

        if w != None:
             ww[:,:] = w

        # Energy range
        if emin == None:
            emin = 1.1*np.array([ee[ispin].min() for ispin in range(self.spin.nspin)]).min()
        if emax == None:
            emax = 1.1*np.array([ee[ispin].max() for ispin in range(self.spin.nspin)]).max()
        epts = np.array([emin + ipt*(emax-emin)/npts for ipt in range(npts)])

        # DOS
        dos = np.zeros((len(epts),self.spin.nspin))
        for ispin in range(self.spin.nspin):
            mask = (ee > emin) & (ee<emax)
            mask[:,:,ispin-1] = False

            for ipt in range(npts):
                dos[ipt,ispin] = np.dot(np.exp(-((epts[ipt]-ee[mask])**2/smear**2)), ww[mask[:,:,ispin]])

        if interpol:
            f = interp1d(epts, dos.T)
            return f
        else:
            return epts, np.array(dos)



    def compute_dos2(self, npts=500, smear=0.1, kind='gaussian', \
             emin=None, emax=None, wmin=None, wmax=None, bands=[], kpts=[], interpol=False):

        from scipy.interpolate import interp1d

        etmp =  self.eigenvalues_matrix('E')
        if len(kpts) == 0:
            if len(bands) == 0:
                ee = etmp
            else:
                ee = etmp[:,bands,:]
            ww = np.empty(ee.shape[:2])
            for ib in range(ee.shape[1]): 
                ww[:,ib] = self.wkpts

        else:
            if len(bands) == 0:
                ee = etmp[kpts,:,:]
            else:
                ee0 = etmp[kpts,:,:]
                ee = ee0[:,bands,:]
            ww = np.empty(ee.shape[:2])
            for ib in range(ee.shape[1]): 
                ww[:,ib] = self.wkpts[kpts]

        # Energy range
        if emin == None:
            emin = 1.1*np.array([ee[ispin].min() for ispin in range(self.spin.nspin)]).min()
        if emax == None:
            emax = 1.1*np.array([ee[ispin].max() for ispin in range(self.spin.nspin)]).max()
        epts = np.array([emin + ipt*(emax-emin)/npts for ipt in range(npts)])

        # DOS
        dos = np.zeros((len(epts),self.spin.nspin))
        for ispin in range(self.spin.nspin):
            mask = (ee > emin) & (ee<emax)
            mask[:,:,ispin-1] = False
            for ipt in range(npts):
                dos[ipt,ispin] = np.dot(np.exp(-((epts[ipt]-ee[mask])**2/smear**2)), ww[mask[:,:,ispin]])

        if interpol:
            f = interp1d(epts, dos.T)
            return f
        else:
            return epts, np.array(dos)



    def compute_dos(self, npts=500, smear=0.1, kind='gaussian', \
             emin=None, emax=None, wmin=None, wmax=None, bands=[], kpts=[], interpol=False):

        from scipy.interpolate import interp1d

        ww = []
        ee = []

        if len(kpts) == 0:
            kpts = list(self.keys())  

        if len(bands) != 0:
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []   
                for ikp in kpts:
                    w = self.wkpts[ikp]
                    for iband in bands:
                        if iband in self[ikp]:
                            if self[ikp][iband][ispin]['E'] < wmax and \
                               self[ikp][iband][ispin]['E'] > wmin :
                                etmp.append(self[ikp][iband][ispin]['E'])
                                wtmp.append(w)
                ww.append(np.array(wtmp))
                ee.append(np.array(etmp))
        else:
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []   
                for ikp in kpts:
                    w = self.wkpts[ikp]
                    for iband in self[ikp]:
                            if self[ikp][iband][ispin]['E'] < wmax and \
                               self[ikp][iband][ispin]['E'] > wmin :
                                etmp.append(self[ikp][iband][ispin]['E'])
                                wtmp.append(w)
                ww.append(np.array(wtmp))
                ee.append(np.array(etmp))
        
        # Energy range
        if emin == None:
            emin = 1.1*np.array([ee[ispin].min() for ispin in range(self.spin.nspin)]).min()
        if emax == None:
            emax = 1.1*np.array([ee[ispin].max() for ispin in range(self.spin.nspin)]).max()
        epts = np.array([emin + ipt*(emax-emin)/npts for ipt in range(npts)])

        # DOS
        dos = []
        for ispin in range(self.spin.nspin):
            dos.append(np.zeros(npts))
            for ipt in range(npts):
                if kind == 'lorentz':
                    dos[ispin][ipt] = \
                    np.sum(np.dot((smear/2.)/((epts[ipt]-ee[ispin])**2 + (smear/2.)**2),ww[ispin]))
                else:
                    dos[ispin][ipt] = \
                    np.sum(np.dot(np.exp(-((epts[ipt]-ee[ispin])**2/smear**2)),ww[ispin]))

        if interpol:
            f = interp1d(epts, dos)
            return f
        else:
            return epts, np.array(dos)


    def plot_dos(self, npts=500, smear=0.1, kind='gaussian', \
             emin=None, emax=None, ymin=None, ymax=None,
             spincolors=[], eshift=0., labelint=False, \
             ylabel=None, xlabel=None, efactor=1.,\
             delta_bands=[], bands=[], yfactor=1., **kwargs):

        import matplotlib.pyplot as plt

        # Plot eigenvalues
        ww = {}
        ee = {}

        if len(bands) != 0:
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []   
                for ikp in self:
                    w = self.wkpts[ikp]
                    for iband in bands:
                        if iband in self[ikp]:
                            etmp.append(self[ikp][iband][ispin]['E'])
                            wtmp.append(w)
                ww[ispin] = np.array(wtmp)
                ee[ispin] = np.array(etmp)
                    
        elif len(delta_bands) != 0:
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []   
                for ikp in self:
                    w = self.wkpts[ikp]
                    for iband, jband in delta_bands:
                        if iband in self[ikp] and  \
                           jband in self[ikp]:
                            etmp.append(self[ikp][jband][ispin]['E']-\
                                         self[ikp][iband][ispin]['E']) 
                            wtmp.append(w)
                ww[ispin] = np.array(wtmp)
                ee[ispin] = np.array(etmp)

        else:
            for ispin in range(self.spin.nspin):
                wtmp = []   
                etmp = []   
                for ikp in self:
                    w = self.wkpts[ikp]
                    for iband in self[ikp]:
                        etmp.append(self[ikp][iband][ispin]['E'])
                        wtmp.append(w)
                ww[ispin] = np.array(wtmp)
                ee[ispin] = np.array(etmp)

        # Energy range
        if emin == None:
            emin = 1.1*np.array([ee[ispin].min() for ispin in range(self.spin.nspin)]).min()
        if emax == None:
            emax = 1.1*np.array([ee[ispin].max() for ispin in range(self.spin.nspin)]).max()
        epts = np.array([emin + ipt*(emax-emin)/npts for ipt in range(npts)])

        # DOS
        dos = {} 
        for ispin in range(self.spin.nspin):
            dos[ispin] = np.zeros(npts)
            for ipt in range(npts):
                if kind == 'lorentz':
                    dos[ispin][ipt] = \
                    np.sum(np.dot((smear/2.)/((epts[ipt]-ee[ispin])**2 + (smear/2.)**2),ww[ispin]))
                else:
                    dos[ispin][ipt] = \
                    np.sum(np.dot(np.exp(-((epts[ipt]-ee[ispin])**2/smear**2)),ww[ispin])) 

        # Plot
        for ispin in range(self.spin.nspin):
            if len(spincolors) == self.spin.nspin:
                kwargs['color'] = spincolors[ispin]
            plt.plot(epts+eshift, dos[ispin], **kwargs)

        # Plotting range
        if ymin == None:
            ymin = 0.
        if ymax == None:
            ymax = 1.1*np.array([dos[ispin].max() for ispin in range(self.spin.nspin)]).max()
        plt.xlim(emin, emax)
        plt.ylim(ymin, ymax)

        # Label and ticks of the y-axis
        if ylabel == None:
            ylabel = 'DOS '

        plt.ylabel(ylabel,fontsize=16)
        plt.yticks(fontsize=12)

        if xlabel == None:
            xlabel = 'Energy  (eV)'

        plt.xlabel(xlabel,fontsize=16)
        plt.xticks(fontsize=12)

        ax = plt.gca()  
        xticks = ax.get_xticks()
        xticklabels = []
        for xlab in xticks:
            xtick = xlab*efactor
            if labelint:
                xticklabels.append(str(int(xtick)))
            else:
                xticklabels.append(str(xtick))
        ax.set_xticklabels(xticklabels)


    def plot_eigenvalues(self, emin=None, emax=None, xmin=None, xmax=None, \
             kshift=0, spincolors=[], eshift=0., xscale=True, \
             klabels=[], vertices=[], labelint=False, ylabel=None, \
             delta_bands=[], bands=[], yfactor=1., **kwargs):

        import matplotlib.pyplot as plt

        # K-points coordinates
        nkpts = self.kpts.shape[0]
        if xscale:
            xaxis = [0.]
            for ik in range(1,nkpts):
                d = np.linalg.norm((self.kpts[ik]-self.kpts[ik-1])*self.wkpts[ik])
                xaxis.append(d+xaxis[ik-1])
            xaxis = np.array(xaxis)-xaxis[kshift]
        else:
            xaxis = np.array(range(nkpts))
            xaxis = xaxis - xaxis[kshift]

        # Plot eigenvalues
        xx = {}
        ee = {}

        if len(bands) != 0:
            for ispin in range(self.spin.nspin):
                xtmp = []   
                etmp = []   
                for ikp in self:
                    for iband in bands:
                        if iband in self[ikp]:
                            etmp.append(self[ikp][iband][ispin]['E'])
                            xtmp.append(xaxis[ikp])
                xx[ispin] = np.array(xtmp)
                ee[ispin] = np.array(etmp)
                    
        elif len(delta_bands) != 0:
            for ispin in range(self.spin.nspin):
                xtmp = []   
                etmp = []   
                for ikp in self:
                    for iband, jband in delta_bands:
                        if iband in self[ikp] and  \
                           jband in self[ikp]:
                            etmp.append(self[ikp][jband][ispin]['E']-\
                                         self[ikp][iband][ispin]['E']) 
                            xtmp.append(xaxis[ikp])
                xx[ispin] = np.array(xtmp)
                ee[ispin] = np.array(etmp)

        else:
            for ispin in range(self.spin.nspin):
                xtmp = []   
                etmp = []   
                for ikp in self:
                    for iband in self[ikp]:
                        etmp.append(self[ikp][iband][ispin]['E'])
                        xtmp.append(xaxis[ikp])
                xx[ispin] = np.array(xtmp)
                ee[ispin] = np.array(etmp)
       

 
        for ispin in range(self.spin.nspin):
            if len(spincolors) == self.spin.nspin:
                kwargs['color'] = spincolors[ispin]
            plt.scatter(xx[ispin], ee[ispin]+eshift, **kwargs)

        # Plotting range
        if xmin == None:
            xmin = xaxis.min()
        if xmax == None:
            xmax = xaxis.max()
        if emin == None:
            emin = np.array([ee[ispin].min() for ispin in range(self.spin.nspin)]).min()
        if emax == None:
            emax = np.array([ee[ispin].max() for ispin in range(self.spin.nspin)]).max()

        # Labelling of the k-point vertices
        if len(vertices) != 0:
            vindex = vertices
        elif hasattr(self,'vertices'):
            vindex = self.vertices
        else:
            vindex = []

        if len(klabels) != 0 and len(vindex) != len(klabels):
            error_msg = "Error in plot_bands: inconsistent number of k-points labels !"
            raise IndexError(error_msg)

        xticks = [] 
        xticks_labels = [] 
        for ilab, klab in enumerate(klabels):
            xticks.append(xaxis[vindex[ilab]])
            xticks_labels.append(klab) 
            plt.plot([xticks]*2,[emin,emax],'-', color='black')
        plt.xticks(xticks, xticks_labels, fontsize=24)

        # Plotting range
        plt.xlim(xmin, xmax)
        plt.ylim(emin, emax)

        # Label and ticks of the y-axis
        if ylabel == None:
            ylabel = 'Energy  (eV)'

        plt.ylabel(ylabel,fontsize=24)
        plt.yticks(fontsize=18)

        ax = plt.gca()  
        yticks = ax.get_yticks()
        yticklabels = []
        for ylab in yticks:
            ytick = ylab*yfactor
            if labelint:
                yticklabels.append(str(int(ytick)))
            else:
                yticklabels.append(str(ytick))
        ax.set_yticklabels(yticklabels)


    def plot_projected_eigenvalues(self, emin=None, emax=None, xmin=None, xmax=None, \
             kshift=0, spincolors=[], spincmaps=[], eshift=0., xscale=True, colorbar=False, clabel=None,\
             atoms=[], lproj=[], jspin=-1, iproj=[], normalize=False, varsize=False, size=20, \
             klabels=[], vertices=[], labelint=False, ylabel=None, kmin=None, kmax=None,\
             delta_bands=[], bands=[], yfactor=1., **kwargs):

        import matplotlib as mplt
        import matplotlib.pyplot as plt
        mplt.rcParams['axes.linewidth'] = 2. #set the value globally
        mplt.rcParams['xtick.major.width'] = 2.
        mplt.rcParams['ytick.major.width'] = 2.

        # K-points coordinates
        nkpts = self.kpts.shape[0]
        if xscale:
            xaxis = [0.]
            for ik in range(1,nkpts):
                d = np.linalg.norm((self.kpts[ik]-self.kpts[ik-1])*self.wkpts[ik])
                xaxis.append(d+xaxis[ik-1])
            xaxis = np.array(xaxis)-xaxis[kshift]
        else:
            xaxis = np.array(range(nkpts))
            xaxis = xaxis - xaxis[kshift]

        # Projections
        projdic = {'a':0,'s':1,'p':2,'d':3,'f':4}
        jproj = []
        if len(lproj) > 0:
            for lp in lproj:
                jproj.append(projdic[lp]) 
        elif len(iproj) > 0:
            for ip in iproj:
                jproj.append(ip)
        else:
            jproj.append(0)


        # Plot eigenvalues
        xx = {}
        ee = {}
        pp = {}

        if jspin >= 0:
            spinrange = [jspin]
        else:
            spinrange = range(self.spin.nspin)

        for ispin in spinrange:
            xtmp = []   
            etmp = [] 
            ptmp = []  

            if len(bands) != 0:
                for ikp in self:
                    for iband in bands:
                        if iband in self[ikp]:
                            if len(atoms) == 0:
                                if normalize:
                                    proj = np.sum(np.sum(self[ikp][iband][ispin]['P'][:,jproj],axis=-1),axis=-1)/\
                                           np.sum(self[ikp][iband][ispin]['P'][:,0])
                                else:
                                    proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])
                            else:
                                if normalize:
                                    proj = np.sum(np.sum(self[ikp][iband][ispin]['P'][atoms,jproj],axis=-1),axis=-1)/\
                                           np.sum(self[ikp][iband][ispin]['P'][atoms,0])
                                else:
                                    proj = np.sum(np.sum(self[ikp][iband][ispin]['P'][atoms,jproj],axis=-1),axis=-1)
                            ptmp.append(proj)
                            etmp.append(self[ikp][iband][ispin]['E'])
                            xtmp.append(xaxis[ikp])

            elif len(delta_bands) != 0:
                for ikp in self:
                    for iband, jband in delta_bands:
                        if iband in self[ikp] and  \
                           jband in self[ikp]:
                            if len(atoms) == 0:
                                if normalize:
                                    proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])/\
                                           np.sum(self[ikp][iband][ispin]['P'][:,0])
                                else:
                                    proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])
                            else:
                                if normalize:
                                    proj = np.sum(self[ikp][iband][ispin]['P'][atoms,jproj])/\
                                           np.sum(self[ikp][iband][ispin]['P'][:,0])
                                else:
                                    proj = np.sum(self[ikp][iband][ispin]['P'][atoms,jproj])
                            ptmp.append(proj)
                            etmp.append(self[ikp][jband][ispin]['E']-\
                                         self[ikp][iband][ispin]['E']) 
                            xtmp.append(xaxis[ikp])

            else:
                for ikp in self:
                    for iband in self[ikp]:
                
                        if len(atoms) == 0:
                            if normalize:
                                proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])/\
                                       np.sum(self[ikp][iband][ispin]['P'][:,0])
                            else:
                                proj = np.sum(self[ikp][iband][ispin]['P'][:,jproj])
                        else:
                            if normalize:
                                proj = np.sum(self[ikp][iband][ispin]['P'][atoms,jproj])/\
                                       np.sum(self[ikp][iband][ispin]['P'][:,0])
                            else:
                                proj = np.sum(self[ikp][iband][ispin]['P'][atoms,jproj])

                        ptmp.append(proj)
                        etmp.append(self[ikp][iband][ispin]['E'])
                        xtmp.append(xaxis[ikp])

            ee[ispin] = np.array(etmp)
            xx[ispin] = np.array(xtmp)
            pp[ispin] = np.array(ptmp)
                    
        if 'vmin' not in kwargs:
            vmin = np.array([pp[ispin].min() for ispin in pp]).min()
            kwargs['vmin'] = vmin
        else:
            vmin = kwargs['vmin']
        if 'vmax' not in kwargs:
            vmax = np.array([pp[ispin].max() for ispin in pp]).max()
            kwargs['vmax'] = vmax
        else:
            vmax = kwargs['vmax']

        for ispin in spinrange:

            if varsize:
                kwargs['s'] = (size*(pp[ispin]-vmin)/(vmax-vmin+1e-10))
            else:
                kwargs['s'] = size


            if len(spincmaps) == self.spin.nspin:
                kwargs['cmap'] = spincmaps[ispin]
                kwargs['c'] = pp[ispin]
            elif len(spincolors) == self.spin.nspin:
                kwargs['color'] = spincolors[ispin]
            else:
                kwargs['c'] = pp[ispin]
                if 'cmap' not in kwargs:
                    kwargs['cmap'] = 'jet'

            sc = plt.scatter(xx[ispin], ee[ispin]+eshift, **kwargs)
            if colorbar:
                if clabel == None:
                    clabel = 'Projection  (%)'
                cb = plt.colorbar(sc)
                cb.ax.tick_params(labelsize=14) 
                cb.set_label(clabel, fontsize=18)
        

        # Plotting range
        if xmin == None:
            xmin = xaxis.min()
        if xmax == None:
            xmax = xaxis.max()
        if emin == None:
            emin = np.array([ee[ispin].min() for ispin in range(self.spin.nspin)]).min()
        if emax == None:
            emax = np.array([ee[ispin].max() for ispin in range(self.spin.nspin)]).max()

        # Labelling of the k-point vertices
        if len(vertices) != 0:
            vindex = vertices
        elif hasattr(self,'vertices'):
            vindex = self.vertices
        else:
            vindex = []

        if len(klabels) != 0 and len(vindex) != len(klabels):
            error_msg = "Error in plot_bands: inconsistent number of k-points labels !"
            raise IndexError(error_msg)

        xticks = [] 
        xticks_labels = [] 
        for ilab, klab in enumerate(klabels):
            xticks.append(xaxis[vindex[ilab]])
            xticks_labels.append(klab) 
            plt.plot([xticks]*2,[emin,emax],'-', color='black')
        plt.xticks(xticks, xticks_labels, fontsize=24)

        # Plotting range
        if kmin != None:
            xmin = xaxis[vindex[kmin]]
        if kmax != None:
            xmax = xaxis[vindex[kmax]]

        plt.xlim(xmin, xmax)
        plt.ylim(emin, emax)

        # Label and ticks of the y-axis
        if ylabel == None:
            ylabel = 'Energy  (eV)'

        plt.ylabel(ylabel,fontsize=24)
        plt.yticks(fontsize=18)

        ax = plt.gca()  
        yticks = ax.get_yticks()
        yticklabels = []
        for ylab in yticks:
            ytick = ylab*yfactor
            if labelint:
                yticklabels.append(str(int(ytick)))
            else:
                yticklabels.append(str(ytick))
        ax.set_yticklabels(yticklabels)

    def plot_projected_eigenvalues_mask(self, emin=None, emax=None, xmin=None, xmax=None, \
             kshift=0, spincolors=[], spincmaps=[], eshift=0., xscale=True, colorbar=False, clabel=None,\
             jspin=-1, varsize=False, size=20, \
             klabels=[], vertices=[], labelint=False, ylabel=None, kmin=None, kmax=None,\
             delta_bands=[], bands=[], yfactor=1., **kwargs):

        import matplotlib as mplt
        import matplotlib.pyplot as plt
        mplt.rcParams['axes.linewidth'] = 2. #set the value globally
        mplt.rcParams['xtick.major.width'] = 2.
        mplt.rcParams['ytick.major.width'] = 2.

        # K-points coordinates
        nkpts = self.kpts.shape[0]
        if xscale:
            xaxis = [0.]
            for ik in range(1,nkpts):
                d = np.linalg.norm((self.kpts[ik]-self.kpts[ik-1])*self.wkpts[ik])
                xaxis.append(d+xaxis[ik-1])
            xaxis = np.array(xaxis)-xaxis[kshift]
        else:
            xaxis = np.array(range(nkpts))
            xaxis = xaxis - xaxis[kshift]

        # Plot eigenvalues
        xx = {}
        ee = {}
        pp = {}

        if jspin >= 0:
            spinrange = [jspin]
        else:
            spinrange = range(self.spin.nspin)

        mask = kwargs.pop('mask', [True])
     
        normalize = False
        if 'rmask' in kwargs:
            rmask = kwargs.pop('rmask',[True])
            normalize = True
         

        for ispin in spinrange:
            xtmp = []   
            etmp = [] 
            ptmp = []  

            if len(bands) != 0:
                for ikp in self:
                    for iband in bands:
                        if iband in self[ikp]:
                            if normalize:
                                proj = np.sum(np.sum(self[ikp][iband][ispin]['P'][mask],axis=-1),axis=-1)/\
                                       np.sum(self[ikp][iband][ispin]['P'][rmask])
                            else:
                                proj = np.sum(np.sum(self[ikp][iband][ispin]['P'][mask],axis=-1),axis=-1)
                            ptmp.append(proj)
                            etmp.append(self[ikp][iband][ispin]['E'])
                            xtmp.append(xaxis[ikp])

            elif len(delta_bands) != 0:
                for ikp in self:
                    for iband, jband in delta_bands:
                        if iband in self[ikp] and  \
                           jband in self[ikp]:
                            if normalize:
                                proj = np.sum(self[ikp][iband][ispin]['P'][mask])/\
                                       np.sum(self[ikp][iband][ispin]['P'][rmask])
                            else:
                                proj = np.sum(self[ikp][iband][ispin]['P'][mask])
                            ptmp.append(proj)
                            etmp.append(self[ikp][jband][ispin]['E']-\
                                         self[ikp][iband][ispin]['E']) 
                            xtmp.append(xaxis[ikp])

            else:
                for ikp in self:
                    for iband in self[ikp]:
                        if normalize:
                            proj = np.sum(self[ikp][iband][ispin]['P'][mask])/\
                                   np.sum(self[ikp][iband][ispin]['P'][rmask])
                        else:
                            proj = np.sum(self[ikp][iband][ispin]['P'][mask])

                        ptmp.append(proj)
                        etmp.append(self[ikp][iband][ispin]['E'])
                        xtmp.append(xaxis[ikp])

            ee[ispin] = np.array(etmp)
            xx[ispin] = np.array(xtmp)
            pp[ispin] = np.array(ptmp)
                    
        if 'vmin' not in kwargs:
            vmin = np.array([pp[ispin].min() for ispin in pp]).min()
            kwargs['vmin'] = vmin
        else:
            vmin = kwargs['vmin']
        if 'vmax' not in kwargs:
            vmax = np.array([pp[ispin].max() for ispin in pp]).max()
            kwargs['vmax'] = vmax
        else:
            vmax = kwargs['vmax']

        for ispin in spinrange:

            if varsize:
                kwargs['s'] = (size*(pp[ispin]-vmin)/(vmax-vmin+1e-10))
            else:
                kwargs['s'] = size


            if len(spincmaps) == self.spin.nspin:
                kwargs['cmap'] = spincmaps[ispin]
                kwargs['c'] = pp[ispin]
            elif len(spincolors) == self.spin.nspin:
                kwargs['color'] = spincolors[ispin]
            else:
                kwargs['c'] = pp[ispin]
                if 'cmap' not in kwargs:
                    kwargs['cmap'] = 'jet'

            sc = plt.scatter(xx[ispin], ee[ispin]+eshift, **kwargs)
            if colorbar:
                if clabel == None:
                    clabel = 'Projection  (%)'
                cb = plt.colorbar(sc)
                cb.ax.tick_params(labelsize=14) 
                cb.set_label(clabel, fontsize=18)
        

        # Plotting range
        if xmin == None:
            xmin = xaxis.min()
        if xmax == None:
            xmax = xaxis.max()
        if emin == None:
            emin = np.array([ee[ispin].min() for ispin in range(self.spin.nspin)]).min()
        if emax == None:
            emax = np.array([ee[ispin].max() for ispin in range(self.spin.nspin)]).max()

        # Labelling of the k-point vertices
        if len(vertices) != 0:
            vindex = vertices
        elif hasattr(self,'vertices'):
            vindex = self.vertices
        else:
            vindex = []

        if len(klabels) != 0 and len(vindex) != len(klabels):
            error_msg = "Error in plot_bands: inconsistent number of k-points labels !"
            raise IndexError(error_msg)

        xticks = [] 
        xticks_labels = [] 
        for ilab, klab in enumerate(klabels):
            xticks.append(xaxis[vindex[ilab]])
            xticks_labels.append(klab) 
            plt.plot([xticks]*2,[emin,emax],'-', color='black')
        plt.xticks(xticks, xticks_labels, fontsize=24)

        # Plotting range
        if kmin != None:
            xmin = xaxis[vindex[kmin]]
        if kmax != None:
            xmax = xaxis[vindex[kmax]]

        plt.xlim(xmin, xmax)
        plt.ylim(emin, emax)

        # Label and ticks of the y-axis
        if ylabel == None:
            ylabel = 'Energy  (eV)'

        plt.ylabel(ylabel,fontsize=24)
        plt.yticks(fontsize=18)

        ax = plt.gca()  
        yticks = ax.get_yticks()
        yticklabels = []
        for ylab in yticks:
            ytick = ylab*yfactor
            if labelint:
                yticklabels.append(str(int(ytick)))
            else:
                yticklabels.append(str(ytick))
        ax.set_yticklabels(yticklabels)

