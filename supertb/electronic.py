"""
This module defines the classes and functions that enables to compute single 
particle hamiltonians and eigenstates from interaction parameters 
(e.g. tight-binding or Slatter-Koster).
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
import re

from multiprocessing import Pool   

class ElectronicStructure(object):
    """
    An object that gathers what is needed to compute an electronic structure
    at the tight-binding level. 

    .. attribute:: graph

       The StructureGraph object use to determine the geometry of the system as well 
       as the connectivity of the tight-binding Hamiltonian.
 
    .. attribute:: lattice

       The lattice defining the periodicity of the system.

    .. attribute:: table

       The AtomicTable listing the numerical atoms present in the system.
       
    .. attribute:: spin

       The Spinor object ruling the dimensionality of the spin manifold

    .. attribute:: basis

       The AtomicBasisSet used to expand the Hamiltonian

    .. attribute:: basis

       A copy of the StructureGraph provided used to store the computed blocks 
       of the Hamiltonian and Overlap matrices 

    .. attribute:: inter

       A dictionary of all the user provided interactions (i.e. the place where 
       all the Slater-Koster parameters are stored). 

    .. attribute:: spacegroup

       The computed spacegroup of the system

    """

    def __init__(self, graph, table, spinpol=False, spinorb=False, spacegroup=None, **kwargs):
        """
        Creates an Electronic Structure object.

        Args:
            graph: (StructureGraph object)
                Structure and connectivity of the system under consideration.
            table: (AtomicTable object)     
                Definition of the numerical atoms present in the system.
            spinpol: (Boolean)
                Determines wheter the system is spin polarized or not.
                Default: False.
            spinorb: (Boolean)
                Determines wheter the system is described by means of spinor
                wavefunctions or not, (i.e. does the description account for 
                the spin-orbit coupling).
                Default: False.
            spacegroup: (Spacegroup object)
                Spacegroup of the system. 
                Default: None.
            **kwargs: 
                List of user provided interaction provided as keyword arguments.
        """

        self.graph = graph
        self.lattice = graph.lattice
        self.table = table
        self.spin = Spinor(spinpol=spinpol,spinorb=spinorb)
        self._bloch = BlochPhase(self.table, dim=self.spin.dim)

        # Create basis set
        nodes = np.array(range(self.graph.size()))
        self.basis = AtomicBasisSet(nodes, self.graph.species, table) 
        del nodes

        # Electronic structure
        self.egraph = self.graph.empty_graph()

        # Atomic interactions
        self.inter = dict()
        for key in kwargs:
            self.inter[key] = deepcopy(kwargs[key])

        # Symmetries
        if spacegroup != None:
            self.spacegroup = spacegroup

    @property
    def orthogonal(self):
        orthogonal = True
        for key in self.inter:
            if self.inter[key].overlap:
                orthogonal = False
                break
        return orthogonal

    def compute_integrals(self, *args):

        if len(args) == 0:
            args = [key for key in self.inter]

        local_inter = []
        non_local_inter = []
        non_local_phase = []
        for arg in args:
            if  hasattr(self.inter[arg],'local_block'):
                local_inter.append(arg)
            if  hasattr(self.inter[arg],'non_local_block'):
                if self.inter[arg].peierls:
                    non_local_phase.append(arg)
                else:
                    non_local_inter.append(arg)
        
        for inode in self.graph.nodes():
            for arg in local_inter:
                self.egraph.nodes[inode][arg] = \
                    self.inter[arg].local_block(inode, self.graph) 

        for edge in  self.graph.edges(keys=True): 
            inode, jnode, jedge = edge
            for arg in non_local_inter:
                self.egraph[inode][jnode][jedge][arg] =  \
                    self.inter[arg].non_local_block(edge, self.graph) 

        for edge in  self.graph.edges(keys=True):
            inode, jnode, jedge = edge
            for arg in non_local_phase:
                if arg in self.egraph[inode][jnode][jedge]:
                    next
                elif self.graph[inode][jnode][jedge]['order'] == 1:
                    self.egraph[inode][jnode][jedge][arg] =  \
                       self.inter[arg].non_local_phase(edge, self.graph)
                else:
                    phase = 1.+1j*0.
                    for ip, jp, kp in self.graph[inode][jnode][jedge]['path']:
                        if arg not in self.egraph[ip][jp][kp]:
                            self.egraph[ip][jp][kp][arg] = \
                                self.inter[arg].non_local_phase((ip,jp,kp), self.graph)
                        phase = phase*self.egraph[ip][jp][kp][arg]
                    self.egraph[inode][jnode][jedge][arg] = phase

    def electronic_supercell(self, scalings, spacegroup=None, centered=False, periodic=True, rtol=1.e-6):

        sc_graph = self.graph.supercell(scalings, centered=centered, periodic=periodic, rtol=rtol)

        estruct = ElectronicStructure(sc_graph, self.table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)
      
        unit_cell_size = len(self.graph)
        for inode in estruct.graph.nodes():
            ii = inode%unit_cell_size
            estruct.egraph.node[inode] = deepcopy(self.egraph.node[ii])

        for inode, jnode, kedge, data in estruct.graph.edges(keys=True,data=True): 
            ii = inode%unit_cell_size
            jj = jnode%unit_cell_size
            vec = data['vector']

            for kk in self.graph[ii][jj]:
                if np.linalg.norm(self.graph[ii][jj][kk]['vector']-vec) < rtol:
                    estruct.egraph[inode][jnode][kedge] = deepcopy(self.egraph[ii][jj][kk])

        return estruct

    def electronic_subcell(self, nodes, scalings, centered=False, spacegroup=None, rtol=1.e-06):

        sub_graph = self.graph.subcell(nodes, scalings, centered=centered, rtol=rtol)
        estruct = ElectronicStructure(sub_graph, self.table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)
      
        mapping = {}
        for idx, inode in enumerate(nodes):
            mapping[inode] = idx

        for inode in nodes:
            ii = mapping[inode]
            estruct.egraph.node[ii] = deepcopy(self.egraph.node[inode])

        for inode in nodes:
            ii = mapping[inode]
            for jnode in self.graph[inode]:
                for kedge in self.graph[inode][jnode]:
                    vec = self.graph[inode][jnode][kedge]['vector']
                    for jj in estruct.graph[ii]:
                        for kk in estruct.graph[ii][jj]:
                            sub_vec = estruct.graph[ii][jj][kk]['vector']
                            if np.linalg.norm(vec-sub_vec) < rtol:
                                estruct.egraph[ii][jj][kk] = deepcopy(self.egraph[inode][jnode][kedge])            

        return estruct


    def connected_basis_iter(self, inode, iorb, arg, order='f', local=True):

        if 'nodemap' in self.egraph.graph:

            if order == 'f' or order == 'bf':
                for jnode in self.graph[inode]:
                    jmask = self.egraph.graph['nodemap'][jnode]
                    jspec = self.graph.graph['species'][jnode]
                    for kedge in self.graph[inode][jnode]:
                        vec = self.graph[inode][jnode][kedge]['vector']
                        for jorb in np.arange(self.table[jspec].num_orbitals)[jmask]:
                            ixgrid = np.ix_([iorb], [jorb])
                            yield jnode, jorb, vec, \
                                  deepcopy(self.egraph[inode][jnode][kedge][arg][ixgrid][:,:])
            
                if local:
                    ispec = self.graph.graph['species'][inode]
                    imask = self.egraph.graph['nodemap'][inode]
                    for jorb in np.arange(self.table[ispec].num_orbitals)[imask]:
                        if jorb == iorb:
                            continue
                        else:
                            ixgrid = np.ix_([iorb], [jorb])
                            yield inode, jorb, np.zeros(3), \
                                  deepcopy(self.egraph.node[inode][arg][ixgrid][:,:])
            
            if order == 'b' or order == 'bf':
                for jnode in self.graph[inode]:
                    jmask = self.egraph.graph['nodemap'][jnode]
                    jspec = self.graph.graph['species'][jnode]
                    for kedge in self.graph[jnode][inode]:
                        vec = self.graph[jnode][inode][kedge]['vector']
                        for jorb in np.arange(self.table[jspec].num_orbitals)[jmask]:
                           ixgrid = np.ix_([jorb], [iorb])
                           yield jnode, jorb, vec, \
                                 deepcopy(self.egraph[jnode][inode][kedge][arg][ixgrid][:,:])
            
                if local:
                    ispec = self.graph.graph['species'][inode]
                    imask = self.egraph.graph['nodemap'][inode]
                    for jorb in np.arange(self.table[ispec].num_orbitals)[imask]:
                        if jorb == iorb:
                            continue
                        else:
                            ixgrid = np.ix_([jorb], [iorb])
                            yield inode, jorb, np.zeros(3), \
                                  deepcopy(self.egraph.node[inode][arg][ixgrid][:,:])

        else:

            if order == 'f' or order == 'bf':
                for jnode in self.graph[inode]:
                    jspec = self.graph.graph['species'][jnode]
                    for kedge in self.graph[inode][jnode]:
                        vec = self.graph[inode][jnode][kedge]['vector']
                        for jorb in range(self.table[jspec].num_orbitals):
                            ixgrid = np.ix_([iorb], [jorb])
                            yield jnode, jorb, vec, \
                                  deepcopy(self.egraph[inode][jnode][kedge][arg][ixgrid][:,:])
            
                if local:
                    ispec = self.graph.graph['species'][inode]
                    for jorb in range(self.table[ispec].num_orbitals):
                        if jorb == iorb:
                            continue
                        else:
                            ixgrid = np.ix_([iorb], [jorb])
                            yield inode, jorb, np.zeros(3), \
                                  deepcopy(self.egraph.node[inode][arg][ixgrid][:,:])
            
            if order == 'b' or order == 'bf':
                for jnode in self.graph[inode]:
                    jspec = self.graph.graph['species'][jnode]
                    for kedge in self.graph[jnode][inode]:
                         vec = self.graph[jnode][inode][kedge]['vector']
                         for jorb in range(self.table[jspec].num_orbitals):
                            ixgrid = np.ix_([jorb], [iorb])
                            yield jnode, jorb, vec, \
                                  deepcopy(self.egraph[jnode][inode][kedge][arg][ixgrid][:,:])
            
                if local:
                    ispec = self.graph.graph['species'][inode]
                    for jorb in range(self.table[ispec].num_orbitals):
                        if jorb == iorb:
                            continue
                        else:
                            ixgrid = np.ix_([jorb], [iorb])
                            yield inode, jorb, np.zeros(3), \
                                  deepcopy(self.egraph.node[inode][arg][ixgrid][:,:])


    def complex_shift(self, inter, eshift=0., iener=0.001):

        for inode in self.graph.nodes(data=False):
            for arg in inter:
                if arg in self.egraph.node[inode]:
                    for iorb in range(self.egraph.node[inode][arg].shape[0]):
                        self.egraph.node[inode][arg][iorb,iorb] +=  (eshift + 1j*iener)*self.spin.identity

    def decimate_orbital(self, node, orb, decimate=[], etol=1.e-8):

        if len(decimate) == 0:
            decimate = [key for key in self.inter]

        spec = self.graph.species[node]
        orbitals = {}
        orbitals[spec] = [orb]

        deltas = []
        itergraph = self.graph.empty_graph()
        for arg in decimate:

            ixgrid = np.ix_([orb], [orb])
            nodemat = self.spin.spinor_inv(self.egraph.node[node][arg][ixgrid][:,:])

            # First loop over connected orbitals:
            for inode, iorb, ivec, iargmat in self.connected_basis_iter(node, \
                                              orb, arg, order='b', local=True):
                imat = self.spin.spinor_product(iargmat,nodemat)

                # Second loop over connected orbitals:
                for jnode, jorb, jvec, jargmat in self.connected_basis_iter(node, \
                                                  orb, arg, order='f', local=True):
                    ijmat = self.spin.spinor_product(imat, jargmat)
                    ijvec = ivec + jvec
                    
                    # Decimation items
                    if np.amax(np.abs(ijmat)) > etol:
                        #print '............radius :', np.linalg.norm(ijvec), np.amax(np.abs(ijmat))
                        deltas.append((inode,jnode,deepcopy(ijvec),iorb, jorb, {arg:deepcopy(-ijmat)}))

        for ii,jj,ijvec,iorb,jorb,args in deltas:
            ispec = self.graph.graph['species'][ii]
            jspec = self.graph.graph['species'][jj]
            inum = self.table[ispec].num_orbitals
            jnum = self.table[jspec].num_orbitals

            if np.linalg.norm(ijvec) < 1.e-6:
                for arg in args:
                    self.egraph.node[ii][arg][iorb,jorb] += args[arg].reshape((self.spin.dim[0],self.spin.dim[1])) 

            else: 
                equivalent = self.graph.is_an_edge(ii, jj, ijvec)
                for arg in args:
                    mat = np.zeros((inum, jnum, self.spin.dim[0],self.spin.dim[1]),dtype=complex)
                    mat[iorb,jorb] += args[arg].reshape((self.spin.dim[0],self.spin.dim[1]))
                
                    if len(equivalent) > 0:
                        i, j, k = equivalent[0]
                        if arg in self.egraph[i][j][k]:
                            self.egraph[i][j][k][arg] += deepcopy(mat)
                        else:
                            self.egraph[i][j][k][arg] = deepcopy(mat)
                    else:      
                        path = self.graph.underlying_path(ii,jj,ijvec,itermax=12)
                        self.graph.add_edge(ii,jj,vector=deepcopy(ijvec),order=len(path),path=deepcopy(path))
                        kwargs = {arg:deepcopy(mat)}
                        self.egraph.add_edge(ii, jj, **kwargs)

        del deltas

    def decimate_orbitals_all3(self, orbitals, scalings, label, epts=[0.], ewpts=[1.], iener=1.e-6, \
                               include=[], decimate=[], spacegroup=None, path=True, rtol=1.e-6, etol=1.e-8):

        # Temporary electronic supercell
        esc = self.electronic_supercell(scalings, spacegroup=spacegroup, centered=True, periodic=False, rtol=rtol)
        scmap = self.graph.supercell_mapping(scalings, centered=True)
        #print esc.graph.coords

        # Decimation mask
        d_mask = []
        for spec in esc.graph.species:
            node_mask = np.ones(self.table[spec].num_orbitals,dtype=bool)
            if spec in orbitals:
                node_mask[orbitals[spec]] = False
            d_mask.append(node_mask)
        d_mask = np.array(d_mask).flatten()

        if self.orthogonal:

            hsc = esc.compute_H(include=decimate, bloch=False, spinscalar=False) 
            selfe = np.zeros((esc.basis.size, esc.basis.size, \
                        self.spin.dim[0], self.spin.dim[1]),dtype=complex)

            for ener, weigth in zip(epts,ewpts):          
                ei = np.zeros((esc.basis.size, esc.basis.size, \
                            self.spin.dim[0], self.spin.dim[1]),dtype=complex)
                for iorb in range(esc.basis.size):
                    ei[iorb,iorb][:,:] = (ener + 1j*iener)*self.spin.identity 
                
                ksc = ei-hsc
                
                oogrid = np.ix_(~d_mask, ~d_mask)
                htmp = ksc[oogrid][:,:]
                hoo = self.spin.spinor_inv(ksc[oogrid][:,:]) 
                #print 'hoo :', hoo.shape
                del oogrid
                
                iogrid = np.ix_(d_mask, ~d_mask) 
                hioo = self.spin.spinor_product(ksc[iogrid][:,:],hoo)
                #print 'hioo :', hioo.shape
                del hoo
                del iogrid
                
                oigrid = np.ix_(~d_mask, d_mask) 
                hiooi = self.spin.spinor_product(hioo,ksc[oigrid][:,:])  
                #print 'hiooi :', hiooi.shape
                del hioo
                del oigrid
                
                for iix, ix in enumerate(np.arange(len(d_mask))[d_mask]):
                    for jjx, jx in enumerate(np.arange(len(d_mask))[d_mask]):
                        ksc[ix, jx, :, :] -= hiooi[iix, jjx, :, :]  
  
                # heff = -(ksc-ei)
                # selfe = hsc - heff
                # heff = hsc - selfe
                selfe += (hsc + (ksc-ei))*weigth

            hsc -= selfe 
            for inode in self.egraph.nodes():
                icart = esc.graph.coords[inode]
                istart = esc.basis.first_basis_on_site(inode)
                istop = esc.basis.last_basis_on_site(inode)+1
                for arg in decimate:
                    del self.egraph.node[inode][arg]
                    self.egraph.node[inode][label] = \
                        deepcopy(hsc[istart:istop,istart:istop,:,:])
            
                for jsc in esc.egraph.nodes():
                    vec = esc.graph.coords[jsc] - icart
                    if np.linalg.norm(vec) < 1.e-9:
                        continue

                    jnode = jsc%len(self.graph.coords)
                    jstart = esc.basis.first_basis_on_site(jsc)
                    jstop = esc.basis.last_basis_on_site(jsc)+1
                    equivalent = self.graph.is_an_edge(inode, jnode, vec)
                    if len(equivalent) > 0:
                        i, j, k = equivalent[0]
                        for arg in decimate:
                            del self.egraph[i][j][k][arg]  
                            self.egraph[i][j][k][label] = \
                                deepcopy(hsc[istart:istop,jstart:jstop,:,:])
                    else:      
                        path = self.graph.underlying_path(inode,jnode,vec,itermax=22)
                        self.graph.add_edge(inode,jnode,vector=deepcopy(vec),order=len(path),path=deepcopy(path))
                        kwargs = {label:deepcopy(hsc[istart:istop,jstart:jstop,:,:])}
                        self.egraph.add_edge(inode, jnode, **kwargs)

        # Reduction of basis size
        mapping = {}
        new_table = deepcopy(self.table)
        for label in self.table:
            mask = np.ones(self.table[label].num_orbitals,dtype=bool)
            if label in orbitals:
                for iorb in np.sort(orbitals[label])[::-1]:
                    mask[iorb] = False
                    new_table[label].remove_orbital(iorb)
            mapping[label]=np.arange(self.table[label].num_orbitals)[mask]

        # Decimated electronic structure
        estruct = ElectronicStructure(self.graph, new_table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)

        if len(include) == 0:
            include = [key for key in self.inter]

        for inode, data in self.egraph.nodes(data=True):
            spec = self.graph.graph['species'][inode]
            matmap = np.ix_(mapping[spec],mapping[spec])
            for arg in include:
                if arg in self.egraph.node[inode]:
                    estruct.egraph.node[inode][arg] = \
                       deepcopy(self.egraph.node[inode][arg][matmap][:,:])

        for inode, jnode, jedge, data in self.egraph.edges(keys=True,data=True): 
            ispec = self.graph.graph['species'][inode]
            jspec = self.graph.graph['species'][jnode]
            matmap = np.ix_(mapping[ispec],mapping[jspec])
            for arg in include:
                if arg in self.egraph[inode][jnode][jedge]:
                    estruct.egraph[inode][jnode][jedge][arg] = \
                        deepcopy(self.egraph[inode][jnode][jedge][arg][matmap][:,:])

        return estruct


    def decimate_orbitals_all2_loc(self, orbitals, scalings, label, ener=0., iener=1.e-6, \
                               include=[], decimate=[], spacegroup=None, path=True, rtol=1.e-6, etol=1.e-8):

        # Temporary electronic supercell
        esc = self.electronic_supercell(scalings, spacegroup=spacegroup, centered=True, periodic=False, rtol=rtol)
        scmap = self.graph.supercell_mapping(scalings, centered=True)
        #print esc.graph.coords

        # Decimation mask
        d_mask = []
        for spec in esc.graph.species:
            node_mask = np.ones(self.table[spec].num_orbitals,dtype=bool)
            if spec in orbitals:
                node_mask[orbitals[spec]] = False
            d_mask.append(node_mask)
        d_mask = np.array(d_mask).flatten()

        if self.orthogonal:

            ei = np.zeros((esc.basis.size, esc.basis.size, \
                        self.spin.dim[0], self.spin.dim[1]),dtype=complex)
            for iorb in range(esc.basis.size):
                ei[iorb,iorb][:,:] = (ener + 1j*iener)*self.spin.identity 

            ksc = ei-esc.compute_H(include=decimate, bloch=False, spinscalar=False)

            hei = np.zeros((esc.basis.size, esc.basis.size, \
                        self.spin.dim[0], self.spin.dim[1]),dtype=complex)

            for inode in esc.egraph.nodes(data=False):
                istart = esc.basis.first_basis_on_site(inode)
                istop = esc.basis.last_basis_on_site(inode)+1
                hei[istart:istop,istart:istop,:,:] += ksc[istart:istop,istart:istop,:,:]

            oogrid = np.ix_(~d_mask, ~d_mask)
            hoo = self.spin.spinor_inv(hei[oogrid][:,:]) 
            #print 'hoo :', hoo.shape
            del oogrid

            iogrid = np.ix_(d_mask, ~d_mask) 
            hioo = self.spin.spinor_product(ksc[iogrid][:,:],hoo)
            #print 'hioo :', hioo.shape
            del hoo
            del iogrid

            oigrid = np.ix_(~d_mask, d_mask) 
            hiooi = self.spin.spinor_product(hioo,ksc[oigrid][:,:])  
            #print 'hiooi :', hiooi.shape
            del hioo
            del oigrid
            
            for iix, ix in enumerate(np.arange(len(d_mask))[d_mask]):
                for jjx, jx in enumerate(np.arange(len(d_mask))[d_mask]):
                    ksc[ix, jx, :, :] -= hiooi[iix, jjx, :, :]  

            hsc = -(ksc-ei)
 
            for inode in self.egraph.nodes():
                icart = esc.graph.coords[inode]
                istart = esc.basis.first_basis_on_site(inode)
                istop = esc.basis.last_basis_on_site(inode)+1
                for arg in decimate:
                    del self.egraph.node[inode][arg]
                    self.egraph.node[inode][label] = \
                        deepcopy(hsc[istart:istop,istart:istop,:,:])
            
                for jsc in esc.egraph.nodes():
                    vec = esc.graph.coords[jsc] - icart
                    if np.linalg.norm(vec) < 1.e-9:
                        continue

                    jnode = jsc%len(self.graph.coords)
                    jstart = esc.basis.first_basis_on_site(jsc)
                    jstop = esc.basis.last_basis_on_site(jsc)+1
                    equivalent = self.graph.is_an_edge(inode, jnode, vec)
                    if len(equivalent) > 0:
                        i, j, k = equivalent[0]
                        for arg in decimate:
                            del self.egraph[i][j][k][arg]  
                            self.egraph[i][j][k][label] = \
                                deepcopy(hsc[istart:istop,jstart:jstop,:,:])
                    else:      
                        path = self.graph.underlying_path(inode,jnode,vec,itermax=22)
                        self.graph.add_edge(inode,jnode,vector=deepcopy(vec),order=len(path),path=deepcopy(path))
                        kwargs = {label:deepcopy(hsc[istart:istop,jstart:jstop,:,:])}
                        self.egraph.add_edge(inode, jnode, **kwargs)

        # Reduction of basis size
        mapping = {}
        new_table = deepcopy(self.table)
        for label in self.table:
            mask = np.ones(self.table[label].num_orbitals,dtype=bool)
            if label in orbitals:
                for iorb in np.sort(orbitals[label])[::-1]:
                    mask[iorb] = False
                    new_table[label].remove_orbital(iorb)
            mapping[label]=np.arange(self.table[label].num_orbitals)[mask]

        # Decimated electronic structure
        estruct = ElectronicStructure(self.graph, new_table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)

        if len(include) == 0:
            include = [key for key in self.inter]

        for inode, data in self.egraph.nodes(data=True):
            spec = self.graph.graph['species'][inode]
            matmap = np.ix_(mapping[spec],mapping[spec])
            for arg in include:
                if arg in self.egraph.node[inode]:
                    estruct.egraph.node[inode][arg] = \
                       deepcopy(self.egraph.node[inode][arg][matmap][:,:])

        for inode, jnode, jedge, data in self.egraph.edges(keys=True,data=True): 
            ispec = self.graph.graph['species'][inode]
            jspec = self.graph.graph['species'][jnode]
            matmap = np.ix_(mapping[ispec],mapping[jspec])
            for arg in include:
                if arg in self.egraph[inode][jnode][jedge]:
                    estruct.egraph[inode][jnode][jedge][arg] = \
                        deepcopy(self.egraph[inode][jnode][jedge][arg][matmap][:,:])

        return estruct


    def decimate_orbitals_all2k(self, orbitals, scalings, label, kpt=[0.,0.,0.], fractional=True, ener=0., iener=1.e-6, \
                               include=[], decimate=[], spacegroup=None, path=True, rtol=1.e-6, etol=1.e-8):

        if fractional: 
           kpt = np.dot(kpt,self.lattice.reciprocal_lattice.matrix)
        else:
           kpt = kpt

        # Temporary electronic supercell
        esc = self.electronic_supercell(scalings, spacegroup=spacegroup, centered=True, periodic=True, rtol=rtol)
        scmap = self.graph.supercell_mapping(scalings, centered=True)
        #print esc.graph.coords
        esc.compute_bloch_phase(kpt=kpt)

        # Decimation mask
        d_mask = []
        for spec in esc.graph.species:
            node_mask = np.ones(self.table[spec].num_orbitals,dtype=bool)
            if spec in orbitals:
                node_mask[orbitals[spec]] = False
            d_mask.append(node_mask)
        d_mask = np.array(d_mask).flatten()

        if self.orthogonal:

            ei = np.zeros((esc.basis.size, esc.basis.size, \
                        self.spin.dim[0], self.spin.dim[1]),dtype=complex)
            for iorb in range(esc.basis.size):
                ei[iorb,iorb][:,:] = (ener + 1j*iener)*self.spin.identity 


            ksc = ei-esc.compute_H(include=decimate, bloch=True, spinscalar=False) 

            oogrid = np.ix_(~d_mask, ~d_mask)
            hoo = self.spin.spinor_inv(ksc[oogrid][:,:]) 
            #print 'hoo :', hoo.shape
            del oogrid

            iogrid = np.ix_(d_mask, ~d_mask) 
            hioo = self.spin.spinor_product(ksc[iogrid][:,:],hoo)
            #print 'hioo :', hioo.shape
            del hoo
            del iogrid

            oigrid = np.ix_(~d_mask, d_mask) 
            hiooi = self.spin.spinor_product(hioo,ksc[oigrid][:,:])  
            #print 'hiooi :', hiooi.shape
            del hioo
            del oigrid
            
            for iix, ix in enumerate(np.arange(len(d_mask))[d_mask]):
                for jjx, jx in enumerate(np.arange(len(d_mask))[d_mask]):
                    ksc[ix, jx, :, :] -= hiooi[iix, jjx, :, :]  

            hsc = -(ksc-ei)
 
            for inode in self.egraph.nodes():
                icart = esc.graph.coords[inode]
                istart = esc.basis.first_basis_on_site(inode)
                istop = esc.basis.last_basis_on_site(inode)+1
                for arg in decimate:
                    del self.egraph.node[inode][arg]
                    self.egraph.node[inode][label] = \
                        deepcopy(hsc[istart:istop,istart:istop,:,:])
            
                for jsc in esc.egraph.nodes():
                    vec = esc.graph.coords[jsc] - icart
                    if np.linalg.norm(vec) < 1.e-9:
                        continue

                    jnode = jsc%len(self.graph.coords)
                    jstart = esc.basis.first_basis_on_site(jsc)
                    jstop = esc.basis.last_basis_on_site(jsc)+1
                    equivalent = self.graph.is_an_edge(inode, jnode, vec)
                    if len(equivalent) > 0:
                        i, j, k = equivalent[0]
                        for arg in decimate:
                            del self.egraph[i][j][k][arg]  
                            self.egraph[i][j][k][label] = \
                                deepcopy(hsc[istart:istop,jstart:jstop,:,:])
                    else:      
                        path = self.graph.underlying_path(inode,jnode,vec,itermax=22)
                        self.graph.add_edge(inode,jnode,vector=deepcopy(vec),order=len(path),path=deepcopy(path))
                        kwargs = {label:deepcopy(hsc[istart:istop,jstart:jstop,:,:])}
                        self.egraph.add_edge(inode, jnode, **kwargs)

        # Reduction of basis size
        mapping = {}
        new_table = deepcopy(self.table)
        for label in self.table:
            mask = np.ones(self.table[label].num_orbitals,dtype=bool)
            if label in orbitals:
                for iorb in np.sort(orbitals[label])[::-1]:
                    mask[iorb] = False
                    new_table[label].remove_orbital(iorb)
            mapping[label]=np.arange(self.table[label].num_orbitals)[mask]

        # Decimated electronic structure
        estruct = ElectronicStructure(self.graph, new_table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)

        if len(include) == 0:
            include = [key for key in self.inter]

        for inode, data in self.egraph.nodes(data=True):
            spec = self.graph.graph['species'][inode]
            matmap = np.ix_(mapping[spec],mapping[spec])
            for arg in include:
                if arg in self.egraph.node[inode]:
                    estruct.egraph.node[inode][arg] = \
                       deepcopy(self.egraph.node[inode][arg][matmap][:,:])

        for inode, jnode, jedge, data in self.egraph.edges(keys=True,data=True): 
            ispec = self.graph.graph['species'][inode]
            jspec = self.graph.graph['species'][jnode]
            matmap = np.ix_(mapping[ispec],mapping[jspec])
            for arg in include:
                if arg in self.egraph[inode][jnode][jedge]:
                    estruct.egraph[inode][jnode][jedge][arg] = \
                        deepcopy(self.egraph[inode][jnode][jedge][arg][matmap][:,:])

        return estruct




    def decimate_orbitals_all2(self, orbitals, scalings, label, ener=0., iener=1.e-6, \
                               include=[], decimate=[], spacegroup=None, path=True, rtol=1.e-6, etol=1.e-8):

        # Temporary electronic supercell
        esc = self.electronic_supercell(scalings, spacegroup=spacegroup, centered=True, periodic=False, rtol=rtol)
        scmap = self.graph.supercell_mapping(scalings, centered=True)
        #print esc.graph.coords

        # Decimation mask
        d_mask = []
        for spec in esc.graph.species:
            node_mask = np.ones(self.table[spec].num_orbitals,dtype=bool)
            if spec in orbitals:
                node_mask[orbitals[spec]] = False
            d_mask.append(node_mask)
        d_mask = np.array(d_mask).flatten()

        if self.orthogonal:

            ei = np.zeros((esc.basis.size, esc.basis.size, \
                        self.spin.dim[0], self.spin.dim[1]),dtype=complex)
            for iorb in range(esc.basis.size):
                ei[iorb,iorb][:,:] = (ener + 1j*iener)*self.spin.identity 


            ksc = ei-esc.compute_H(include=decimate, bloch=False, spinscalar=False) 

            oogrid = np.ix_(~d_mask, ~d_mask)
            hoo = self.spin.spinor_inv(ksc[oogrid][:,:]) 
            #print 'hoo :', hoo.shape
            del oogrid

            iogrid = np.ix_(d_mask, ~d_mask) 
            hioo = self.spin.spinor_product(ksc[iogrid][:,:],hoo)
            #print 'hioo :', hioo.shape
            del hoo
            del iogrid

            oigrid = np.ix_(~d_mask, d_mask) 
            hiooi = self.spin.spinor_product(hioo,ksc[oigrid][:,:])  
            #print 'hiooi :', hiooi.shape
            del hioo
            del oigrid
            
            for iix, ix in enumerate(np.arange(len(d_mask))[d_mask]):
                for jjx, jx in enumerate(np.arange(len(d_mask))[d_mask]):
                    ksc[ix, jx, :, :] -= hiooi[iix, jjx, :, :]  

            hsc = -(ksc-ei)
 
            for inode in self.egraph.nodes():
                icart = esc.graph.coords[inode]
                istart = esc.basis.first_basis_on_site(inode)
                istop = esc.basis.last_basis_on_site(inode)+1
                for arg in decimate:
                    del self.egraph.node[inode][arg]
                    self.egraph.node[inode][label] = \
                        deepcopy(hsc[istart:istop,istart:istop,:,:])
            
                for jsc in esc.egraph.nodes():
                    vec = esc.graph.coords[jsc] - icart
                    if np.linalg.norm(vec) < 1.e-9:
                        continue

                    jnode = jsc%len(self.graph.coords)
                    jstart = esc.basis.first_basis_on_site(jsc)
                    jstop = esc.basis.last_basis_on_site(jsc)+1
                    equivalent = self.graph.is_an_edge(inode, jnode, vec)
                    if len(equivalent) > 0:
                        i, j, k = equivalent[0]
                        for arg in decimate:
                            del self.egraph[i][j][k][arg]  
                            self.egraph[i][j][k][label] = \
                                deepcopy(hsc[istart:istop,jstart:jstop,:,:])
                    else:      
                        path = self.graph.underlying_path(inode,jnode,vec,itermax=22)
                        self.graph.add_edge(inode,jnode,vector=deepcopy(vec),order=len(path),path=deepcopy(path))
                        kwargs = {label:deepcopy(hsc[istart:istop,jstart:jstop,:,:])}
                        self.egraph.add_edge(inode, jnode, **kwargs)

        # Reduction of basis size
        mapping = {}
        new_table = deepcopy(self.table)
        for label in self.table:
            mask = np.ones(self.table[label].num_orbitals,dtype=bool)
            if label in orbitals:
                for iorb in np.sort(orbitals[label])[::-1]:
                    mask[iorb] = False
                    new_table[label].remove_orbital(iorb)
            mapping[label]=np.arange(self.table[label].num_orbitals)[mask]

        # Decimated electronic structure
        estruct = ElectronicStructure(self.graph, new_table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)

        if len(include) == 0:
            include = [key for key in self.inter]

        for inode, data in self.egraph.nodes(data=True):
            spec = self.graph.graph['species'][inode]
            matmap = np.ix_(mapping[spec],mapping[spec])
            for arg in include:
                if arg in self.egraph.node[inode]:
                    estruct.egraph.node[inode][arg] = \
                       deepcopy(self.egraph.node[inode][arg][matmap][:,:])

        for inode, jnode, jedge, data in self.egraph.edges(keys=True,data=True): 
            ispec = self.graph.graph['species'][inode]
            jspec = self.graph.graph['species'][jnode]
            matmap = np.ix_(mapping[ispec],mapping[jspec])
            for arg in include:
                if arg in self.egraph[inode][jnode][jedge]:
                    estruct.egraph[inode][jnode][jedge][arg] = \
                        deepcopy(self.egraph[inode][jnode][jedge][arg][matmap][:,:])

        return estruct




    def decimate_orbitals_all(self, orbitals, scalings, label, include=[], decimate=[], spacegroup=None, path=True, rtol=1.e-6, etol=1.e-8):

        # Temporary electronic supercell
        esc = self.electronic_supercell(scalings, spacegroup=spacegroup, centered=True, rtol=rtol)
        scmap = self.graph.supercell_mapping(scalings, centered=True)
        #print 'Decimate orbitals :', orbitals

        #for inode in self.egraph.nodes_iter():
        #    print 'Supercell initial node ', inode
        #    for arg in decimate:
        #        print esc.egraph.node[inode][arg][:,:,0,0]
                


        # Decimation mask
        d_mask = []
        for spec in esc.graph.species:
            node_mask = np.ones(self.table[spec].num_orbitals,dtype=bool)
            if spec in orbitals:
                node_mask[orbitals[spec]] = False
            d_mask.append(node_mask)
        d_mask = np.array(d_mask).flatten()

        if self.orthogonal:
            hsc = esc.compute_H(include=decimate, bloch=False, spinscalar=False) 

            for inode in self.egraph.nodes():
                icart = esc.graph.coords[inode]
                istart = esc.basis.first_basis_on_site(inode)
                istop = esc.basis.last_basis_on_site(inode)+1
                #print 'Initial node ', inode, icart, istart, istop
                #for arg in decimate:
                #    print esc.egraph.node[inode][arg][:,:,0,0]
                
                #print 'Initial ham ', inode, icart, istart, istop
                #for arg in decimate:
                #    print hsc[istart:istop,istart:istop,0,0]


 
            oogrid = np.ix_(~d_mask, ~d_mask)
            htmp = hsc[oogrid][:,:]
            hoo = self.spin.spinor_inv(hsc[oogrid][:,:]) 
            #print 'hoo :', hoo.shape
            #for iix, ix in enumerate(np.arange(len(d_mask))[~d_mask]):
            #    for jjx, jx in enumerate(np.arange(len(d_mask))[~d_mask]):
            #        print ix, jx, iix, jjx, hsc[ix,jx][0,0], htmp[iix, jjx, 0, 0], hoo[iix, jjx, 0, 0] 

            del oogrid

            iogrid = np.ix_(d_mask, ~d_mask) 
            hioo = self.spin.spinor_product(hsc[iogrid][:,:],hoo)
            #print 'hioo :', hioo.shape
            del hoo
            del iogrid

            oigrid = np.ix_(~d_mask, d_mask) 
            hiooi = self.spin.spinor_product(hioo,hsc[oigrid][:,:])  
            #print 'hiooi :', hiooi.shape
            del hioo
            del oigrid
            
            #iigrid = np.ix_(d_mask, d_mask)  
            #print 'before : ', hsc[iigrid][:,:]
            for iix, ix in enumerate(np.arange(len(d_mask))[d_mask]):
                for jjx, jx in enumerate(np.arange(len(d_mask))[d_mask]):
                    #print '... before ', ix, jx, iix, jjx, hsc[ix, jx, :, :]
                    hsc[ix, jx, :, :] -= hiooi[iix, jjx, :, :]  
                    #print '... after ', ix, jx, iix, jjx, hsc[ix, jx, :, :]
            #hsc[iigrid][:,:] -= hiooi
            #print 'after : ', hsc[iigrid][:,:]

            for inode in self.egraph.nodes():
                icart = esc.graph.coords[inode]
                istart = esc.basis.first_basis_on_site(inode)
                istop = esc.basis.last_basis_on_site(inode)+1
                #print 'Recast node ', inode, icart, istart, istop
                for arg in decimate:
                    del self.egraph.node[inode][arg]
                    self.egraph.node[inode][label] = \
                        deepcopy(hsc[istart:istop,istart:istop,:,:])
                    #print hsc[istart:istop,istart:istop,0,0]

            
                for jsc in esc.egraph.nodes():
                    vec = esc.graph.coords[jsc] - icart
                    if np.linalg.norm(vec) < 1.e-9:
                        continue

                    jnode = jsc%len(self.graph.coords)
                    jstart = esc.basis.first_basis_on_site(jsc)
                    jstop = esc.basis.last_basis_on_site(jsc)+1
                    #print '...connection with ', jnode, jsc, esc.graph.coords[jsc], vec

                    equivalent = self.graph.is_an_edge(inode, jnode, vec)
                    if len(equivalent) > 0:
                        i, j, k = equivalent[0]
                        #print '......found equivalent ', i,j,k, self.graph[i][j][k]['vector']
                        for arg in decimate:
                            del self.egraph[i][j][k][arg]  
                            self.egraph[i][j][k][label] = \
                                deepcopy(hsc[istart:istop,jstart:jstop,:,:])
                            #print hsc[istart:istop,jstart:jstop,0,0]
                    else:      
                        path = self.graph.underlying_path(inode,jnode,vec,itermax=22)
                        self.graph.add_edge(inode,jnode,vector=deepcopy(vec),order=len(path),path=deepcopy(path))
                        #print ' ------------------------------- '
                        #print ' new edge :', inode, jnode, vec
                        #print hsc[istart:istop,jstart:jstop,0,0]
                        kwargs = {label:deepcopy(hsc[istart:istop,jstart:jstop,:,:])}
                        self.egraph.add_edge(inode, jnode, **kwargs)

        # Reduction of basis size
        mapping = {}
        new_table = deepcopy(self.table)
        for label in self.table:
            mask = np.ones(self.table[label].num_orbitals,dtype=bool)
            if label in orbitals:
                for iorb in np.sort(orbitals[label])[::-1]:
                    mask[iorb] = False
                    new_table[label].remove_orbital(iorb)
            mapping[label]=np.arange(self.table[label].num_orbitals)[mask]

        # Decimated electronic structure
        estruct = ElectronicStructure(self.graph, new_table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)

        if len(include) == 0:
            include = [key for key in self.inter]

        for inode, data in self.egraph.nodes(data=True):
            spec = self.graph.graph['species'][inode]
            matmap = np.ix_(mapping[spec],mapping[spec])
            for arg in include:
                if arg in self.egraph.node[inode]:
                    estruct.egraph.node[inode][arg] = \
                       deepcopy(self.egraph.node[inode][arg][matmap][:,:])

        for inode, jnode, jedge, data in self.egraph.edges(keys=True,data=True): 
            ispec = self.graph.graph['species'][inode]
            jspec = self.graph.graph['species'][jnode]
            matmap = np.ix_(mapping[ispec],mapping[jspec])
            for arg in include:
                if arg in self.egraph[inode][jnode][jedge]:
                    estruct.egraph[inode][jnode][jedge][arg] = \
                        deepcopy(self.egraph[inode][jnode][jedge][arg][matmap][:,:])

        return estruct




    def decimate_orbitals_iter(self, orbitals, scalings, include=[], decimate=[], spacegroup=None, path=True, rtol=1.e-6, etol=1.e-8):

        # Temporary electronic supercell
        esc = self.electronic_supercell(scalings, spacegroup=spacegroup, centered=True, rtol=rtol)
        scmap = self.graph.supercell_mapping(scalings, centered=True)
        esc.egraph.graph['nodemap'] = []
        for spec in esc.graph.species:
            nodemap = np.ones(self.table[spec].num_orbitals,dtype=bool)
            esc.egraph.graph['nodemap'].append(nodemap)

        # Decimation of the periodically repeated orbitals 
        for spec in orbitals:
            for orb in orbitals[spec]:
                #print '## Decimation of ', spec, orb
                for node in self.graph.nodes_of_species_iter(spec):
                    sc_coords = esc.graph.coords[scmap[node]]
                    dist = np.linalg.norm(sc_coords-self.graph.coords[node], axis=-1)
                    #print '...node ', node, dist
                    for idx in np.argsort(dist):
                        #print '......image', idx, dist[idx]
                        sc_node = scmap[node][idx]
                        esc.decimate_orbital(sc_node, orb, decimate=decimate, etol=1.e-8)
                        esc.egraph.graph['nodemap'][sc_node][orb] = False
        del scmap
        
        # Temporary electronic subcell
        esub = esc.electronic_subcell(np.arange(len(self.graph.coords)),scalings,centered=True, spacegroup=spacegroup, rtol=rtol)
        del esc

        # Reduction of basis size
        mapping = {}
        new_table = deepcopy(self.table)
        for label in self.table:
            mask = np.ones(self.table[label].num_orbitals,dtype=bool)
            if label in orbitals:
                mask[orbitals[label]] = False
                for iorb in orbitals[label]:
                    new_table[label].remove_orbital(iorb)
            mapping[label]=np.arange(self.table[label].num_orbitals)[mask]

        # Decimated electronic structure
        estruct = ElectronicStructure(esub.graph, new_table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)

        if len(include) == 0:
            include = [key for key in self.inter]

        for inode, data in esub.egraph.nodes(data=True):
            spec = esub.graph.graph['species'][inode]
            matmap = np.ix_(mapping[spec],mapping[spec])
            for arg in include:
                if arg in esub.egraph.node[inode]:
                    estruct.egraph.node[inode][arg] = \
                       deepcopy(esub.egraph.node[inode][arg][matmap][:,:])

        for inode, jnode, jedge, data in esub.egraph.edges(keys=True,data=True): 
            ispec = esub.graph.graph['species'][inode]
            jspec = esub.graph.graph['species'][jnode]
            matmap = np.ix_(mapping[ispec],mapping[jspec])
            for arg in include:
                if arg in esub.egraph[inode][jnode][jedge]:
                    estruct.egraph[inode][jnode][jedge][arg] = \
                        deepcopy(esub.egraph[inode][jnode][jedge][arg][matmap][:,:])

        return estruct


    def decimate_orbitals_local(self, orbitals, include=[],  decimate=[], spacegroup=None, eshift=0., iener=1.e-6, path=True, etol=1.e-8):

        if len(decimate) == 0:
            decimate = [key for key in self.inter]
        
        self.complex_shift(decimate, eshift=eshift, iener=iener)

        deltas = []
        itergraph = self.graph.empty_graph()    
        for inode in itergraph.nodes(data=False):
            spec = self.graph.graph['species'][inode]

            if spec in orbitals:
                for iorb in orbitals[spec]:
                    for arg in decimate:
                        if arg not in self.egraph.node[inode]:
                            continue

                        ixgrid = np.ix_([iorb], [iorb])
                        nodemat = self.spin.spinor_inv(self.egraph.node[inode][arg][ixgrid][:,:])

                        # First loop over connected orbitals on same node
                        for lorb in range(self.table[spec].num_orbitals): 
                            if lorb in orbitals[spec]:
                                continue
                            lxgrid = np.ix_([lorb], [iorb])
                            tmp_mat0 = deepcopy(self.egraph.node[inode][arg][lxgrid][:,:])
                            imat = self.spin.spinor_product(tmp_mat0,nodemat)

                            # Second loop over connected orbitals on same node
                            for morb in range(self.table[spec].num_orbitals): 
                                if morb in orbitals[spec]:
                                    continue
                                mxgrid = np.ix_([iorb], [morb])
                                tmp_mat1 = deepcopy(self.egraph.node[inode][arg][mxgrid][:,:])
                                ijmat = self.spin.spinor_product(imat,tmp_mat1)
 
                                # Decimation
                                ijxgrid = np.ix_([lorb], [morb])
                                if np.amax(np.abs(ijmat)) > etol: 
                                    deltas.append((inode,inode,np.zeros(3),lorb, morb, {arg:deepcopy(-ijmat)}))
                                
                        # First loop over orbitals on connected nodes
                        for ii in itergraph[inode]:
                            ispec = self.graph.graph['species'][ii]
                            for lorb in range(self.table[ispec].num_orbitals):
                                if ispec in orbitals:
                                    if lorb in orbitals[ispec]:
                                        continue
                                for mm in itergraph[ii][inode]:
                                    if arg not in self.egraph[ii][inode][mm]:
                                        continue
                                    ivec = self.graph[ii][inode][mm]['vector']
                                    lxgrid = np.ix_([lorb], [iorb])
                                    tmp_mat0 = deepcopy(self.egraph[ii][inode][mm][arg][lxgrid][:,:])
                                    imat = self.spin.spinor_product(tmp_mat0,nodemat) 

                                    # Second loop over orbitals on connected nodes
                                    for jj in itergraph[inode]:
                                        jspec = self.graph.graph['species'][jj]
                                        for morb in range(self.table[jspec].num_orbitals): 
                                            if jspec in orbitals:
                                                if morb in orbitals[jspec]:
                                                    continue
                                            for nn in itergraph[inode][jj]:
                                                if arg not in self.egraph[inode][jj][nn]:
                                                    continue
                                                jvec = self.graph[inode][jj][nn]['vector']
                                                ijvec = ivec + jvec
                                                mxgrid = np.ix_([iorb], [morb])
                                                tmp_mat1 = deepcopy(self.egraph[inode][jj][nn][arg][mxgrid][:,:])
                                                ijmat = self.spin.spinor_product(imat,tmp_mat1)

                                                # Decimation
                                                if np.amax(np.abs(ijmat)) > etol: 
                                                    deltas.append((ii,jj,deepcopy(ijvec),lorb, morb, {arg:deepcopy(-ijmat)}))


        #print '####################################################################'
        for ii,jj,ijvec,iorb,jorb,args in deltas:
            ispec = self.graph.graph['species'][ii]
            jspec = self.graph.graph['species'][jj]
            inum = self.table[ispec].num_orbitals
            jnum = self.table[jspec].num_orbitals

            if np.linalg.norm(ijvec) < 1.e-6:
                for arg in args:
                    #print " Local {:1d} {:1d} - {:1d} {:1d}".format(ii,iorb,jj,jorb),':', "   {: 8.6f} {: 8.6f}".format(args[arg][0,0,0,0].real, args[arg][0,0,0,0].imag)
                    old = deepcopy(self.egraph.node[ii][arg][iorb,jorb,0,0])
                    self.egraph.node[ii][arg][iorb,jorb] += args[arg].reshape((self.spin.dim[0],self.spin.dim[1])) 
                    new = deepcopy(self.egraph.node[ii][arg][iorb,jorb,0,0])
                    #print 'Old ',  "   {: 8.6f} {: 8.6f}".format(old.real, old.imag), " - New {: 8.6f} {: 8.6f}".format(new.real, new.imag)
           
         
        for ii,jj,ijvec,iorb,jorb,args in deltas:
            ispec = self.graph.graph['species'][ii]
            jspec = self.graph.graph['species'][jj]
            inum = self.table[ispec].num_orbitals
            jnum = self.table[jspec].num_orbitals

            if np.linalg.norm(ijvec) >= 1.e-6:
                equivalent = self.graph.is_an_edge(ii, jj, ijvec)
                for arg in args:
                    mat = np.zeros((inum, jnum, self.spin.dim[0],self.spin.dim[1]),dtype=complex)
                    mat[iorb,jorb] += args[arg].reshape((self.spin.dim[0],self.spin.dim[1]))
                
                    if len(equivalent) > 0:
                        i, j, k = equivalent[0]
                        #print " Local {:1d} {:1d} - {:1d} {:1d}".format(ii,iorb,jj,jorb),':', " {:1d} {:1d} {:1d}".format(i,j,k), \
                        #" {: 8.6f} {: 8.6f} {: 8.6f}".format(*ijvec),  "   {: 8.6f} {: 8.6f}".format(args[arg][0,0,0,0].real, args[arg][0,0,0,0].imag)
                        if arg in self.egraph[i][j][k]:
                            self.egraph[i][j][k][arg] += deepcopy(mat)
                        else:
                            self.egraph[i][j][k][arg] = deepcopy(mat)
                    else:      
                        path = self.graph.underlying_path(ii,jj,ijvec,itermax=12)
                        #print " Local {:1d} {:1d} - {:1d} {:1d}".format(ii,iorb,jj,jorb),':', " new", \
                        #" {: 8.6f} {: 8.6f} {: 8.6f}".format(*ijvec),  "   {: 8.6f} {: 8.6f}".format(args[arg][0,0,0,0].real, args[arg][0,0,0,0].imag)
                        self.graph.add_edge(ii,jj,vector=deepcopy(ijvec),order=len(path),path=deepcopy(path))
                        kwargs = {arg:deepcopy(mat)}
                        self.egraph.add_edge(ii, jj, **kwargs)
        del deltas

        new_atoms = []
        for label in self.table:
            if label not in orbitals:
                new_atoms.append(deepcopy(self.table[label]))
            else:
                atom_label = deepcopy(label)
                atom_symbol = deepcopy(self.table[label].symbol)
                atom_orbs = []
                for iorb, orb in enumerate(self.table[label].orbitals):
                    if iorb not in orbitals[label]:
                        atom_orbs.append(orb[0])
                atom = NumericalAtom(atom_label,atom_symbol,orbs=atom_orbs)
                new_atoms.append(deepcopy(atom))
        new_table = AtomicTable(new_atoms)

        mapping = {}
        for label in self.table:
            mask = np.ones(self.table[label].num_orbitals,dtype=bool)
            if label in orbitals:
                mask[orbitals[label]] = False
            mapping[label]=np.arange(self.table[label].num_orbitals)[mask]

        estruct = ElectronicStructure(self.graph, new_table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)

        if len(include) == 0:
            include = [key for key in self.inter]

        for inode, data in self.egraph.nodes(data=True):
            spec = self.graph.graph['species'][inode]
            matmap = np.ix_(mapping[spec],mapping[spec])
            for arg in include:
                if arg in self.egraph.node[inode]:
                    estruct.egraph.node[inode][arg] = \
                       deepcopy(self.egraph.node[inode][arg][matmap][:,:])


        for inode, jnode, jedge, data in  self.egraph.edges(keys=True,data=True): 
            ispec = self.graph.graph['species'][inode]
            jspec = self.graph.graph['species'][jnode]
            matmap = np.ix_(mapping[ispec],mapping[jspec])
            for arg in include:
                if arg in self.egraph[inode][jnode][jedge]:
                    estruct.egraph[inode][jnode][jedge][arg] = \
                        deepcopy(self.egraph[inode][jnode][jedge][arg][matmap][:,:])

        return estruct


    def decimate_species_local(self, species, include=[], decimate=[], spacegroup=None, eshift=0., iener=0.001, path=True, etol=1.e-6):

        if len(decimate) == 0:
            decimate = [key for key in self.inter]

        for inode in self.graph.nodes(data=False):
            for arg in decimate:
                if arg in self.egraph.node[inode]:
                    self.egraph.node[inode][arg] +=  eshift + 1j*iener

        deltas = []
        itergraph = self.graph.empty_graph()

        for inode in itergraph.nodes(data=False):
            spec = self.graph.graph['species'][inode]

            if spec in species:
                for arg in decimate:
                    if arg not in self.egraph.node[inode]:
                        next
                    nodemat = self.spin.spinor_inv(self.egraph.node[inode][arg])

                    # First loop over connected nodes
                    for ii in itergraph[inode]:
                        ispec = self.graph.graph['species'][ii]
                        if ispec in species:
                            next
                        for mm in itergraph[ii][inode]:
                            if arg not in self.egraph[ii][inode][mm]:
                                next
                            ivec = self.graph[ii][inode][mm]['vector']
                            imat = self.spin.spinor_product(self.egraph[ii][inode][mm][arg],nodemat)

                            # Second loop over connected nodes
                            for jj in itergraph[inode]:
                                jspec = self.graph.graph['species'][jj]
                                if jspec in species:
                                    next
                                for nn in itergraph[inode][jj]:
                                    if arg not in self.egraph[inode][jj][nn]:
                                        next
                                    jvec = self.graph[inode][jj][nn]['vector']
                                    ijvec = ivec + jvec
                                    ijmat = self.spin.spinor_product(imat,self.egraph[inode][jj][nn][arg])
                                    if np.amax(np.abs(ijmat)) > etol: 
                                        deltas.append((ii,jj,deepcopy(ijvec),{arg:deepcopy(-ijmat)})) 
 
        for ii,jj,ijvec,args in deltas:

            if np.linalg.norm(ijvec) < 1.e-6:
                for arg in args:
                    self.egraph.node[ii][arg] += deepcopy(args[arg])
            else:
                equivalent = self.graph.is_an_edge(ii, jj, ijvec)
                if len(equivalent) > 0:
                    i, j, k = equivalent[0]
                    for arg in args:
                        if arg in self.egraph[i][j][k]:
                            self.egraph[i][j][k][arg] += deepcopy(args[arg])
                        else:
                            self.egraph[i][j][k][arg] = deepcopy(args[arg])
                else:      
                    path = self.graph.underlying_path(ii,jj,ijvec,itermax=12,except_species=species)
                    self.graph.add_edge(ii,jj,vector=deepcopy(ijvec),order=len(path),path=deepcopy(path))
                    self.egraph.add_edge(ii, jj, **args)
        del deltas


        graph, mapping = self.graph.remove_nodes_by_species(species)
        estruct = ElectronicStructure(graph, self.table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)

        if len(include) == 0:
            include = [key for key in self.inter]

        for inode, data in self.egraph.nodes(data=True):

            if inode in mapping:
                jnode = mapping[inode]
                for arg in include:
                    if arg in self.egraph.node[inode]:
                        estruct.egraph.node[jnode][arg] = \
                            deepcopy(self.egraph.node[inode][arg])

        for inode, jnode, jedge, data in  self.egraph.edges(keys=True,data=True): 

            if inode in mapping:
                if jnode in mapping:
                    knode = mapping[inode] 
                    lnode = mapping[jnode] 
                    
                    new_path = []
                    for ip,jp,kp in self.graph[inode][jnode][jedge]['path']:
                        mpath = (mapping[ip],mapping[jp],kp)
                        new_path.append(mpath)
                    estruct.graph[knode][lnode][jedge]['path'] = deepcopy(new_path)

                    for arg in include:
                        if arg in self.egraph[inode][jnode][jedge]:
                            estruct.egraph[knode][lnode][jedge][arg] = \
                                deepcopy(self.egraph[inode][jnode][jedge][arg])

        return estruct

    def compute_bloch_phase(self, kpt=[0.,0.,0.]):

        for inode, jnode, jedge, data in  self.graph.edges(keys=True,data=True):
            vec = data['vector']
            self.egraph[inode][jnode][jedge]['bloch'] =  \
                    self._bloch.non_local_block(vec, kpt) 

    def compute_S(self, include=[], bloch=False, spinscalar=True):

        ovlp = np.zeros((self.basis.size, self.basis.size, \
                        self.spin.dim[0], self.spin.dim[1]),dtype=complex)

        if len(include) == 0:
            include = [key for key in self.inter]

        slocs = [el for el in include if (self.inter[el].overlap and \
                   hasattr(self.inter[el],'local_block'))]  

        for inode, data in self.egraph.nodes(data=True):

            mat = deepcopy(data[slocs[0]])
            for arg in slocs[1:]:
                mat += data[arg]

            istart = self.basis.first_basis_on_site(inode)
            istop = self.basis.last_basis_on_site(inode)+1
            ovlp[istart:istop,istart:istop,:,:] += mat
 
        snlocs = [el for el in include if (self.inter[el].overlap and \
                   hasattr(self.inter[el],'non_local_block'))]  
          
        sphase = [el for el in include if (self.inter[el].peierls and \
                   hasattr(self.inter[el],'non_local_block'))]  
  
        for inode, jnode, jedge, data in  self.egraph.edges(keys=True,data=True): 

            mat = deepcopy(data[snlocs[0]])
            for arg in snlocs[1:]:
                mat = mat + data[arg] 
 
            for arg in sphase:
                mat = mat * data[arg] 

            if bloch:
                mat = mat * data['bloch']

            istart = self.basis.first_basis_on_site(inode)
            istop = self.basis.last_basis_on_site(inode)+1
            jstart = self.basis.first_basis_on_site(jnode)
            jstop = self.basis.last_basis_on_site(jnode)+1
            ovlp[istart:istop,jstart:jstop,:,:] += mat
    
        if spinscalar:
            return self.spin.scalar_operator(ovlp)
        else:
            return ovlp

    def compute_H(self, include=[], bloch=False, spinscalar=True):

        ham = np.zeros((self.basis.size, self.basis.size, \
                        self.spin.dim[0], self.spin.dim[1]),dtype=complex)

        if len(include) == 0:
            include = [key for key in self.inter]

        hlocs = [el for el in include if (self.inter[el].hamiltonian and \
                   hasattr(self.inter[el],'local_block'))]  

        for inode, data in self.egraph.nodes(data=True):

            mat = deepcopy(data[hlocs[0]])
            for arg in hlocs[1:]:
                mat = mat + data[arg]

            istart = self.basis.first_basis_on_site(inode)
            istop = self.basis.last_basis_on_site(inode)+1
            ham[istart:istop,istart:istop,:,:] += mat
 
 
        hnlocs = [el for el in include if (self.inter[el].hamiltonian and \
                   hasattr(self.inter[el],'non_local_block'))]  
        
        hphase = [el for el in include if (self.inter[el].peierls and \
                   hasattr(self.inter[el],'non_local_block'))]  
  
        for inode, jnode, jedge, data in  self.egraph.edges(keys=True,data=True): 

            mat = deepcopy(data[hnlocs[0]])
            for arg in hnlocs[1:]:
                mat = mat + data[arg] 

            for arg in hphase:
                mat = mat * data[arg] 

            if bloch:
                mat = mat * data['bloch']

            istart = self.basis.first_basis_on_site(inode)
            istop = self.basis.last_basis_on_site(inode)+1
            jstart = self.basis.first_basis_on_site(jnode)
            jstop = self.basis.last_basis_on_site(jnode)+1
            ham[istart:istop,jstart:jstop,:,:] += mat

        if spinscalar:
            return self.spin.scalar_operator(ham)
        else:
            return ham

    def compute_HS(self, include=[], bloch=False, spinscalar=True):
        
        ham = self.compute_H(include=include, bloch=bloch, spinscalar=spinscalar)
        ovlp = self.compute_S(include=include, bloch=bloch, spinscalar=spinscalar)
    
        return ham, ovlp

    def sparse_solveigh(self, include=[], bloch=False, **kwargs):

        eigenvals = []

        if not self.orthogonal:
            H, S = self.compute_HS(include=include, bloch=bloch)  

            for ispin in range(self.spin.nspin):
                H_sparse = sparse.csc_matrix(H[:,:,ispin])
                S_sparse = sparse.csc_matrix(S[:,:,ispin])
                eigh_vals = sparse_eigh(H_sparse, \
                    M=S_sparse, return_eigenvectors=False, \
                    **kwargs)
                eigenvals.append(eigh_vals)

        else:
            H = self.compute_H(include=include, bloch=bloch)               

            for ispin in range(self.spin.nspin):
                H_sparse = sparse.csc_matrix(H[:,:,ispin])
                eigh_vals = sparse_eigh(H_sparse, \
                    return_eigenvectors=False, **kwargs)
                eigenvals.append(eigh_vals)

        # Transpose to achieve the order [iband, ispin] and
        # [iband, ispin, iorb] for the eigenvalues and eigenvectors 
        return np.transpose(np.array(eigenvals),(1,0))

    def solveigh(self, include=[], eigvec=False, bloch=False):

        eigenvecs = []
        eigenvals = []

        if not self.orthogonal:
            H, S = self.compute_HS(include=include, bloch=bloch)  
            for ispin in range(self.spin.nspin):
                if eigvec:
                   eigh_vals, eigh_vecs = linalg.eigh(H[:,:,ispin], \
                       b=S[:,:,ispin],\
                       lower=False, eigvals_only=False)
                   eigenvals.append(eigh_vals)
                   eigenvecs.append(eigh_vecs.T)
                else:
                   eigh_vals = linalg.eigh(H[:,:,ispin], \
                       b=S[:,:,ispin],\
                       lower=False, eigvals_only=True)
                   eigenvals.append(eigh_vals)

        else:
            H = self.compute_H(include=include, bloch=bloch)               
            for ispin in range(self.spin.nspin):
                if eigvec:
                   eigh_vals, eigh_vecs = linalg.eigh(H[:,:,ispin],\
                       lower=False, eigvals_only=False)
                   eigenvals.append(eigh_vals)
                   eigenvecs.append(eigh_vecs.T)
                else:
                   eigh_vals = linalg.eigh(H[:,:,ispin],\
                       lower=False, eigvals_only=True)
                   eigenvals.append(eigh_vals)

        # Transpose to achieve the order [iband, ispin] and
        # [iband, ispin, iorb] for the eigenvalues and eigenvectors 
        if eigvec:
            return np.transpose(np.array(eigenvals),(1,0)), \
                   np.transpose(np.array(eigenvecs),(1,0,2))
        else:
            return np.transpose(np.array(eigenvals),(1,0))

    def lines_kpoints(self, kpoints, nkpts, fractional=True):
        
        kpts = np.empty((sum(nkpts)+1,3))

        ptr = 0
        for il in range(int(len(kpoints)/2)):
            kstart = np.array(kpoints[2*il])
            kstop = np.array(kpoints[2*il+1])
            step = (kstop-kstart)/nkpts[il]
            kpts[ptr:ptr+nkpts[il],:] = np.array([kstart+ik*step for ik in range(nkpts[il])])
            ptr += nkpts[il]

        kpts[ptr] = np.array(kpoints[-1])

        if fractional: 
           kpts = np.array([np.dot(kpt,self.lattice.reciprocal_lattice.matrix) for kpt in kpts])

        return kpts, np.array([1.]*len(kpts))


    def irreducible_kpoints(self, mesh=(0,0,0), is_shift=(0,0,0), shift=(0.,0.,0.)):
        
        if not hasattr(self,'spacegroup'):
            mp = []
            for idir in range(3):
                mp.append(np.array([((2.*(ik+1)-mesh[idir]-1)/(2.*mesh[idir]))+shift[idir] for ik in range(mesh[idir])]))
            kx,ky,kz = np.meshgrid(mp[0],mp[1],mp[2])
            kpts = np.array([np.dot(kpt, self.lattice.reciprocal_lattice.matrix) \
                             for kpt in np.vstack((kx.flatten(),ky.flatten(),kz.flatten())).T])
            wkpts = np.array([1 for ik in range(len(kpts))])
        else:
            kptlist = self.spacegroup.get_ir_reciprocal_mesh(mesh=mesh, is_shift=is_shift)
            kpts = np.array([np.dot(kpt[0],self.lattice.reciprocal_lattice.matrix) for kpt in kptlist])
            kcounts = np.array([np.float(kpt[1]) for kpt in kptlist])
            wkpts = kcounts/np.sum(kcounts)
  
        return kpts, wkpts


    def eigenvalues(self, include=[], compute=[], eigvec=False, gamma=False,\
                 kpts=[[0.,0.,0.]], wkpts=[1.], fractional=False, sparse=False, **kwargs):

        if len(compute) > 0:
            self.compute_integrals(*compute)

        if fractional: 
           kpts = np.array([np.dot(kpt,self.lattice.reciprocal_lattice.matrix) for kpt in kpts])
        else:
           kpts = np.array(kpts)

        if not hasattr(include,'__iter__'):
            include = [include]
      
        eigenset = Eigenset(self.lattice, spin=self.spin, kpts=kpts, wkpts=wkpts, fractional=False)

        bloch = not gamma
        for ikpt, kpt in enumerate(kpts):  

            if not gamma:
                self.compute_bloch_phase(kpt=kpt)
        
            if sparse:

                eigenvals = self.sparse_solveigh(include=include, bloch=bloch, **kwargs)
                eigenset.set_eigenvalues_k(ikpt, E=eigenvals)

            else:

                if eigvec:
                    eigenvals, eigenvecs = self.solveigh(include=include, eigvec=eigvec, bloch=bloch)
                    eigenset.set_eigenvalues_k(ikpt, E=eigenvals, V=eigenvecs) 
                else:
                    eigenvals = self.solveigh(include=include, eigvec=eigvec, bloch=bloch)
                    eigenset.set_eigenvalues_k(ikpt, E=eigenvals) 

        return eigenset 


    def eigen_on_grid(self, include=[], compute=[], eigvec=False,\
                      mesh=(0,0,0), is_shift=(0,0,0), sparse=False, **kwargs):

        kpts, wkpts = self.irreducible_kpoints(mesh=mesh, is_shift=is_shift)
 
        return self.eigenvalues(include=include, compute=compute, eigvec=eigvec, gamma=False,\
                              kpts=kpts, wkpts=wkpts, fractional=False, sparse=sparse, **kwargs)
         
    def bands_structure(self, include=[], compute=[], eigvec=False,\
                       lines=[], nkpts=[], kpts=[], wkpts=[], fractional=True):

        if len(kpts) != 0:
            if fractional: 
               kpts = np.array([np.dot(kpt,self.lattice.reciprocal_lattice.matrix) for kpt in kpts])
            else:
               kpts = np.array(kpts)
        elif len(lines) != 0:
            kpts, wkpts = self.lines_kpoints(lines, nkpts, fractional=fractional)

        return self.eigenvalues(include=include, compute=compute, eigvec=eigvec, gamma=False,\
                              kpts=kpts, wkpts=wkpts, fractional=False)

    @property
    def optim_variables(self):

        optim_variables = []
        for key in self.inter:
            optim_variables = optim_variables + self.inter[key].optim_variables

        return optim_variables

    def update_optim_variables(self, **kwargs):

        for key in self.inter:
            #print 'update_optim_vairables : ', key
            self.inter[key].update_optim_variables(**kwargs)


