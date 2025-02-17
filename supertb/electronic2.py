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
from supertb import AtomicBasisSet, Eigenset, BlochPhase, Spinor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings

#from memory_profiler import profile

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
        for arg in args:
            if  hasattr(self.inter[arg],'local_block'):
                local_inter.append(arg)
            if  hasattr(self.inter[arg],'non_local_block'):
                non_local_inter.append(arg)

        for inode in self.graph.nodes_iter():
            for arg in local_inter:
                self.egraph.node[inode][arg] = \
                    self.inter[arg].local_block(inode, self.graph) 

        for edge in  self.graph.edges_iter(keys=True): 
            inode, jnode, jedge = edge
            for arg in non_local_inter:
                self.egraph[inode][jnode][jedge][arg] =  \
                    self.inter[arg].non_local_block(edge, self.graph) 


    def complex_shift(self, eshift=0., iener=0.001):

        for inode in self.graph.nodes_iter(data=False):
            for arg in decimate:
                if arg in self.egraph.node[inode]:
                    self.egraph.node[inode][arg] +=  eshift + 1j*iener


    def decimate_orbitals(self, orbitals, decimate=[], spacegroup=None, eshift=0., iener=0.001, path=True):


        if len(decimate) == 0:
            decimate = [key for key in self.inter]
        
        self.complex_shift(eshift=eshift, iener=iener)

        itergraph = self.graph.empty_graph()    
    
        for inode in itergraph.nodes_iter(data=False):
            spec = self.graph.graph['species'][inode]

            if spec in orbitals:
                for iorb in orbitals[spec]:
                    for arg in decimate:
                        if arg not in self.egraph.node[inode]:
                            next
                        nodemat = self.spin.spinor_inv(self.egraph.node[inode][arg][[iorb],[iorb],:,:])

                        ## Orbitals on same site
                        for lorb in range(self.table[spec].num_orbitals): 
                            if lorb in orbitals[spec]:
                                next
                            tmp_mat = self.egraph.node[inode][arg][[lorb][iorb],:,:]
                            imat = self.spin.spinor_product(tmp_mat,nodemat)
                            for morb in range(self.table[spec].num_orbitals): 
                                if morb in orbitals[spec]:
                                    next
                                tmp_mat = self.egraph.node[inode][arg][[iorb][morb],:,:,]
                                ijmat = self.spin.spinor_product(imat,tmp_mat)
                                self.egraph.node[inode][arg][[lorb][morb],:,:] += ijmat
                                
                        ## Orbitals on other sites
                        for ii in itergraph[inode]:
                            ispec = self.graph.graph['species'][ii]
                            for lorb in range(self.table[ispec].num_orbitals):
                                if ispec in orbitals:
                                    if lorb in orbitals[ispec]:
                                        next 
                                for mm in itergraph[ii][inode]:
                                    if arg not in self.egraph[ii][inode][mm]:
                                        next
                                    ivec = self.graph[ii][inode][mm]['vector']
                                    tmp_mat = self.egraph[ii][inode][mm][arg][[lorb][iorb],:,:]
                                    imat = self.spin.spinor_product(tmp_mat,nodemat) 
                                    for jj in itergraph[inode]:
                                        jspec = self.graph.graph['species'][jj]
                                        for morb in range(self.table[jspec].num_orbitals): 
                                            if jspec in orbitals:
                                                if morb in orbitals[jspec]:
                                                    next 
                                            for nn in itergraph[inode][jj]:
                                                if arg not in self.egraph[inode][jj][nn]:
  						    next
					        jvec = self.graph[inode][jj][nn]['vector']
                                                ijvec = ivec + jvec
                                                tmp_mat = self.egraph[inode][jj][nn][arg][[iorb][morb],:,:]
                                                ijmat = self.spin.spinor_product(imat,tmp_mat)

                                                decimated = False
                                                if np.linalg.norm(ijvec) < 1.e-6:
						    self.egraph.node[jj][arg][[lorb][morb],:,:] += ijmat
                                                    decimated = True

                                                if jj in self.egraph[ii] and decimated = False:
						    for pp in self.egraph[ii][jj]:
 						        pvec = self.graph[ii][jj][pp]['vector']
							if np.linalg.norm(ijvec-pvec) < 1.e-6:
							    decimated = True
 							    if arg in self.egraph[ii][jj][pp]:
							        self.egraph[ii][jj][pp][arg] += -ijmat





    def decimate_species(self, species, include=[], decimate=[], spacegroup=None, eshift=0., iener=0.001, path=True):

        if len(decimate) == 0:
            decimate = [key for key in self.inter]

        for inode in self.graph.nodes_iter(data=False):
            for arg in decimate:
                if arg in self.egraph.node[inode]:
                    self.egraph.node[inode][arg] +=  eshift + 1j*iener

        itergraph = self.graph.empty_graph()

        for inode in itergraph.nodes_iter(data=False):
            spec = self.graph.graph['species'][inode]

            if spec in species:

                #print "Decimation ", inode, spec

                for arg in decimate:

                    if arg not in self.egraph.node[inode]:
                        next

                    nodemat = self.spin.spinor_inv(self.egraph.node[inode][arg])
                    for ii in itergraph[inode]:

                        ispec = self.graph.graph['species'][ii]
                        if ispec in species:
                            next

                        for mm in itergraph[ii][inode]:

                            if arg not in self.egraph[ii][inode][mm]:
                                next

                            ivec = self.graph[ii][inode][mm]['vector']
                            imat = self.spin.spinor_product(self.egraph[ii][inode][mm][arg],nodemat)

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

                                    if np.linalg.norm(ijvec) < 1.e-6:
                                        self.egraph.node[jj][arg] += ijmat

                                    decimated = False
                                    if jj in self.egraph[ii]:

                                        for pp in self.egraph[ii][jj]:
                                            pvec = self.graph[ii][jj][pp]['vector']

        			            if np.linalg.norm(ijvec-pvec) < 1.e-6:
                                                decimated = True
                                                if arg in self.egraph[ii][jj][pp]:
                                                    self.egraph[ii][jj][pp][arg] += -ijmat 
                                                else:
                                                    self.egraph[ii][jj][pp][arg] = -deepcopy(ijmat) 
                                                break

                                    if not decimated:                                               
                                        self.graph.add_edge(ii, jj, vector=deepcopy(ijvec))
                                        kw = {arg:deepcopy(-ijmat)}
                                        self.egraph.add_edge(ii, jj, **kw)

        graph, mapping = self.graph.remove_nodes_by_species(species)
        for i,j,k,data in graph.edges_iter(keys=True,data=True):
            if not 'order' in data:
                vec = data['vector']
                path = graph.underlying_path(i,j,vec,itermax=12)
                order = len(path)
                graph[i][j][k]['order'] = order
                graph[i][j][k]['path'] = deepcopy(path)

        estruct = ElectronicStructure(graph, self.table, \
                      spinpol=self.spin.spinpol, spinorb=self.spin.spinorb, \
                      spacegroup=spacegroup, **self.inter)


        if len(include) == 0:
            include = [key for key in self.inter]

        for inode, data in self.egraph.nodes_iter(data=True):

            if inode in mapping:
                jnode = mapping[inode]
                for arg in include:
                    if arg in self.egraph.node[inode]:
                        estruct.egraph.node[jnode][arg] = \
                            deepcopy(self.egraph.node[inode][arg])

        for inode, jnode, jedge, data in  self.egraph.edges_iter(keys=True,data=True): 

            if inode in mapping:
                if jnode in mapping:
                    knode = mapping[inode] 
                    lnode = mapping[jnode] 
                    
                    for arg in include:
                        if arg in self.egraph[inode][jnode][jedge]:
                            estruct.egraph[knode][lnode][jedge][arg] = \
                                deepcopy(self.egraph[inode][jnode][jedge][arg])

        return estruct

    def compute_bloch_phase(self, kpt=[0.,0.,0.]):

        for inode, jnode, jedge, data in  self.graph.edges_iter(keys=True,data=True):
            vec = data['vector']
            self.egraph[inode][jnode][jedge]['bloch'] =  \
                    self._bloch.non_local_block(vec, kpt) 

    def compute_S(self, include=[], bloch=False):

        ovlp = np.zeros((self.basis.size, self.basis.size, \
                        self.spin.dim[0], self.spin.dim[1]),dtype=complex)

        if len(include) == 0:
            include = [key for key in self.inter]

        slocs = [el for el in include if (self.inter[el].overlap and \
                   hasattr(self.inter[el],'local_block'))]  

        for inode, data in self.egraph.nodes_iter(data=True):

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
  
        for inode, jnode, jedge, data in  self.egraph.edges_iter(keys=True,data=True): 

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

        return self.spin.scalar_operator(ovlp)

    def compute_H(self, include=[], bloch=False):

        ham = np.zeros((self.basis.size, self.basis.size, \
                        self.spin.dim[0], self.spin.dim[1]),dtype=complex)

        if len(include) == 0:
            include = [key for key in self.inter]

        hlocs = [el for el in include if (self.inter[el].hamiltonian and \
                   hasattr(self.inter[el],'local_block'))]  

        for inode, data in self.egraph.nodes_iter(data=True):

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
  
        for inode, jnode, jedge, data in  self.egraph.edges_iter(keys=True,data=True): 

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

        return self.spin.scalar_operator(ham)

    def compute_HS(self, include=[], bloch=False):
        
        ham = self.compute_H(include=include, bloch=bloch)
        ovlp = self.compute_S(include=include, bloch=bloch)
    
        return ham, ovlp

    def sparse_solveigh(self, include=[], bloch=False, **kwargs):

        eigenvals = []

        #t0 = time.clock()
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

        #t0 = time.clock()
        if not self.orthogonal:
            H, S = self.compute_HS(include=include, bloch=bloch)  
            for ispin in range(self.spin.nspin):
                if eigvec:
                   eigh_vals, eigh_vecs = linalg.eigh(H[:,:,ispin], \
                       b=S[:,:,ispin],\
                       lower=False, turbo=True, eigvals_only=False)
                   eigenvals.append(eigh_vals)
                   eigenvecs.append(eigh_vecs.T)
                else:
                   eigh_vals = linalg.eigh(H[:,:,ispin], \
                       b=S[:,:,ispin],\
                       lower=False, turbo=True, eigvals_only=True)
                   eigenvals.append(eigh_vals)

        else:
            H = self.compute_H(include=include, bloch=bloch)               
            for ispin in range(self.spin.nspin):
                if eigvec:
                   eigh_vals, eigh_vecs = linalg.eigh(H[:,:,ispin],\
                       lower=False, turbo=True, eigvals_only=False)
                   eigenvals.append(eigh_vals)
                   eigenvecs.append(eigh_vecs.T)
                else:
                   eigh_vals = linalg.eigh(H[:,:,ispin],\
                       lower=False, turbo=True, eigvals_only=True)
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
	for il in range(len(kpoints)/2):
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

        if kpts != []:
            if fractional: 
               kpts = np.array([np.dot(kpt,self.lattice.reciprocal_lattice.matrix) for kpt in kpts])
            else:
               kpts = np.array(kpts)
        elif lines != []:
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
            self.inter[key].update_optim_variables(**kwargs)


