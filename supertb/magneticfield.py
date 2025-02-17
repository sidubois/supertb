"""
This module defines the class that enables to deal with constant magnetic fields
"""

import numpy as np
import scipy.optimize as opt
from math import exp, log, sin, cos, pi, sqrt, trunc, ceil
from copy import deepcopy
import scipy
import scipy.constants as const
from supertb import Spinor, mpauli
from .utils import ratio_factor
from .tightbinding import ParametrizedInteraction
from .structure import Lattice

class MagneticPhase(ParametrizedInteraction):

    def __init__(self, table, **kwargs):

        ParametrizedInteraction.__init__(self, table, \
            hamiltonian=False, overlap=False, peierls=True, **kwargs)

        # Defaut gauge shift
        self._gauge_shift = np.array([0.,0.,0.])

        # Defaut integration options
        self.iopt = {}
        self.iopt['method'] = 'fixed'
        self.iopt['tol'] = 0.0001
        self.iopt['rtol'] = 0.001
        self.iopt['divmax'] = 100
        self.iopt['maxiter'] = 100
        self.iopt['miniter'] = 5
        self.iopt['ndiv'] = 8 
        self.iopt['err'] = False
        self.iopt['epsabs'] = 0.0001
        self.iopt['epsrel'] = 0.001
        self.iopt['limit'] = 100

    @property
    def integration_options(self):
        return self.iopt

    @property
    def gauge_shift(self):
        return self._gauge_shift

    @gauge_shift.setter
    def gauge_shift(self, gshift):
        self._gauge_shift = gshift

    def set_magnetic_flux(self, lattice, nquanta=1, nlines=1, ecut=4.):
        """   
        Defines the magnetic flux across the unit-cell and
        computes the singular gauge transformation used to obtain
        a periodic vector potential.

        Args:
            lattice: (Lattice Object)
                Lattice defining the periodicity of atomic structure.
            nquanta: (integer)
                Number of magnetic flux quanta per flux line.
                Default is 1.
            nlines: (integer)
                Number of magnetic flux lines within the simulation cell.
                Default is 1.
            ecut: (float)
                Cutoff energy in Hartree for the expansion of
                periodic vector potential.
                Default is 20.
        """

        # Compute the magnetic flux lines lattice
        Blattice = MagneticPhase.flux_lines_lattice(lattice, nlines)
        b1 = Blattice.reciprocal_lattice.matrix[0]
        b2 = Blattice.reciprocal_lattice.matrix[1]
        S = np.linalg.norm(np.cross(Blattice.matrix[0],Blattice.matrix[1]))
        self.Blattice = Blattice
        self.nquanta = nquanta

        # Compute the wavevectors that enter the fourier
        # transform of the periodic vector potential up to
        # a given cutoff energy. Note that in atomic units 
        # Gmax = (2*m*ecut)^0.5/hbar reduces to (2*ecut)^0.5        
        Gmax = (2.*ecut)**0.5
        b1p = b1 - b2*(np.dot(b1,b2)/np.linalg.norm(b2)**2)
        b2p = b2 - b1*(np.dot(b2,b1)/np.linalg.norm(b1)**2)
        imax = int(ceil(Gmax/np.linalg.norm(b1p)))
        jmax = int(ceil(Gmax/np.linalg.norm(b2p)))
 
        # Compute the expansion coefficients of the 2D 
        # potential Xi as described in PRL91,056405(2003),
        # Eqs 18 & 19.
        Xi_basis = []
        Xi_coeff = []
        for ig in range(-imax-1, imax+2):
            for jg in range(-jmax-1, jmax+2):
                if ig != 0 or jg != 0:
                    G = ig*b1 + jg*b2
                    Gn = np.linalg.norm(G)
                    if Gn < Gmax :
                        coeff = -((-1.)**(ig+jg+ig*jg))*np.exp(-S*(Gn**2)/(8*pi))
                        Xi_basis.append(G)
                        Xi_coeff.append(coeff)

        self.Xi_basis = np.array(Xi_basis)
        self.Xi_coeff = np.array(Xi_coeff) 


    @classmethod
    def flux_lines_lattice(cls, lattice, nlines):
        """   
        Computes the magnetic flux lines lattice that nullifies the
        average magnetic flux over the unit cell.
        """

        mat = lattice.matrix
        ratio = np.linalg.norm(mat[0,:])/np.linalg.norm(mat[1,:])
        div = ratio_factor(nlines,ratio)
        Bmat = np.zeros((3,3))
        Bmat[0] = mat[0]/div[0]
        Bmat[1] = mat[1]/div[1]
        Bmat[2] = mat[2]

        return Lattice(Bmat)

    @property
    def magnetic_field_amplitude(self):
        """   
        Computes the actual value of the magnetic field in Tesla.
        """

        if hasattr(self,'Blattice'):

            m = self.Blattice.matrix
            S = np.linalg.norm(np.cross(m[0,:],m[1,:]))
            B = 1./np.linalg.norm(S)
            convfac = const.physical_constants["mag. flux quantum"][0]/1.0e-20
            # Note that B is in unit of [Phi0/Ang**2],
            # B*convfac is converted to Tesla.

            return B*convfac

        else:

            return 0.


    def vector_potential(self, point):
        """
        Evaluates the vector potential at a point in space 
        as described in Eq.14 of PRL 91, 056405 (2003),
        in units of Phi_0= h/2e.

        Arg:
            point: (1D array)
                Cartesian coordinates
        """

        ez = np.array([0.,0.,1.])
        x = np.array(point)-self.gauge_shift

        gx = np.dot(x,self.Xi_basis.T)
        xi = np.sum(((1.-np.cos(gx))*self.Xi_coeff),axis=-1)
        grad_pot =-np.divide(np.dot(np.sin(gx)*self.Xi_coeff, self.Xi_basis).T,4*pi*xi).T
        vec_pot = np.cross(ez,grad_pot)

        return self.nquanta*vec_pot

    def peierls_integrand(self, frac, start, stop):
        """
        Estimates the function that enters the line integral associated with the
        Peierls Phase factors induced by a magnetic field:
        phi = (-ie/hbar) \int_l A.dl
        """

        xfrac = np.array(frac)
        vec = np.array(stop)-np.array(start)

        # deal with multiple line segments at once,
        # i.e. start and stop are 2d-arrays
        if len(vec.shape) > 1:
            rshape = [(i,j) for j in range(vec.shape[0]) for i in range(len(frac))]
            point = np.array([np.outer(xfrac,vec)[i,j*3:(j+1)*3]+start[j] for i,j in rshape])
            vec_pot = self.vector_potential(point)
            return np.array([np.dot(vec_pot[i*len(frac):(i+1)*len(frac)],vec[i]) for i in range(vec.shape[0])])

        # deal with a single line segment,
        # i.e. start and stop are 1d-arrays
        else:
            point = start+np.outer(xfrac,vec)
            vec_pot = self.vector_potential(point)
            return np.dot(vec_pot,vec.T)

    def peierls_weight_integrand(self, frac, start, stop):
        """
        Estimates the function that enters the line integral associated with the
        Peierls Phase factors induced by a magnetic field:
        phi = (-ie/hbar) \int_l A.dl
        """

        xfrac = np.array(frac)
        vec = np.array(stop)-np.array(start)

        # deal with multiple line segments at once,
        # i.e. start and stop are 2d-arrays
        if len(vec.shape) > 1:
            rshape = [(i,j) for j in range(vec.shape[0]) for i in range(len(frac))]
            point = np.array([np.outer(xfrac,vec)[i,j*3:(j+1)*3]+start[j] for i,j in rshape])
            vec_pot = self.vector_potential(point)
            return np.array([np.linalg.norm(vec_pot[i*len(frac):(i+1)*len(frac)],axis=-1) for i in range(vec.shape[0])])

        # deal with a single line segment,
        # i.e. start and stop are 1d-arrays
        else:
            point = start+np.outer(xfrac,vec)
            vec_pot = self.vector_potential(point)
            return np.linalg.norm(vec_pot, axis=-1)

    def set_integration_options(self, **kwargs):

        if 'method' in kwargs:
            self.iopt['method'] = kwargs['method']

        if 'tol' in kwargs:
            self.iopt['tol'] = kwargs['tol']
        if 'rtol' in kwargs:
            self.iopt['rtol'] = kwargs['rtol']
        if 'divmax' in kwargs:
            self.iopt['divmax'] = kwargs['divmax']
        if 'maxiter' in kwargs:
            self.iopt['maxiter'] = kwargs['maxiter']
        if 'miniter' in kwargs:
            self.iopt['miniter'] = kwargs['miniter']

        if 'epsabs' in kwargs:
            self.iopt['epsabs'] = kwargs['epsabs']
        if 'epsrel' in kwargs:
            self.iopt['epsrel'] = kwargs['epsrel']
        if 'limit' in kwargs:
            self.iopt['limit'] = kwargs['limit']

        if 'ndiv' in kwargs:
            self.iopt['ndiv'] = kwargs['ndiv']

        if 'err' in kwargs:
            self.iopt['err'] = kwargs['err']
        
    def peierls_phase(self, start, stop):
        """
        Estimates the Peierls Phase factors induced by a magnetic field by 
        means of the adaptative quadrature available within scipy
        """

        integrand = lambda x: self.peierls_integrand(x, start, stop)
        if self.iopt['method'] == 'quadrature':
            integral = scipy.integrate.quadrature(integrand,0.,1.,\
                            tol=self.iopt['tol'],\
                            rtol=self.iopt['rtol'],\
                            maxiter=self.iopt['maxiter'],\
                            miniter=self.iopt['miniter'],\
                            vec_func=True)
            if self.iopt['err']:
                return 1j*2*pi*integral[0], 2*pi*integral[1]
            else:
                return 1j*2*pi*integral[0]

        elif self.iopt['method'] == 'romberg':
            integral = scipy.integrate.romberg(integrand,0.,1.,\
                            tol=self.iopt['tol'],\
                            rtol=self.iopt['rtol'],\
                            divmax=self.iopt['divmax'],\
                            show=False,vec_func=True)
            return 1j*2*pi*integral

        elif self.iopt['method'] == 'quad':
            integral = scipy.integrate.quad(integrand,0.,1.,\
                            epsabs=self.iopt['epsabs'],\
                            epsrel=self.iopt['epsrel'],\
                            limit=self.iopt['limit'])
            if self.iopt['err']:
                return 1j*2*pi*integral[0], 2*pi*integral[1]
            else:
                return 1j*2*pi*integral[0]

        else:
            x, w = np.polynomial.legendre.leggauss(self.iopt['ndiv'])
            frac = (0.5*x + 0.5)
            weigth = 0.5*w
            integrand = self.peierls_integrand(frac,start,stop)
            return 1j*2*pi*np.dot(integrand,weigth)

    def peierls_weight(self, start, stop, ndiv=8):
        """
        Estimates the Peierls Weights (PW = \int A^2 dr) associated with the 
        Peierls phase along the atomic bonds.
        """

        x, w = np.polynomial.legendre.leggauss(ndiv)
        frac = (0.5*x + 0.5)
        weigth = 0.5*w
        integrand = self.peierls_weight_integrand(frac,start,stop)
        return 2*pi*np.dot(integrand,weigth)

    def peierls_weight_over_graph(self, graph, ndiv=8):

        starts = []
        stops = []
        for i,j,k in graph.edges_iter(data=False, keys=True):
            start = graph.graph['cart_coords'][i]
            vec = graph[i][j][k]['vector']
            starts.append(start)
            stops.append(start+vec)

        return np.sum(self.peierls_weight(starts, stops, ndiv=ndiv)) 


    def optimize_gauge_shift(self, graph, ndiv=8, method='CG', \
                             options={'maxiter':100,'eps':0.001}):
        
        starts = []
        stops = []
        for i,j,k in graph.edges_iter(data=False, keys=True):
            start = graph.graph['cart_coords'][i]
            vec = graph[i][j][k]['vector']
            starts.append(start)
            stops.append(start+vec)

        def shift_penalty(x):
            shift = x[0]*graph.lattice.matrix[0]+\
                    x[1]*graph.lattice.matrix[1]
            self.gauge_shift += shift
            w = np.sum(self.peierls_weight(starts, stops, ndiv=ndiv))
            self.gauge_shift -= shift
            return w

        res = opt.minimize(shift_penalty,[0.,0.],method=method,options=options)
        shift = res.x[0]*graph.lattice.matrix[0]+\
                res.x[1]*graph.lattice.matrix[1]
        self.gauge_shift += shift
    
    def non_local_block(self, edge, graph):
    
        i, j, k = edge
        data = graph[i][j][k]
        vec = data['vector']
        spec_a = graph.graph['species'][i]
        spec_b = graph.graph['species'][j]
    
        num_a = self.table[spec_a].num_orbitals
        num_b = self.table[spec_b].num_orbitals
        mat = np.ones((num_a, num_b, self.spin.dim[0],self.spin.dim[1]),dtype=complex)
    
        start = graph.graph['cart_coords'][i]
        stop = start + vec
        phase = self.peierls_phase(start, stop)
        #print 'field:', i,j,k, phase
    
        return mat*np.exp(phase)
    
    
    
    


    def non_local_block2(self, edge, graph):
    
        i, j, k = edge
        spec_a = graph.graph['species'][i]
        spec_b = graph.graph['species'][j]
    
        num_a = self.table[spec_a].num_orbitals
        num_b = self.table[spec_b].num_orbitals
        mat = np.ones((num_a, num_b, self.spin.dim[0],self.spin.dim[1]),dtype=complex)
    
        path = graph[i][j][k]['path']
    
        phase = 0.+1j*0.
        start = deepcopy(graph.graph['cart_coords'][i])
        for iteredge in path:
            l,m,n = iteredge
            vec = deepcopy(graph[l][m][n]['vector'])
            stop = start + vec
            phase += self.peierls_phase(start, stop)
            start += vec
        #print 'field:', i,j,k, phase, path
        
        return mat*np.exp(phase)

    def non_local_phase_mod(self, edge, graph):
    
        i, j, k = edge
        vec = graph[i][j][k]['vector']
        start = graph.graph['cart_coords'][i]
        stop = start + vec
        phase = self.peierls_phase(start, stop)    

        spec_a = graph.graph['species'][i]
        spec_b = graph.graph['species'][j]
        num_a = self.table[spec_a].num_orbitals
        num_b = self.table[spec_b].num_orbitals
        mat = np.ones((num_a, num_b, self.spin.dim[0],self.spin.dim[1]),dtype=complex)

        return phase*mat


    def non_local_phase(self, edge, graph):
    
        i, j, k = edge
        start = graph.graph['cart_coords'][i]
        stop = start + graph[i][j][k]['vector']
        phase = self.peierls_phase(start, stop)    

        return np.exp(phase)



