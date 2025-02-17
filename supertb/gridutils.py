""" Python Classes and functions
Simple Pyhton classes and functions providing some geometrical functionalities
to handle atomic systems
""" 

import numpy as np
import itertools
import pymatgen as pm
import scipy.special
import scipy.misc
from scipy.interpolate import CubicSpline
from math import sin, cos, asin, acos, pi, radians
from .structure import Lattice
import sys
from supertb import NumericalAtom, AtomicBasisSet
from copy import deepcopy

################################################################################
# Various geometrical functions
################################################################################

def corner_vertices_iter(frac):
    """ Filter points with respect to a bounding box

    Parameters:
    ----------
    frac: 1d-array 
         Fractional coordinates of a point

    Returns:
    -------
    corner_vertices_iter: iterator over array_like of int 
         The fractional coordinates of cell vertices surrounding the point

    """
    return itertools.product(*[[np.int(np.floor(ifrac)), np.int(np.ceil(ifrac))] for ifrac in frac])

def corner_vertices(frac):
    """ Filter points with respect to a bounding box

    Parameters:
    ----------
    frac: 1d-array 
         Fractional coordinates of a point

    Returns:
    -------
    corner_vertices: 2D array of int 
         The set of fractional coordinates of the vertices surrounding the point

    """
    return np.array(list(itertools.product(*[[np.floor(ifrac), np.ceil(ifrac)] for ifrac in frac]))).astype(np.int)

def closest_nodes(lattice, coords, coords_are_cartesian=True):

    pts = np.array(coords)
    d = np.zeros(pts.shape[0])
    l = np.zeros(pts.shape,dtype=np.int)

    if coords_are_cartesian:
        fracs = lattice.get_fractional_coords(pts)
    else:
        fracs = pts
    for i in range(fracs.shape[0]):
        vertices = corner_vertices(fracs[i])
        dist = np.linalg.norm(lattice.get_cartesian_coords(vertices-fracs[i]), axis=-1)
        idx = dist.argmin()

        d[i] = dist[idx]
        l[i] = vertices[idx]

    return d, l

def bounding_box_filter(p, pmin, pmax):

    x = np.array(p)
    lmin = np.array(pmin).flatten()
    lmax = np.array(pmax).flatten()
    if (x.shape[-1]!=lmin.shape[0]) or (x.shape[-1]!=lmax.shape[0]):
        raise ValueError('Shape mismatch')
   
    #print (x.shape, lmin.shape, lmax.shape) 
    bound = [np.logical_and(x[:,i] > lmin[i], x[:,i] < lmax[i])  for i in range(x.shape[-1])]

    return np.logical_and(*bound)

def super_centered_iter(n):
    
    return itertools.product(*[list(itertools.chain(range(x+1),range(-x,0))) for x in n])

def super_centered(n):
    
    return np.array(list(itertools.product(*[list(itertools.chain(range(x+1),range(-x,0))) for x in n])))

def super_increment_max(cmin, cmax, d):
    
    return itertools.chain(itertools.product(range(cmax[0],cmax[0]+d[0]),range(cmin[1],cmax[1]), range(cmin[2],cmax[2])),
                    itertools.product(range(cmin[0],cmax[0]+d[0]), range(cmax[1],cmax[1]+d[1]), range(cmin[2],cmax[2])),
                    itertools.product(range(cmin[0],cmax[0]+d[0]), range(cmin[1],cmax[1]+d[1]), range(cmax[2],cmax[2]+d[2])))
    
    
def super_increment_min(cmin, cmax, d):
    
    return itertools.chain(itertools.product(range(cmin[0]+d[0],cmin[0]),range(cmin[1],cmax[1]), range(cmin[2],cmax[2])),
                    itertools.product(range(cmin[0]+d[0],cmax[0]), range(cmin[1]+d[1],cmin[1]), range(cmin[2],cmax[2])),
                    itertools.product(range(cmin[0]+d[0],cmax[0]), range(cmin[1]+d[1],cmax[1]), range(cmin[2]+d[2],cmin[2])))


def cart2spherical(xyz):
    p = np.array(xyz)
    pnew = np.zeros(p.shape)
    xy = p[:,0]**2 + p[:,1]**2
    pnew[:,0] = np.sqrt(xy + p[:,2]**2)
    pnew[:,1] = np.arctan2(np.sqrt(xy), p[:,2])
    pnew[:,2] = np.arctan2(p[:,1], p[:,0])
    return pnew

def direct_real_spherical_harmonics(l,m,p):

    ylm = np.zeros(p.shape[:-1])

    if l == 0:
        ylm[...] = 1./np.sqrt(4*np.pi)

    elif l == 1:

        r = np.linalg.norm(p,axis=-1)
        fac = np.sqrt(3./(4.*np.pi))
        mask = r!=0
        if m == -1:
            ylm[mask] = fac*p[mask,1]/r[mask]
        elif m == 0:
            ylm[mask] = fac*p[mask,2]/r[mask]
        elif m == 1:
            ylm[mask] = fac*p[mask,0]/r[mask]

    elif l == 2:

        r2 = np.linalg.norm(p,axis=-1)**2
        mask = r2>0
        if m == -2:
            fac = np.sqrt(15./(4.*np.pi))
            ylm[mask] = fac*p[mask,0]*p[mask,1]/r2[mask]

        elif m == -1:
            fac = np.sqrt(15./(4.*np.pi))
            ylm[mask] = fac*p[mask,1]*p[mask,2]/r2[mask]

        elif m == 0:
            fac = np.sqrt(5./(16.*np.pi))
            ylm[mask] = fac*(2.*p[mask,2]**2-p[mask,0]**2-p[mask,1]**2)/r2[mask]

        elif m == 1:
            fac = np.sqrt(15./(4.*np.pi))
            ylm[mask] = fac*p[mask,2]*p[mask,0]/r2[mask]

        elif m == 2:
            fac = np.sqrt(15./(16.*np.pi))
            ylm[mask] = fac*(p[mask,0]**2-p[mask,1]**2)/r2[mask]

    return ylm

def real_spherical_harmonics(m,l,theta,phi):

    if m < 0:
        lpn = scipy.special.lpmv(-m,l,np.cos(theta))
        f1 = scipy.misc.factorial(l+m, exact=True)
        f2 = scipy.misc.factorial(l-m, exact=True)
        fac = np.sqrt(2.*((2.*l+1.)/(4*np.pi))*(np.float(f1)/f2))
        cos = np.sin(-m*phi)
    elif m == 0:
        lpn = scipy.special.lpmv(0,l,np.cos(theta))
        fac = np.sqrt((2.*l+1.)/(4*np.pi))
        cos = 1.
    else:
        lpn = scipy.special.lpmv(m,l,np.cos(theta))
        f1 = scipy.misc.factorial(l-m, exact=True)
        f2 = scipy.misc.factorial(l+m, exact=True)
        fac = np.sqrt(2.*((2.*l+1.)/(4*np.pi))*(np.float(f1)/f2))
        cos = np.cos(m*phi)

    return fac*lpn*cos


def radial_slater(n,zeta,r):

    f1 = scipy.misc.factorial(2*n, exact=True)
    fac = np.sqrt(2.*zeta/f1)*(2.*zeta)**n

    return fac*np.exp(-zeta*r)*r**(n-1)

def radial_slater_localized(n, zeta, nnodes, rcut, normal=True):
    
    def slater_spline_nodes(n, zeta, nnodes, rcut):
        delta = rcut/(nnodes)
        x = np.arange(0.,rcut, delta)
        return radial_slater(n, zeta, x)
    
    
    ynodes = slater_spline_nodes(n,zeta,nnodes,rcut)

    delta = rcut/(len(ynodes))
    x = np.arange(0.,rcut+1*delta, delta)
    y = np.append(ynodes,[0.])
    ftmp = CubicSpline(x,y,bc_type=((2, 0.0), (1, 0.0)))


    if normal:
        fac = 1./ftmp.integrate(0.,rcut)
    else:
        fac = 1.

    def f(r):
        if r <= rcut:
            return fac*ftmp(r)
        else:
            return 0.


    return np.vectorize(f)

################################################################################
        
class PointCollection(object):
    """ 
    Basic collection of points. This is essentially an array of points
    to which weigths can be associated. The class provide methods to 
    define subsets of points and to perform basic set operations.      
    """

    def __init__(self, coords, weigths=[], normalize=False):

        self._coords = np.array(coords)
        self.size = self._coords.shape[0]

        if len(weigths) > 0:
            if len(weigths) == self.size:
                self.weigths = np.array(weigths)
            else:
                raise IndexError('shapes of arguments are not compatible')
        elif self.size > 0 and normalize:
            self.weigths = np.array([1.]*self.size)/self.size
        elif self.size > 0:
            self.weigths = np.array([1.]*self.size)
        else:
            self.weigths = []

        self.subsets = {}
        self.subsets[None] = np.ones(self.size, dtype=bool) 
        
    @property
    def coords(self):
        """
        The coordinates of the points in the collection
        """
        return self._coords
    
    
    def create_subset(self, label, mask):
        
        self.subsets.update({label:mask})
    
    def create_subset_from_list(self, label, list_of_index):
        """
        Defines a subset of points from a list of indices
        """
        mask = np.zeros(self.size, dtype=bool)
        mask[np.unique(list_of_index)] = True
        self.subsets.update({label:mask})
        
    def create_subset_from_sphere(self, label, center, radius):
        """
        Defines a subset of points corresponding to a given spherical
        region
        """
        mask = (np.linalg.norm(self.coords-center, axis=-1) < radius)
        self.subsets.update({label:mask})
        
    def subset_from_sphere(self, center, radius):
        """
        Defines a subset of points corresponding to a given spherical
        region
        """
        return (np.linalg.norm(self.coords-center, axis=-1) < radius)
        
    def subset_coords(self, A):
        """
        Returns the coordinates of points in a given subset
        """
        return self.coords[self.subsets[A]]
    
    def delete_subset(self, label):
        """
        Delete a specified subset. 
        """
        self.subsets.pop(label)
        
    def subsets_intersect(self, A, B, new=None):

        mask = np.logical_and(self.subsets[A],self.subsets[B])
        if new == None:
            return mask
        else:
            self.create_subset(new, mask)
            
    def subsets_union(self, A, B, new=None):

        mask = np.logical_or(self.subsets[A],self.subsets[B])
        if new == None:
            return mask
        else:
            self.create_subset(new, mask)
    
    def subsets_BinA(A, B):
        
        mA = self.subsets[A]
        mB = self.subsets[B]
        return(np.argwhere(mB[mA]).flatten())

    def expand_subset(self, fa):

        mA = self.subsets[fa.subset]
        f = np.zeros(self.size).astype(fa.type)
        f[mA] = fa.values
        return f
 
    def overlap_integral(self, fa, fb):

        mA = self.subsets[fa.subset]
        mB = self.subsets[fb.subset]
        mI = np.logical_and(mA,mB)
    
        inA = np.argwhere(mI[mA]).flatten() 
        inB = np.argwhere(mI[mB]).flatten() 

        if hasattr(self,'weights'):
            return np.sum(np.conj(fa.values[inA])*fb.values[inB]*self.weigths[mI])
        else:
            return np.sum(np.conj(fa.values[inA])*fb.values[inB])    

    def init_field(self, fieldtype, dim=1, subset=None, **kwargs):

        if subset != None:
           size = self.subset_size(subset)
        else:
           size = self.size
 
        idata = np.zeros((size,dim)).astype(fieldtype)

        return FieldOnGrid(idata, fieldtype, subset=subset, **kwargs)

################################################################################>-------

class PeriodicPointCollection(PointCollection):
    """ 
    Collection of points defined with respect to a given lattice. This is basically
    a collection of points whose both cartesian and fractional coordinates are stored.
    """

    def __init__(self, lattice, coords, weigths=[], cartesian=True, normalize = False):

        if isinstance(lattice,Lattice):
            self.lattice = lattice
        else:
            self.lattice = Lattice(lattice)

        if normalize:
            weigths = np.array([1.]*len(coords))/len(coords)

        if cartesian:
            super(PeriodicPointCollection, self).__init__(coords, weigths=weigths)
            if self.size > 0:
                self._frac_coords = self.lattice.get_fractional_coords(self.coords)
        else:
            super(PeriodicPointCollection, self).__init__( \
                  self.lattice.get_cartesian_coords(coords), \
                  weigths=weigths)
            self._frac_coords = coords
            
    @property
    def frac_coords(self):
        """
        The fractional coordinates of the points in the collection
        """
        return self._frac_coords
    
    def subset_frac_coords(self, A):
        """
        Returns the coordinates of points in a given subset
        """
        return self.frac_coords[self.subsets[A]]
    
################################################################################	

class RegularGrid(PeriodicPointCollection):

    def __init__(self, lattice, shifts=[[0.,0.,0.]], ndiv=None, lmax=None, normalize=False):

        if isinstance(lattice,Lattice):
            glattice = lattice
        else:
            glattice = Lattice(lattice)

        if lmax != None:
            na = int(math.ceil(glattice.matrix[0]/lmax[0]))
            nb = int(math.ceil(glattice.matrix[1]/lmax[1]))
            nc = int(math.ceil(glattice.matrix[2]/lmax[2]))

        elif ndiv != None:
            na = int(max(ndiv[0],0.))
            nb = int(max(ndiv[1],0.))
            nc = int(max(ndiv[2],0.))

        self.dim = ((na, nb, nc))        
        pa = np.arange(0.,1.,1./na) 
        pb = np.arange(0.,1.,1./nb) 
        pc = np.arange(0.,1.,1./nc) 
        
        x, y, z = np.meshgrid(pa,pb,pc, indexing='ij')
        prim_coords = np.stack((x,y,z),axis=-1).reshape(-1,3) 

        super(RegularGrid, self).__init__(glattice, prim_coords, \
              cartesian = False, normalize = normalize)
      
        self.x = x
        self.y = y
        self.z = z      

################################################################################

class FieldOnGrid(object):

    def __init__(self, values, fieldtype, subset=None, **kwargs):
        
        self.subset = subset
        self.values = np.array(values)
        self.params = kwargs
        self.type = fieldtype

    @classmethod
    def slater_orbital(cls, n, m, l, zeta, center, grid, **kwargs):

       if 'subset' in kwargs :
           subset = kwargs['subset']
           pts = grid.subset_coords(subset)
       else:
           subset = None
           pts = grid.coords

       r, theta, phi = np.hsplit(cart2spherical(pts-center),3)
       values = real_spherical_harmonics(m,l,theta,phi)*radial_slater(n,zeta,r)

       return FieldOnGrid(values.flatten(), np.float, subset=subset, n=n, m=m, l=l, zeta=zeta, center=center)

    #@classmethod
    #def spline_slater_orbital(cls, n, m, l, zeta, order, rcut, center, grid, **kwargs):

    #   if 'subset' in kwargs :
    #       subset = kwargs['subset']
    #       pts = grid.subset_coords(subset)
    #   else:
    #       subset = None
    #       pts = grid.coords


    #   r, theta, phi = np.hsplit(cart2spherical(pts-center),3)
    #   angular = real_spherical_harmonics(m,l,theta,phi)

    #   f = radial_slater_localized(n, zeta, order, rcut)
    #   radial = f(r)

    #   values = angular*radial

    #   return FieldOnGrid(values, np.float, subset=subset, n=n, m=m, l=l, zeta=zeta, center=center, order=order, rcut=rcut)

    @classmethod
    def spline_slater_orbital(cls, n, m, l, zeta, order, rcut, center, grid, **kwargs):
  
       if 'subset' in kwargs :
           subset = kwargs['subset']
           pts = grid.subset_coords(subset)-center
       else:
           subset = None
           pts = grid.coords-center

       r = np.linalg.norm(pts,axis=-1)

       f = radial_slater_localized(n, zeta, order, rcut, normal=True)

       #y = slater_spline(n, zeta, order, rcut)
       #f = localized_radial_spline(y, rcut)

       radial = f(r)
       angular = direct_real_spherical_harmonics(l,m,pts)

       values = angular*radial

       #for idx in range(len(pts)):
       #    print r[idx], radial[idx], values[idx]
       #    #print " {: 10.6f} {: 10.6f} {: 10.6f} {: 10.6f} {: 10.6f}".format(r[idx][0], theta[idx][0], phi[idx][0], radial[idx][0], values[idx][0])

       return FieldOnGrid(values.flatten(), np.float, subset=subset, n=n, m=m, l=l, zeta=zeta, center=center, order=order, rcut=rcut)

################################################################################

class BasisOnGrid(object):

    def __init__(self, table, graph, grid):


        self.table = table
        self.graph = graph
        self.basis = AtomicBasisSet(self.graph.nodes(data=False), self.graph.species, table)

        # Determine the largest orbital radius
        rcmax = 0.
        for inode in self.graph.nodes(data=False):
        
            istart = self.basis.first_basis_on_site(inode)
            istop = self.basis.last_basis_on_site(inode)+1
            for ibasis in range(istart, istop):
                at = self.basis.atom_of_basis(ibasis)
                iorb = self.basis.local_index_of_basis(ibasis)
                
                params = self.table[at]['params'][iorb]
                if params['rcut'] > rcmax:
                    rcmax = params['rcut']

        # List of atoms with non-zero orbital on the unit-cell grid
        lattice = self.graph.lattice
        ortho = np.array([v/np.linalg.norm(v) for v in lattice.inv_matrix.T])*rcmax
        delta = np.array([np.dot(ortho[i],lattice.inv_matrix)[i] for i in range(3)])
        nsc = np.ceil(delta).astype(np.int)
        vertices = super_centered(nsc)

        self.site_images = []
        for isite, ifrac in enumerate(self.graph.frac_coords):
            imidx = bounding_box_filter(ifrac+vertices, -delta, delta+1.) 
            #imidx = bounding_box(ifrac+vertices,-delta,1.+delta)
            self.site_images.append(vertices[imidx])
 
        # orbital centers
        self.oct = []
        
        # orbitals on grid
        self.oog = []

        # orbital number of images
        self.onim = np.zeros(self.basis.size).astype(int)

        # orbital pointers
        self.optr = np.zeros(self.basis.size).astype(int)

        # Compute orbitals on grid
        nsites = len(self.graph.frac_coords)
        iptr = 0
        for inode in range(nsites):
            istart = self.basis.first_basis_on_site(inode)
            istop = self.basis.last_basis_on_site(inode)+1
            for ibasis in range(istart, istop):

                coords = self.graph.cart_coords[inode]
                at = self.basis.atom_of_basis(ibasis)
                iorb = self.basis.local_index_of_basis(ibasis)
                orb = self.basis.orbital_of_basis(ibasis)
                params = self.table[at]['params'][iorb]
                rcut = params['rcut']
                zeta = params['zeta']
                n = params['n']
                order = params['order']
                l,m = NumericalAtom.orbital_angmom[orb[0]]
                data = {'coords':coords, 'atom':at, 'orb':orb, 'n':n, 'l':l, 'm':m, 'rcut':rcut, 'zeta':zeta, 'order':order }

                self.optr[ibasis] = iptr
                for v in self.site_images[inode]:
                    center = coords + lattice.get_cartesian_coords(v)
                    data['center'] = center
                    mask = grid.subset_from_sphere(center, rcut)  
                    if np.any(mask):
                        grid.create_subset(iptr,mask) 
                        self.oct.append(center)
                        self.oog.append(FieldOnGrid.spline_slater_orbital(n,m,l,zeta,order,\
                                                    rcut,center,grid,subset=iptr))
                        self.oog[-1].params = data
                        iptr += 1
                self.onim[ibasis] = iptr-self.optr[ibasis]

        # total number of orbital in extended cell
        self.norb = self.basis.size
        self.nsco = len(self.oog)

        # normalize orbitals
        self.normalize_orbitals(grid)

    def normalize_orbitals(self, grid):

        norms = np.zeros(self.norb).astype(float)
        for iorb in range(self.norb): 
            for iscorb in range(self.optr[iorb],self.optr[iorb]+self.onim[iorb]):
                norms[iorb] += grid.overlap_integral(self.oog[iscorb],self.oog[iscorb])    
        facs = np.sqrt(norms)         
        
        for iorb in range(self.norb):
            for iscorb in range(self.optr[iorb],self.optr[iorb]+self.onim[iorb]):
                self.oog[iscorb].values = self.oog[iscorb].values/facs[iorb]                 

    def compute_overlap_integrals(self, grid, keep=True):
        
        overlaps = []
        vecs = []

        for iorb in range(self.norb):
            overlaps.append([])
            vecs.append([])
            for jorb in range(self.norb):
                overlaps[iorb].append([])
                vecs[iorb].append([])

        for iorb in range(self.norb):
            for iscorb in range(self.optr[iorb],self.optr[iorb]+self.onim[iorb]):
                for jorb in range(self.norb):
                    for jscorb in range(self.optr[jorb],self.optr[jorb]+self.onim[jorb]):

                        v = self.oct[jscorb] - self.oct[iscorb]
                        s = grid.overlap_integral(self.oog[iscorb],self.oog[jscorb])
 
                        if np.linalg.norm(s) > 2.e-15:
                             overlaps[iorb][jorb].append(s)
                             vecs[iorb][jorb].append(v)

        if keep:
           self.S_vecs = vecs
           self.S_ints = overlaps

        return vecs, overlaps

    def compute_overlap_matrix(self, grid, kpt=[0.,0.,0.], fractional=True):

        if fractional:
           kpt = np.dot(kpt,grid.lattice.reciprocal_lattice.matrix)
        else:
           kpt = kpt

        if hasattr(self, 'S_vecs') and hasattr(self, 'S_ints'):
           vecs = self.S_vecs
           overlaps = self.S_ints
        else:
           vecs, overlaps = self.compute_overlap_integrals(grid)

        S = np.zeros((self.norb,self.norb)).astype(complex)
        for iorb in range(self.norb):
            for jorb in range(self.norb):
                for ijorb in range(len(overlaps[iorb][jorb])): 
                    S[iorb,jorb] += overlaps[iorb][jorb][ijorb]*np.exp(1j*np.dot(vecs[iorb][jorb][ijorb],kpt))
        return S

    def compute_proj_integrals(self, grid, field):
        
        projections = []
        vecs = []
 
        for iorb in range(self.norb):
            projections.append([])
            vecs.append([])
            for iscorb in range(self.optr[iorb],self.optr[iorb]+self.onim[iorb]): 
                p = grid.overlap_integral(self.oog[iscorb],field)
                v = self.oct[iscorb]

                vecs[iorb].append(vec)
                projections[iorb].append(p)

        return vecs, projections

    def compute_proj_matrix(self, grid, fields):

        P = np.zeros((len(fields),self.norb)).astype(complex)

        for iw in range(len(fields)):
            vecs, projections = self.compute_proj_integrals(grid, fields[iw])
            for iorb in range(self.norb):
                P[iw, iorb] = np.sum(np.array(projections[iorb]))

        return P

    def compute_proj_eigh(self, grid, fields, eners, kpt=[0.,0.,0.], fractional=True,\
                          eigv=False, kernel=False):

        S = self.compute_overlap_matrix(grid, kpt=kpt, fractional=fractional)
        P = self.compute_proj_matrix(grid, fields)
        H = np.zeros((self.norb,self.norb)).astype(complex)

        nw = len(fields)
        for iw in range(nw):
            H += eners[iw]*np.outer(P[iw],np.conj(P[iw]))

        if kernel:
            Sinv = np.linalg.inv(S)
            K = np.dot(np.conj(P), np.dot(Sinv,np.transpose(P)))
            return [H,S,K]

        else:
            return [H,S]
