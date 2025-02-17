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


def corner_vertices(frac):
    return np.unique(combine([[np.floor(ifrac), np.ceil(ifrac)] for ifrac in frac]),axis=0)

def bounding_box(p, pmin, pmax):
    """ Filter points with respect to a bounding box

    Parameters:
    ----------
    p: (m,3) array
         Input array of points coordinates

    pmin: (3) array
         Lower limit of the bounding box

    pmax: (3) array
         Higher limit of the bounding box

    Returns:
    -------
    bb: (m) boolean array

    """

    bound_x = np.logical_and(p[:,0] > pmin[0], p[:,0] < pmax[0])
    bound_y = np.logical_and(p[:,1] > pmin[1], p[:,1] < pmax[1])
    bound_z = np.logical_and(p[:,2] > pmin[2], p[:,2] < pmax[2])

    bb = np.logical_and(bound_x, bound_y, bound_z)

    return bb 


def closest_nodes(lattice, points):
    
    pts = np.array(points)
    d = np.zeros(pts.shape[0])
    l = np.zeros(pts.shape,dtype=np.int)
    
    fracs = lattice.get_fractional_coords(pts)
    for i in range(fracs.shape[0]):
        vertices = corner_vertices(fracs[i])
        dist = np.linalg.norm(lattice.get_cartesian_coords(vertices-fracs[i]), axis=-1)
        idx = dist.argmin()
        
        d[i] = dist[idx]
        l[i] = vertices[idx]
        
    return d, l


def super_vertices_centered(n):

    nx, ny, nz = n
    return  np.array([[ix, iy, iz] for ix in range(nx+1)+range(-nx,0) \
                                   for iy in range(ny+1)+range(-ny,0) \
                                   for iz in range(nz+1)+range(-nz,0)],dtype=int)

def cart2spherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    ptsnew[:,0] = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    ptsnew[:,1] = np.arccos(xyz[:,2]/ptsnew[:,0])
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew


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


def real_spherical_harmonics2(m,l,theta,phi):

    ma = np.absolute(m)
    if m == 0:
        return scipy.special.sph_harm(m,n,theta,phi)
    else:
        spha = scipy.special.sph_harm(m,n,theta,phi)
        sphb = scipy.special.sph_harm(-m,n,theta,phi)
        fac = 1./np.sqrt(2.)
        if m < 0:
            return fac*1j*(spha-(-1)**m*sphb)
        elif m > 0:
            return fac*(sphb+(-1)**m*spha)

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

def slater_spline(n, zeta, nnodes, rcut):

    delta = rcut/(nnodes)
    x = np.arange(0.,rcut, delta)
    return radial_slater(n, zeta, x)

def localized_radial_spline(ynodes, rcut, normal=True):

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


class PointCollection(object):
    """ 
    Basic collection of points. This is essentially an array of points
    to which weigths can be associated. The class provide methods to 
    define subsets of points and to perform basic set operations.      
    """

    def __init__(self, coords, weigths=[], normalize=False):
 
        self.size = coords.shape[0]
        self._coords = np.array(coords)

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
        self.subsets[None] = np.array(range(self.size))

    @property
    def coords(self):
        """
        The coordinates of the points in the collection
        """
        return self._coords

    #@coords.setter
    #def coords(self,coords):
    #    """
    #    The coordinates of the points in the collection
    #    """
    #    self._coords = coords
    #    self._frac_coords = self.lattice.get_fractional_coords(coords)

    def create_subset(self, label, list_of_index):
        """
        Defines a subset of points from a list of indices
        """
        #print ' create subset :', label, len(np.unique(list_of_index))
        self.subsets.update({label:np.unique(list_of_index)})

    def create_subset_from_sphere(self, label, center, radius):
        """
        Defines a subset of points corresponding to a given spherical
        region
        """
        subset = []
        for idx, coord in enumerate(self.coords):
            if np.linalg.norm(center-coord) < radius:
                subset.append(idx)
        self.create_subset(label,subset)


    def subset_from_sphere(self, center, radius):
 
        return np.argwhere(np.linalg.norm(self.coords-center,axis=-1) < radius)

    def delete_subset(self, label):
        """
        Delete a specified subset. 
        """
        self.subsets.pop(label)

    def subsets_intersect(self, A, B, new=None):
        """
        Returns the indices of the intersection between
        subsets labelled A and B. 
        """
        newlist = np.intersect1d(self.subsets[A],self.subsets[B],assume_unique=True)
        #print 'intersect size : ',len(newlist), '/', self.size
        if new == None:
            return newlist
        else:
            self.create_subset(new, newlist)

    def subsets_union(self, A, B, new=None):
        """
        Returns the indices of the union between
        subsets labelled A and B. 
        """
        newlist =  np.union1d(self.subsets[A],self.subsets[B])
        if label == None:
            return newlist
        else:
            self.create_subset(new, newlist)

    def subsets_map(self, A, B):
        """
        Returns two boolean arrays mapA and mapB, the same length as subsets A and B respectively. 
        MapA is True where element of A is in B, and False otherwise. 
        MapB is True where element of B is in A, and False otherwise. 
        """
        isec = self.subsets_intersect(A,B)
        return isec, np.in1d(self.subsets[A],isec), np.in1d(self.subsets[B],isec) 

    def subset_coords(self, A):
        """
        Returns the coordinates of points in a given subset
        """
        return self.coords[self.subsets[A]]


    def create_field(self, dim=1, data=None, subset=None, **kwargs):

        if subset != None:
           size = self.subset_size(subset)
        else:
           size = self.size
 
        if data!= None:
            if size != len(data):
                raise IndexError('Length of data array is not compatible')
            idata = data
        else:
            idata = np.zeros((size,dim))

        return FieldOnGrid(idata, subset, **kwargs)

    def overlap_integral(self, fa, fb):

        isec, ma, mb = self.subsets_map(fa.subset,fb.subset)
        #print 'overlap int ', len(isec)
        if hasattr(self,'weights'):
            return np.sum(np.conj(fa.values[ma])*fb.values[mb]*self.weigths[isec])
        else:
            return np.sum(np.conj(fa.values[ma])*fb.values[mb])


################################################################################	

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
                self.frac_coords = self.lattice.get_fractional_coords(self.coords)
        else:
            super(PeriodicPointCollection, self).__init__( \
                  self.lattice.get_cartesian_coords(coords), \
                  weigths=weigths)
            self._frac_coords = coords


    #@property
    #def coords(self):
    #    """
    #    The coordinates of the points in the collection
    #    """
    #    return self._coords

    #@coords.setter
    #def coords(self,coords):
    #    """
    #    The coordinates of the points in the collection
    #    """
    #    self._coords = coords
    #    self._frac_coords = self.lattice.get_fractional_coords(coords)

    @property
    def frac_coords(self):
        """
        The fractional coordinates of the points in the collection
        """
        return self._frac_coords

    #@frac_coords.setter
    #def frac_coords(self, frac_coords):
    #    """
    #    The fractional coordinates of the points in the collection
    #    """
    #    self._frac_coords = frac_coords
    #    self._coords = self.lattice.get_cartesian_coords(frac_coords)
        

    #def subset_coords(self, label, cartesian=True):
    #    """
    #    Returns the coordinates of points in a given subset
    #    """
    #    if cartesian:
    #        return self._coords[self.subsets[label]]
    #    else:
    #        return self._frac_coords[self.subsets[label]]

    def subset_frac_coords(self, A):
        """
        Returns the coordinates of points in a given subset
        """
        #print A
        #print self.subsets[A]
        return self._frac_coords[self.subsets[A]]


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
            
################################################################################	

class FieldOnGrid(object):

    def __init__(self, values, subset=None, **kwargs):
  
       self.subset = subset
       self.values = np.array(values)
       self.params = kwargs
  
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

       if 'scalings' in kwargs:
           periods = grid.lattice.get_cartesian_coords(super_vertices_centered(kwargs['scalings']))
           for period in periods[1:]:
               #print period
               r, theta, phi = np.hsplit(cart2spherical(pts-center-period),3)
               values += real_spherical_harmonics(m,l,theta,phi)*radial_slater(n,zeta,r) 

       return FieldOnGrid(values, subset=subset, n=n, m=m, l=l, zeta=zeta, center=center)
 
    @classmethod
    def spline_slater_orbital(cls, n, m, l, zeta, order, rcut, center, grid, **kwargs):
  
       if 'subset' in kwargs :
           subset = kwargs['subset']
           pts = grid.subset_coords(subset)-center
       else:
           subset = None
           pts = grid.coords-center

       r = np.linalg.norm(pts,axis=-1)
       y = slater_spline(n, zeta, order, rcut)
       f = localized_radial_spline(y, rcut)

       radial = f(r)
       angular = direct_real_spherical_harmonics(l,m,pts)

       values = angular*radial

       #for idx in range(len(pts)):
       #    print r[idx], radial[idx], values[idx]
       #    #print " {: 10.6f} {: 10.6f} {: 10.6f} {: 10.6f} {: 10.6f}".format(r[idx][0], theta[idx][0], phi[idx][0], radial[idx][0], values[idx][0])

       return FieldOnGrid(values, subset=subset, n=n, m=m, l=l, zeta=zeta, center=center, order=order, rcut=rcut)

################################################################################	

class BasisOnGrid(object):

    def __init__(self, table, graph, grid):


        self.table = table
        self.graph = graph
        self.basis = AtomicBasisSet(self.graph.nodes, self.graph.species, table)

        #print self.graph.nodes        

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
        vertices = super_vertices_centered(nsc)

        self.site_images = []
        for isite, ifrac in enumerate(self.graph.frac_coords):
            p = ifrac + vertices
            v = deepcopy(vertices)
            for ix in range(3):
                vtmp = v[np.logical_and(p[:,ix]>-delta[ix], p[:,ix]<1.+delta[ix])]
                ptmp = p[np.logical_and(p[:,ix]>-delta[ix], p[:,ix]<1.+delta[ix])]
                p = ptmp
                v = vtmp
            self.site_images.append(v)
        
        # Compute orbitals on grid
        tmp_oog = [] 
        nsites = len(self.graph.frac_coords)
        for inode in range(nsites):
            istart = self.basis.first_basis_on_site(inode)
            istop = self.basis.last_basis_on_site(inode)+1
            for ibasis in range(istart, istop):
                #print '----'
                #print ibasis
                tmp_oog.append([])

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
                #print at, iorb, l, m, params
                #print '....'

                for v in self.site_images[inode]:
                    #print v
                    center = coords + np.dot(np.array(v),lattice.matrix)
                    tmp_oog[ibasis].append(FieldOnGrid.spline_slater_orbital(n,m,l,zeta,order,rcut,center,grid))

        # orbitals images
        self.oim = []
        
        # orbitals on grid
        self.oog = []

        # orbital number of images
        self.onim = []


        for inode in self.graph.nodes(data=False):
            istart = self.basis.first_basis_on_site(inode)
            istop = self.basis.last_basis_on_site(inode)+1
            for ibasis in range(istart, istop):
                mask = [np.amax(fog.values)!=0. or np.amin(fog.values)!=0. for fog in tmp_oog[ibasis]]
                self.onim.append(np.sum(mask))
                for idx, flag in enumerate(mask):
                    if flag:
                        self.oog.append(tmp_oog[ibasis][idx]) 
                        self.oim.append(tuple(self.site_images[inode][idx]))

        # orbital pointers
        self.optr = np.zeros(self.basis.size).astype(int)
        self.optr[1:] = np.cumsum(self.onim)[:-1]

        # total number of orbital in extended cell
        self.norb = self.basis.size
        self.nsco = len(self.oog)

    def compute_overlap_integrals(self, grid):
        
        #overlaps = np.zeros((self.nsco,self.nsco)).astype(float)
        #vecs = np.zeros(((self.nsco,self.nsco,3))).astype(int)
       
        overlaps = []
   
        vecs = [[[]*self.norb ]*self.norb]
        for iorb in range(self.norb):
            overlaps.append([])
            vecs.append([])
            for iscorb in range(self.optr[iorb],self.optr[iorb]+self.onim[iorb]):
                for jorb in range(self.norb):
                    overlaps[iorb].append([])
                    vecs[iorb].append([])
                    for jscorb in range(self.optr[jorb],self.optr[jorb]+self.onim[jorb]):

                        #overlaps[iscorb, jscorb] = self.oog[iscorb].overlap_integral(self.oog[jscorb])
                        #vecs[iscorb, jscorb] = self.oim[jscorb]-self.oim[iscorb]
                        
                        v = tuple(np.array(self.oim[jscorb]).astype(int) - np.array(self.oim[iscorb]).astype(int))
                        s = grid.overlap_integral(self.oog[iscorb],self.oog[jscorb])
                        try:
                            idx = vecs[iorb][jorb].index(v)
                            overlaps[iorb][jorb][idx] += s
                        except:
                            vecs[iorb][jorb].append(v)
                            overlaps[iorb][jorb].append(s)
                        
        return vecs, overlaps

################################################################################	

class SubBasisOnGrid(object):

    def __init__(self, table, graph, grid):


        self.table = table
        self.graph = graph
        self.basis = AtomicBasisSet(self.graph.nodes, self.graph.species, table)

        #print self.graph.nodes        

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
        vertices = super_vertices_centered(nsc)

        self.site_images = []
        for isite, ifrac in enumerate(self.graph.frac_coords):
            p = ifrac + vertices
            v = deepcopy(vertices)
            for ix in range(3):
                vtmp = v[np.logical_and(p[:,ix]>-delta[ix], p[:,ix]<1.+delta[ix])]
                ptmp = p[np.logical_and(p[:,ix]>-delta[ix], p[:,ix]<1.+delta[ix])]
                p = ptmp
                v = vtmp
            self.site_images.append(v)
        
        # orbitals images
        self.oim = []
        
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
                #print '----'
                #print ibasis

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
                #print at, iorb, l, m, params
                #print '....'

                self.optr[ibasis] = iptr
                for v in self.site_images[inode]:
                    center = coords + np.dot(np.array(v),lattice.matrix)
                    ptlist = grid.subset_from_sphere(center, rcut)  
                   # print " {: 2d} {: 2d} {: 2d}".format(*v), " {: 4.2f} ".format(rcut), " {: 8d} ".format(len(ptlist))
                    if len(ptlist) > 0:
                        grid.create_subset(iptr,ptlist) 
                        self.oim.append(tuple(v))
                        self.oog.append(FieldOnGrid.spline_slater_orbital(n,m,l,zeta,order,\
                                                    rcut,center,grid,subset=iptr))
                        iptr += 1
                self.onim[ibasis] = iptr-self.optr[ibasis]

        # total number of orbital in extended cell
        self.norb = self.basis.size
        self.nsco = len(self.oog)


    def compute_overlap_integrals(self, grid):
        
        #overlaps = np.zeros((self.nsco,self.nsco)).astype(float)
        #vecs = np.zeros(((self.nsco,self.nsco,3))).astype(int)
       
        overlaps = []
        vecs = []
        for iorb in range(self.norb):
            overlaps.append([])
            vecs.append([])
            for iscorb in range(self.optr[iorb],self.optr[iorb]+self.onim[iorb]):
                for jorb in range(self.norb):
                    overlaps[iorb].append([])
                    vecs[iorb].append([])
                    for jscorb in range(self.optr[jorb],self.optr[jorb]+self.onim[jorb]):

                        #overlaps[iscorb, jscorb] = self.oog[iscorb].overlap_integral(self.oog[jscorb])
                        #vecs[iscorb, jscorb] = self.oim[jscorb]-self.oim[iscorb]
                        
                        v = tuple(np.array(self.oim[jscorb]).astype(int) - np.array(self.oim[iscorb]).astype(int))
                        s = grid.overlap_integral(self.oog[iscorb],self.oog[jscorb])
                        #print " {: 4d} {: 4d} {: 4d} {: 4d}".format(iorb, iscorb, jorb, jscorb), \
                        #      " - "," {: 2d} {: 2d} {: 2d}".format(*v),  " {: 8.4f}".format(s)
                        try:
                            idx = vecs[iorb][jorb].index(v)
                            overlaps[iorb][jorb][idx] += s
                        except:
                            vecs[iorb][jorb].append(v)
                            overlaps[iorb][jorb].append(s)
                       
        ## Normalization
        #norms = np.zeros(self.norb)
        #for iorb in range(self.norb):
        #    norms[iorb] = 1./np.sum(np.array(overlaps[iorb][iorb]))

        return vecs, overlaps

    def compute_basis_normalization_factors(self, overlaps, grid):
        
        norms = np.zeros(self.norb)
        for iorb in range(self.norb):
            norms[iorb] = 1./np.sum(np.array(overlaps[iorb][iorb]))
        return norms

    def compute_overlap_matrix_gamma(self, grid):

        vecs, overlaps = self.compute_overlap_integrals(grid)

        S = np.zeros((self.norb,self.norb))
        for iorb in range(self.norb):
            iat = self.basis.site_of_basis(iorb)
            for jorb in range(self.norb): 
                jat = self.basis.site_of_basis(jorb)
                S[iorb,jorb] = np.sum(np.array(overlaps[iorb][jorb]))

        norms = np.array([np.sqrt(S[iorb,iorb]) for iorb in range(self.norb)])

        return S/np.outer(norms,norms)

    def compute_projection_integrals(self, grid, field):
        
        projections = []
        vecs = []
 
        for iorb in range(self.norb):
            projections.append([])
            vecs.append([])
            for iscorb in range(self.optr[iorb],self.optr[iorb]+self.onim[iorb]): 
                p = grid.overlap_integral(self.oog[iscorb],field)
                v = tuple(np.array(self.oim[iscorb]).astype(int))

                vecs[iorb].append(v)
                projections[iorb].append(p)

        return vecs, projections


################################################################################	

class LocalizedBasisOnGrid(object):

    def __init__(self, table, graph, grid):


        self.table = table
        self.graph = graph
        self.basis = AtomicBasisSet(self.graph.nodes(data=False), self.graph.species, table)

        #print self.graph.nodes        

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
        vertices = super_vertices_centered(nsc)

        self.site_images = []
        for isite, ifrac in enumerate(self.graph.frac_coords):
            imidx = bounding_box(ifrac+vertices,-delta,1.+delta)
            self.site_images.append(vertices[imidx])
 
        # orbitals images
        self.oim = []
        
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
                #print '----'
                #print ibasis

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
                #print at, iorb, l, m, params

                self.optr[ibasis] = iptr
                for v in self.site_images[inode]:
                    center = coords + np.dot(np.array(v),lattice.matrix)
                    ptlist = grid.subset_from_sphere(center, rcut)  
                    if len(ptlist) > 0:
                        grid.create_subset(iptr,ptlist) 
                        self.oim.append(tuple(v))
                        self.oog.append(FieldOnGrid.spline_slater_orbital(n,m,l,zeta,order,\
                                                    rcut,center,grid,subset=iptr))
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

    def compute_overlap_integrals(self, grid):
        
        overlaps = []
        vecs = []
 
        for iorb in range(self.norb):
            overlaps.append([])
            vecs.append([])
            for iscorb in range(self.optr[iorb],self.optr[iorb]+self.onim[iorb]):
                for jorb in range(self.norb):
                    overlaps[iorb].append([])
                    vecs[iorb].append([])
                    for jscorb in range(self.optr[jorb],self.optr[jorb]+self.onim[jorb]):

                        v = tuple(np.array(self.oim[jscorb]).astype(int) - np.array(self.oim[iscorb]).astype(int))
                        s = grid.overlap_integral(self.oog[iscorb],self.oog[jscorb])
                        vec = grid.lattice.get_cartesian_coords(v)
 
                        try:
                            idx = vecs[iorb][jorb].index(v)
                            overlaps[iorb][jorb][idx] += s
                        except:
                            vecs[iorb][jorb].append(vec)
                            overlaps[iorb][jorb].append(s)
                      
        return vecs, overlaps

    def compute_projection_integrals(self, grid, field):
        
        projections = []
        vecs = []
 
        for iorb in range(self.norb):
            projections.append([])
            vecs.append([])
            for iscorb in range(self.optr[iorb],self.optr[iorb]+self.onim[iorb]): 
                p = grid.overlap_integral(self.oog[iscorb],field)
                v = tuple(np.array(self.oim[iscorb]).astype(int))
                vec = grid.lattice.get_cartesian_coords(v)

                vecs[iorb].append(vec)
                projections[iorb].append(p)

        return vecs, projections

    def compute_overlap_matrix_gamma(self, grid):

        vecs, overlaps = self.compute_overlap_integrals(grid)

        S = np.zeros((self.norb,self.norb))
        for iorb in range(self.norb):
            iat = self.basis.site_of_basis(iorb)
            for jorb in range(self.norb): 
                jat = self.basis.site_of_basis(jorb)
                S[iorb,jorb] = np.sum(np.array(overlaps[iorb][jorb]))

        return S

    def compute_overlap_matrix(self, grid, kpt=[0.,0.,0.], fractional=True):

        if fractional:
           kpt = np.dot(kpt,grid.lattice.reciprocal_lattice.matrix)
        else:
           kpt = kpt

        vecs, overlaps = self.compute_overlap_integrals(grid)

        S = np.zeros((self.norb,self.norb)).astype(complex)
        for iorb in range(self.norb):
            for jorb in range(self.norb): 
                S[iorb,jorb] = np.sum(np.array(overlaps[iorb][jorb]*np.exp(1j*np.dot(vecs[iorb][jorb],kpt))))

        return S

    def compute_projection_vector(self, grid, field, kpt=[0.,0.,0.], fractional=True):

        S = self.compute_overlap_matrix(grid, kpt=kpt, fractional=fractional)
        Sinv = np.linalg.inv(S) 

        vecs, projections = self.compute_projection_integrals(grid, field)

        P = np.zeros((self.norb)).astype(complex)
        for iorb in range(self.norb):
            P[iorb] = np.sum(np.array(projections[iorb]*np.exp(1j*np.dot(vecs[iorb],kpt))))


        return np.dot(P,Sinv)


    def compute_projection_matrix(self, grid, fields, kpt=[0.,0.,0.], fractional=True, **kwargs):                    

        if 'S' in kwargs:
            S = kwargs['S']
        else:
            S = self.compute_overlap_matrix(grid, kpt=kpt, fractional=fractional)
        if 'Sinv' in kwargs:
            Sinv = kwargs['Sinv']
        else:
            Sinv = np.linalg.inv(S) 

        P = np.zeros((len(fields),self.norb)).astype(complex)

        for iw in range(len(fields)):
            vecs, projections = self.compute_projection_integrals(grid, fields[iw])
            for iorb in range(self.norb):
                P[iw, iorb] = np.sum(np.array(projections[iorb]*np.exp(1j*np.dot(vecs[iorb],kpt))))

        return np.dot(P,Sinv)

    def compute_projected_hamiltonian(self, grid, fields, eners, kpt=[0.,0.,0.], fractional=True,\
                                      eigv=False, kernel=False):

        import scipy.linalg as linalg

        nw = len(fields)
        S = self.compute_overlap_matrix(grid, kpt=kpt, fractional=fractional)
        Sinv = np.linalg.inv(S)

        P = self.compute_projection_matrix(grid, fields, kpt=kpt, fractional=True, S=S, Sinv=Sinv)
        M = np.zeros((self.norb,self.norb)).astype(complex)

        for iw in range(nw):
            M += eners[iw]*np.outer(P[iw],np.conj(P[iw]))

        H = np.matmul(S,np.matmul(M,S))
        out = [H,S]

        if kernel:
            K = np.dot(np.conj(P),np.dot(S,np.transpose(P)))
            out.append(K)

        if eigv:
            ev = linalg.eigh(H, b=S, lower=False, turbo=True, eigvals_only=True) 
            out.append(ev)

        return out

    def compute_projected_eigenvalues(self, grid, fields, eners, kpt=[0.,0.,0.], fractional=True):

        import scipy.linalg as linalg

        nw = len(fields)
        S = self.compute_overlap_matrix(grid, kpt=kpt, fractional=fractional)
        Sinv = np.linalg.inv(S)

        P = self.compute_projection_matrix(grid, fields, kpt=kpt, fractional=True, S=S, Sinv=Sinv)
        M = np.zeros((self.norb,self.norb)).astype(complex)

        for iw in range(nw):
            M += eners[iw]*np.outer(P[iw],np.conj(P[iw]))

        H = np.matmul(S,np.matmul(M,S))
        ev = linalg.eigh(M, b=S, lower=False, turbo=True, eigvals_only=True) 
        
        return H, S, ev

