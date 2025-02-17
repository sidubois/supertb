""" Python Classes and functions
Simple Pyhton classes and functions providing some geometrical functionalities
to handle atomic systems
""" 

import numpy as np
import itertools
import pymatgen as pm
from math import sin, cos, asin, acos, pi, radians
from scipy import spatial
from structure import Lattice
import sys

################################################################################	

class PointCollection(object):
    """ 
    Basic collection of points. This is essentially an array of points
    to which weigths can be associated. The class provide methods to 
    define subsets of points and to perform basic set operations.      
    """

    def __init__(self, coords=None, weigths=None):
 
        if coords != None:
            self.coords = np.array(coords)
            self.size = coords.shape[0]
        else:
            self.size

        if weigths != None:
            if weigths.shape[0] == self.size:
                self.weigths = np.array(weigths)
            else:
                raise IndexError('shapes of arguments are not compatible')

        self.subsets = {} 

    def nearest_neighbors(self, center, nn=1):
        """
        Return the distance and indices of nn points nearest
        to the given reference
        """
    
    def neighbors(self, center, radius):
        """
        Return the indices of the points located within a given radius
        around given reference
        """

    def create_subset(self, label, list_of_index):
        """
        Defines a subset of points from a list of indices
        """
        self.subsets.update({label:np.unique(list_of_index.sort())})

    def create_subset_from_sphere(self, label, center, radius):
        """
        Defines a subset of points corresponding to a given spherical
        region
        """

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
        newlist = np.interesct1d(self.subsets[A],self.subsets[B],assume_unique=True)
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
        if label == None:
            return self.coords
        else:
            return self.coords[self.subsets[A]]

################################################################################	

    def create_field(self, label, dim=1, data=None, subset=None, **kwargs):
        
        if subset != None:
           size = len(self.subsets[subset])
        else:
           size = self.size

        if data!= None:
            if size != len(data):
                raise IndexError('Length of data array is not compatible')
            idata = data
        else:
            idata = np.zeros((size,dim))

        return FieldOnGrid(label, idata, subset, **kwargs)

    def fields_overlap(self,fa,fb):

        isec, ma, mb = self.subsets_map(fa.label,fb.label)
        if self.hasattr('weights'):
            return np.sum(np.conj(fa.data[ma])*fb.data[mb]*self.weigths[isec])
        else:
            return np.sum(np.conj(fa.data[ma])*fb.data[mb])

################################################################################	

class LatticePointCollection(PointCollection):
    """ 
    Collection of points defined with respect to a given lattice. This is basically
    a collection of points whose both cartesian and fractional coordinates are stored.
    """

    def __init__(self, lattice, coords=None, weigths=None, cartesian=True):

        if isinstance(lattice,Lattice):
            self.lattice = lattice
        else:
            self.lattice = Lattice(lattice)

        if cartesian:
            super(LatticePointCollection, self).__init__( \
                  coords=coords, weigths=weigths)
            if coords != None:
                self._frac_coords = self.lattice.get_fractional_coords(coords)
        else:
            super(LatticePointCollection, self).__init__( \
                  coords=self.lattice.get_cartesian_coords(coords), \
                  weigths=weigths)
            if coords != None:
                self._frac_coords = coords

    @property
    def coords(self):
        """
        The coordinates of the points in the collection
        """
        return self._coords

    @coords.setter
    def coords(self,coords):
        """
        The coordinates of the points in the collection
        """
        self._coords = value
        self._frac_coords = self.lattice.get_fractional_coords(coords)

    @property
    def frac_coords(self):
        """
        The fractional coordinates of the points in the collection
        """
        return self._frac_coords

    @frac_coords.setter
    def frac_coords(self, frac_coords):
        """
        The fractional coordinates of the points in the collection
        """
        self._frac_coords = frac_coords
        self._coords = self.lattice.get_cartesian_coords(frac_coords)
        

    def subset_coords(self, label, cartesian=True):
        """
        Returns the coordinates of points in a given subset
        """
        if cartesian:
            return self._coords[self.subsets[label]]
        else:
            return self._frac_coords[self.subsets[label]]



################################################################################	

class PeriodicPointCollection(LatticePointCollection):
    """ 
    Periodic collection of points. This is essentially a collection of points 
    periodically repeated in space. 
    """

    def check_kdtree(self):
        """
        Initialise kdtree if not done yet
        """
        if not hasattr(self,'tree'):
           periodic_distance = lambda x, y: self.lattice.periodic_distance(y-x)
           self.tree = CoverTree(self.coords,periodic_distance)
   
    def update_kdtree(self):
        """
        Initialise kdtree 
        """
        periodic_distance = lambda x, y: self.lattice.periodic_distance(y-x)
        self.tree = CoverTree(self.coords,periodic_distance)


################################################################################	

class RegularGrid(PeriodicPointCollection):

    def __init__(self, lattice, shifts=None, ndiv=None, maxl=None):

        if maxl != None:
            na = int(math.ceil(self.lattice.a/maxl[0]))
            nb = int(math.ceil(self.lattice.b/maxl[1]))
            nc = int(math.ceil(self.lattice.c/maxl[2]))

        elif ndiv != None:
            na = int(max(ndiv[0],0.))
            nb = int(max(ndiv[1],0.))
            nc = int(max(ndiv[2],0.))

        pa = np.arange(0.,1.,1./na) 
        pb = np.arange(0.,1.,1./nb) 
        pc = np.arange(0.,1.,1./nc) 
        
        prim_coords = np.vstack(np.meshgrid(pa,pb,pc,indexing='ij')).reshape(3,-1).T
        
        if shifts != None: 
            ncoords = len(prim_coords) 
            shifted_coords = np.empty((ncoords*len(shifts),3))
            for ishift, shift in enumerate(shifts):
                shifted_coords[ishift*ncoords:(ishift+1)*ncoords] = \
                     prim_coords + shift
        else:
            shifted_coords = prim_coords
        
        super(RegularGrid, self).__init__(lattice, coords=shifted_coords, \
              cartesian = False)


################################################################################	

class IterMesh(PeriodicPointCollection):
    """ 
    Meshes are collections of points organised as regular meshes.
    """

    def __init__(self, voxel, itervec, weigths=None):
        
        frac_coordinates = []
        for x in itertools.product(*[itervec[-i] for i in range(1,len(itervec)+1)]):
           frac_coordinates.append([x[-i] for i in range(1,len(x)+1)])

        super(IterMesh,self).__init__(voxel, frac_coordinates, 
                                      weigths=weights, cartesian=False)


################################################################################	

class FieldOnGrid(object):

  def __init__(self, label, data, subset, **kwargs):


     self.label = label
     self.subset = subset
     self.data = np.array(data)
     self.dim = data.shape[-1]
     self.size = len(data)
     self.info = kwargs

  def create_slater_orbital(self,orb,center,grid):

     if hasattr(self,ptix):
        pts = grid.pts[self.ptix] - center
     else:
        pts = grid.pts[grid.reg[self.region]] - center

     r = np.linalg(pts,axis=-1)
     spham = spherical_harmonic(pts,r,orb['l'],orb['m'])
     rsto = radial_slater(r,orb['n'],orb['z'])
     self.data = spham*rsto     

  def create_slater_orbital(self,orb,center,grid):

     if hasattr(self,ptix):
        pts = grid.pts[self.ptix] - center
     else:
        pts = grid.pts[grid.reg[self.region]] - center

     r = np.linalg(pts,axis=-1)
     spham = spherical_harmonic(pts,r,orb['l'],orb['m'])
     rsto = radial_slater(r,orb['n'],orb['z'])
     self.data = spham*rsto     

################################################################################	

