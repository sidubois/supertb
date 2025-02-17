"""
This module defines basic classes and functions to deal with colinear 
and non-colinear spin on the same footing.... 
"""

import numpy as np
from copy import deepcopy

def mpauli(kind):

    if kind == 'x' or kind == 1:
        return np.array([[0.,1.+0.j],[1.+0.j,0.]])
    elif kind == 'y' or kind == 2:
        return np.array([[0.,-1j],[1j,0.]])
    elif kind == 'z' or kind == 3:
        return np.array([[1.,0.],[0.,-1.]])
    elif kind == 'i' or kind == 0:
        return np.array([[1.+0.j,0.],[0.,1.+0.j]])

class Spinor(int):

    def __new__(cls, *args, **kwargs):

        if 'nspin' in kwargs:
            nspin = kwargs['nspin']
        elif len(args) > 0:
            nspin = args[0]
        else:
            nspin = 1

        if 'spinorb' in kwargs:
            spinorb = kwargs['spinorb']
        else:
            spinorb = False

        if 'spinpol' in kwargs:
            spinpol = kwargs['spinpol']
        else:
            spinpol = False

        if 'dim' in kwargs:
            nspin = np.prod(kwargs['dim'])

        if spinorb or nspin == 4:
            spinorb = True
            spinpol = True
            nspin = 1
            dim = [2,2]
            identity = np.reshape([[1.+0.j,0.+0.j],[0.+0.j,1.+0.j]],dim)
            nbasis = 2
            deg = 1

        elif spinpol or nspin == 2:
            spinorb = False
            spinpol = True
            nspin = 2 
            dim = [2,1]
            identity = np.reshape([1.+0.j,1.+0.j],dim) 
            nbasis = 1
            deg = 1

        elif nspin == 1:
            spinorb = False
            spinpol = False
            nspin = 1
            dim = [1,1]
            identity = np.reshape([1.+0.j],dim)                                         
            nbasis = 1
            deg = 2

        spinor = int.__new__(cls, nspin)
        spinor.nspin = nspin
        spinor.dim = dim
        spinor.identity = identity
        spinor.nbasis = nbasis
        spinor.deg = deg
        spinor.spinorb = spinorb
        spinor.spinpol = spinpol

        return spinor

    def scalar_operator(self, Y):
        """
        Transform a general matrix Y(*,*,P,Q) where (P,Q) = spinor.dim 
        into a matrix S(*,*,R) where R = spinor.nspin.
        """

        row_size = Y.shape[0]
        col_size = Y.shape[1]

        S = np.zeros((row_size*self.nbasis, col_size*self.nbasis,self.nspin),dtype=complex)
        if self.dim[1] == 2:
            S[0:row_size,0:col_size,0] = Y[:,:,0,0]
            S[0:row_size,col_size:2*col_size,0] = Y[:,:,0,1]
            S[row_size:2*row_size,0:col_size,0] = Y[:,:,1,0]
            S[row_size:2*row_size,col_size:2*col_size,0] = Y[:,:,1,1]
        elif self.dim[0] == 2:
            S[:,:,:] = Y[:,:,:,0]
        else:
            S[:,:,:] = Y[:,:,:,0]

        return S

    def spinor_operator(self, Y):
        """
        Transform a general matrix Y(*,*,R) where R = spinor.nspin into a matrix
        S(*,*,P,Q) where (P,Q) = spinor.dim.
        """

        nr = Y.shape[0]/self.nbasis
        nc = Y.shape[1]/self.nbasis
        S = np.zeros((nr,nc,self.dim[0],self.dim[1]),dtype=complex)
        
        if self.dim[1] == 2:
            S[:,:,0,0] = Y[0:nr,0:nc,0]
            S[:,:,0,1] = Y[0:nr,nc:2*nc,0]
            S[:,:,1,0] = Y[nr:2*nr,0:nc,0]
            S[:,:,1,1] = Y[nr:2*nr,nc:2*nc,0]

        else :
            S[:,:,:,0] = Y

        return S

    def spinor_product(self, A, B):
    
        S = np.zeros((A.shape[0],B.shape[1],self.dim[0],self.dim[1]),dtype=complex)
        if self.dim[1] == 2:
            S[:,:,0,0] = np.dot(A[:,:,0,0],B[:,:,0,0]) +\
                         np.dot(A[:,:,0,1],B[:,:,1,0])
            S[:,:,0,1] = np.dot(A[:,:,0,0],B[:,:,0,1]) +\
                         np.dot(A[:,:,0,1],B[:,:,1,1])
            S[:,:,1,0] = np.dot(A[:,:,1,0],B[:,:,0,0]) +\
                         np.dot(A[:,:,1,1],B[:,:,1,0])
            S[:,:,1,1] = np.dot(A[:,:,1,0],B[:,:,0,1]) +\
                         np.dot(A[:,:,1,1],B[:,:,1,1])

        elif self.dim[0] == 2:
            S[:,:,0,0] = np.dot(A[:,:,0,0],B[:,:,0,0])  
            S[:,:,1,0] = np.dot(A[:,:,1,0],B[:,:,1,0])  

        else:
            S[:,:,0,0] = np.dot(A[:,:,0,0],B[:,:,0,0])

        return deepcopy(S)

    def spinor_inv(self, A):
    
        if self.dim[1] == 2:
            Z = self.scalar_operator(A) 
            Y = np.zeros_like(Z)
            Y[:,:,0] = np.linalg.inv(Z[:,:,0])
            S = self.spinor_operator(Y)
            #Y = np.linalg.inv(self.scalar_operator(A)[:,:,0])
            #S = self.spinor_operator(Y) 
        else:
            S = np.empty_like(A)
            for ispin in range(self.nspin):
                S[:,:,ispin,0] = np.linalg.inv(A[:,:,ispin,0])

        return deepcopy(S)

    def spinor_vector(self, V):

        vsize = V.shape[0]
        S = np.zeros((int(vsize/self.nbasis),self.nbasis),dtype=complex)

        S[:,0] = V[0:int(vsize/self.nbasis)] 
        if self.dim[1] == 2:
            S[:,1] = V[int(vsize/self.nbasis):vsize] 

        return S
