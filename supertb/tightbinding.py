"""
This module defines the classes and functions that enable to handle tight-binding
models
"""

import numpy as np
from math import exp, log, sin, cos, pi, sqrt, trunc, ceil
from copy import deepcopy
from supertb import Spinor, mpauli
import itertools
import inspect


class ParametrizedInteraction(object):
    """
    This class is intended to serve as a framework for child classes  
    SlaterKosterParams and Gerneralized SlaterKosterParams.
    """
    
    def __init__(self, table, hamiltonian=True, overlap=False, peierls=False, **kwargs):

        self.table = table
        self.spin = Spinor(**kwargs)

        self.overlap = overlap
        self.hamiltonian = hamiltonian
        self.peierls = peierls

class SlaterKosterParams(ParametrizedInteraction):
  
    """
    Slater-Koster decomposition of two_center_integrals E(i,j) in terms
    of bond-integrals B[A,B](k). The following definitions apply:

    E(i,j) 
    -----------------------------------------------------------
    idx    :  0    1    2    3    4    5    6    7     8    
    orbital:  s,   x,   y,   z,  xy,  yz,  xz, x2-y2, 3z2-r2


    B[A,B](i) for elemental systems
    -----------------------------------------------------------
    idx    :   0         1         2         3      4         5         6       7         8      9
    bond   : ss_sigma, sp_sigma, pp_sigma, pp_pi, sd_sigma, pd_sigma, pd_pi,  dd_sigma, dd_pi, dd_sigma
   
    Note that if species_A != species_B: 
    ps_sigma != sp_sigma, ds_sigma != sd_sigma, dp_sigma != pd_sigma, dp_pi != pd_pi.  

    """

    def __init__(self, table, **kwargs):
        """   
        Defines a Slater-Koster parametrization of two-center integrals.

        Args:
            table: (AtomicTable Object)
                Lists the NumericalAtoms involved in the two-center integrals
            hamiltonian: (Boolean)
                Wether the parametrized integrals are similar to components of 
                an Hamiltonian matrix
                Default: True
            overlap: (Boolean)
                Wether the parametrized integrals are similar to components of 
                an Overlap matrix
                Default: False
            spinpol: (Boolean)
                Determines wheter the integrals are spin dependent or not.
                Default: False.
            spinorb: (Boolean)
                Determines wheter the integrals involves spinor or scalar
                wavefunctions.
                Default: False.
            radialf: (Callable object)
                A callable object that, given the nominal hopping parameter and 
                the bond length, returns the effective hopping term.
                Simple examples are:
                    def radialf(hop, length):
                        return hop
                
                    def radialf(hop, length):
                        return hop*length**(-1)

                The callable object can be further tuned by passing additional 
                user defined keyword arguments. Note that all the keyword arguments
                have to be explicitely defined!
                Examples are: 

                    def radialf(hop, length, a=1., b=2.):
                        
                        return a*hop*length**(-b)

        """   
 
        # Initialise base parameters
        ParametrizedInteraction.__init__(self, table, **kwargs) 

        # Initialise internal containers 
        self.onsites = {}
        self.hoppings = {} 
        self.edge_coeffs = {}
        self.variations = []
     
        # Initialise the radial function 
        if 'radialf' in kwargs:
            self.edge_function = kwargs['radialf']
        else:
            self.edge_function = lambda alpha, length, **kwargs: alpha

        aargs = inspect.getfullargspec(self.edge_function)
        all_args, defvals = aargs.args, aargs.defaults

        edge_coeffs_default = {}
        for iarg, jarg in enumerate(range(2,len(all_args))):
            edge_coeffs_default[all_args[jarg]] = defvals[iarg]
        self.edge_coeffs['defaults'] = deepcopy(edge_coeffs_default)

        # Initialise container for onsite parameters 
        self.init_local_integrals()

        # Initialise container for variations of onsite parameters 
        self.init_local_variations()
 
        # Initialise container for hopping parameters (and radial functions)
        self.init_non_local_integrals()

        # Initialise container for variations of hopping parameters (and radial functions)
        self.init_non_local_variations()


    def non_local_block(self, edge, graph):

        i, j, k = edge
        data = graph[i][j][k]
        spec_a = graph.species[i]
        spec_b = graph.species[j]

        num_a = self.table[spec_a].num_orbitals 
        num_b = self.table[spec_b].num_orbitals 
        mat = np.zeros((num_a, num_b, self.spin.dim[0],self.spin.dim[1]),dtype=complex)

        vec = data['vector']
        nvec = np.linalg.norm(vec) 
        rvec = vec/nvec

        for iloc in range(num_a):
            iorb, ishell = self.table[spec_a].orbitals[iloc]
            for jloc in range(num_b):
                jorb, jshell = self.table[spec_b].orbitals[jloc]
        
                if iorb <= jorb:
                    spinmat = self.non_local_integral(spec_a, iorb, ishell,\
                                            spec_b, jorb, jshell, rvec, nvec)
                    mat[iloc,jloc,:,:] = spinmat
                else: 
                    spinmat = self.non_local_integral(spec_b, jorb, jshell,\
                                            spec_a, iorb, ishell, -rvec, nvec)
                    mat[iloc,jloc,:,:] = np.reshape(spinmat.conj().T,self.spin.dim)
 
        return mat
 
    def non_local_integral(self, spec_a, iorb, ishell, spec_b, jorb, jshell, rvec, length):

        l, m, n = rvec
        hop = self.hoppings[spec_a][spec_b][ishell][jshell]
        edge_coeffs = self.edge_coeffs[spec_a][spec_b][ishell][jshell]             

        def skint(ibond):
            return self.edge_function(hop[ibond], length, **edge_coeffs[ibond])

        # Compute the interatomic matrix element 
        if (iorb == 0):
            if (jorb == 0):
                me = skint(0)
            elif (jorb == 1):
                me = l*skint(1)
            elif (jorb == 2):
                me = m*skint(1)
            elif (jorb == 3):
                me = n*skint(1)
            elif (jorb == 4):
                me = sqrt(3.)*l*m*skint(4) 
            elif (jorb == 5):
                me = sqrt(3.)*m*n*skint(4) 
            elif (jorb == 6):
                me = sqrt(3.)*n*l*skint(4) 
            elif (jorb == 7):
                me = 0.5*sqrt(3.)*(l**2-m**2)*skint(4)
            elif (jorb == 8):
                me = (n**2 - 0.5*(l**2+m**2))*skint(4)

        elif (iorb == 1):
            if (jorb == 1):
                me = l**2*skint(2) + (1-l**2)*skint(3) 
            elif (jorb == 2):
                me = l*m*skint(2) - l*m*skint(3)
            elif (jorb == 3):
                me = l*n*skint(2) - l*n*skint(3)
            elif (jorb == 4):
                me = sqrt(3.)*l**2*m*skint(5) + m*(1.-2*l**2)*skint(6)
            elif (jorb == 5):
                me = sqrt(3.)*l*m*n*skint(5)-2*l*m*n*skint(6)
            elif (jorb == 6):
                me = sqrt(3.)*l**2*n*skint(5) + n*(1.-2*l**2)*skint(6)
            elif (jorb == 7):
                me = 0.5*sqrt(3.)*l*(l**2-m**2)*skint(5) + l*(1.-l**2+m**2)*skint(6)
            elif (jorb == 8):
                me = l*(n**2-0.5*(l**2+m**2))*skint(5) - sqrt(3.)*l*n**2*skint(6)

        elif (iorb == 2):
            if (jorb == 2):
                me = m**2*skint(2) + (1.-m**2)*skint(3) 
            elif (jorb == 3):
                me = m*n*skint(2) - m*n*skint(3)
            elif (jorb == 4):
                me = sqrt(3.)*m**2*l*skint(5) + l*(1.-2*m**2)*skint(6)
            elif (jorb == 5):
                me = sqrt(3.)*m**2*n*skint(5) + n*(1.-2*m**2)*skint(6)
            elif (jorb == 6):
                me = sqrt(3.)*l*m*n*skint(5)-2*l*m*n*skint(6)
            elif (jorb == 7):
                me = 0.5*sqrt(3.)*m*(l**2-m**2)*skint(5) - m*(1+l**2-m**2)*skint(6)
            elif (jorb == 8):
                me = m*(n**2-0.5*(l**2+m**2))*skint(5) - sqrt(3.)*m*n**2*skint(6)

        elif (iorb == 3):
            if (jorb == 3):
                me = n**2*skint(2) + (1.-n**2)*skint(3) 
            elif (jorb == 4):
                me = sqrt(3.)*l*m*n*skint(5)-2*l*m*n*skint(6)
            elif (jorb == 5):
                me = sqrt(3.)*n**2*m*skint(5) + m*(1.-2*n**2)*skint(6)
            elif (jorb == 6):
                me = sqrt(3.)*n**2*l*skint(5) + l*(1.-2*n**2)*skint(6)
            elif (jorb == 7):
                me = 0.5*sqrt(3.)*n*(l**2-m**2)*skint(5) - n*(l**2-m**2)*skint(6)
            elif (jorb == 8):
                me = n*(n**2-0.5*(l**2+m**2))*skint(5) + sqrt(3.)*n*(l**2+m**2)*skint(6)

        elif (iorb == 4):
            if (jorb == 4):
                me = 3.*l**2*m**2*skint(7) + (l**2+m**2-4.*l**2*m**2)*skint(8) + (n**2+l**2*m**2)*skint(9)
            elif (jorb == 5):
                me = 3.*l*m**2*n*skint(7) + l*n*(1.-4.*m**2)*skint(8) + l*n*(m**2-1)*skint(9)
            elif (jorb == 6):
                me = 3.*l**2*m*n*skint(7) + m*n*(1.-4.*l**2)*skint(8) + m*n*(l**2-1)*skint(9)
            elif (jorb == 7):
                me = 1.5*l*m*(l**2-m**2)*skint(7) + 2.*l*m*(m**2-l**2)*skint(8) + 0.5*l*m*(l**2-m**2)*skint(9)
            elif (jorb == 8):
                me = sqrt(3.)*( l*m*(n**2-0.5*(l**2+m**2))*skint(7) - 2.*l*m*n**2*skint(8) + 0.5*l*m*(1.+n**2)*skint(9) )

        elif (iorb == 5):
            if (jorb == 5):
                me = 3.*m**2*n**2*skint(7) + (m**2+n**2-4.*m**2*n**2)*skint(8) + (l**2+m**2*n**2)*skint(9)
            elif (jorb == 6):
                me = 3.*l*n**2*m*skint(7) + l*m*(1.-4.*n**2)*skint(8) + l*m*(n**2-1)*skint(9)
            elif (jorb == 7):
                me = 1.5*m*n*(l**2-m**2)*skint(7) - m*n*(1.+2.*(l**2-m**2))*skint(8) + m*n*(1.+0.5*(l**2-m**2))*skint(9)
            elif (jorb == 8):
                me = sqrt(3.)*( m*n*(n**2-0.5*(l**2+m**2))*skint(7) + m*n*(l**2+m**2-n**2)*skint(8) - m*n*0.5*(l**2+m**2)*skint(9) )

        elif (iorb == 6):
            if (jorb == 6):
                me = 3*l**2*n**2*skint(7) + (l**2+n**2-4.*l**2*n**2)*skint(8) + (m**2+n**2*l**2)*skint(9)
            elif (jorb == 7):
                me = 1.5*n*l*(l**2-m**2)*skint(7) + n*l*(1.-2.*(l**2-m**2))*skint(8) - n*l*(1.-0.5*(l**2-m**2))*skint(9)
            elif (jorb == 8):
                me = sqrt(3.)*( l*n*(n**2-0.5*(l**2+m**2))*skint(7) + l*n*(l**2+m**2-n**2)*skint(8) - l*n*0.5*(l**2+m**2)*skint(9) )

        elif (iorb == 7):
            if (jorb == 7):
                me = 0.75*(l**2-m**2)**2*skint(7) + (l**2 + m**2 - (l**2-m**2)**2)*skint(8) + (n**2+0.25*(l**2-m**2)**2)*skint(9) 
            elif (jorb == 8):
                me = sqrt(3.)*( (l**2-m**2)*(n**2-0.5*(l**2+m**2))*0.5*skint(7) + n**2*(m**2-l**2)*skint(8) + 0.25*(1+n**2)*(l**2-m**2)*skint(9) )

        elif (iorb == 8):
            if (jorb == 8):
                me = (n**2-0.5*(l**2+m**2))**2*skint(7) + 3.*n**2*(l**2+m**2)*skint(8) + 0.75*(l**2+m**2)**2*skint(9)

        return me
   
    def set_non_local_integrals(self, hoppings):
        """
        Sets one or more Slater-Koster interatomic matrix elements.
        (i.e. off-diagonal terms of Hamiltonian or Overlap matrices).

        Args:
            hoppings: (list of tuples)
                In the absence of user defined edge function, each tuple is read as,

                    (spec_a, spec_b, shell_a, shell_b, ibond, alpha)
                
                If an edge function has been defined, each tuple is read as, 

                    (spec_a, spec_b, shell_a, shell_b, ibond, alpha, params).

                Here, 'spec_a/b' determines the NumercialAtoms involved in the integral.
                'shell_a/b' specifies the shell of the atomic orbitals. 'ibond' determines
                the index of the bond, and 'alpha' is the Slater-Koster interatomic matrix 
                element. 'params' is a dictionary passed to the user defined edge
                function as keyword arguments.   
        """
       
        for hopping in hoppings:
            spec_a, spec_b, shell_a, shell_b, ibond, alpha = hopping[:6]

            if not hasattr(alpha,'__iter__'):
                spinmat = self.spin.identity*alpha
            else:
                 spinmat = np.reshape(alpha,self.spin.dim)

            self.hoppings[spec_a][spec_b][shell_a][shell_b][ibond] = spinmat

            if len(hopping) == 7:
                params = hopping[6]
                for key in params:
                    self.edge_coeffs[spec_a][spec_b][shell_a][shell_b][ibond][key] = params[key]
                
            if (spec_a == spec_b or ibond in [0,2,3,7,8,9]):
                self.hoppings[spec_b][spec_a][shell_b][shell_a][ibond] = \
                              np.reshape(spinmat.conj().T,self.spin.dim)
           
                if len(hopping) == 7 :
                    for key in params:
                        self.edge_coeffs[spec_b][spec_a][shell_b][shell_a][ibond][key] = params[key]

    def update_non_local_integrals(self, hoppings):

        #if not isinstance(hoppings[0],list) or \
        #    not isinstance(hoppings[0],tuple):
        #    hoppings = [hoppings]

        for hopping in hoppings:
            spec_a, spec_b, shell_a, shell_b, ibond, alpha = hopping[:6]

            if not hasattr(alpha,'__iter__'):
                spinmat = self.spin.identity*alpha
            else:
                spinmat = np.reshape(alpha,self.spin.dim)

            #print 'update_non_local_integrals ', spinmat, self.hoppings[spec_a][spec_b][shell_a][shell_b][ibond]
            self.hoppings[spec_a][spec_b][shell_a][shell_b][ibond] += spinmat

            if len(hopping) == 7:
                params = hopping[6]
                for key in params:
                    self.edge_coeffs[spec_a][spec_b][shell_a][shell_b][ibond][key] += params[key]
                
            if (spec_a == spec_b and shell_a != shell_b) or \
                (spec_a != spec_b and ibond in [0,2,3,7,8,9]):
                self.hoppings[spec_b][spec_a][shell_b][shell_a][ibond] += \
                              np.reshape(np.transpose(spinmat),self.spin.dim)
           
                if len(hopping) == 7 :
                    for key in params:
                        self.edge_coeffs[spec_b][spec_a][shell_b][shell_a][ibond][key] += params[key]

    def init_non_local_integrals(self):

        for spec_a in self.table:
            self.hoppings[spec_a] = {}
            self.edge_coeffs[spec_a] = {}
            for spec_b in self.table:
                self.hoppings[spec_a]\
                        [spec_b] = {}
                self.edge_coeffs[spec_a]\
                        [spec_b] = {}
                for shell_a in range(self.table[spec_a].num_shells):
                    self.hoppings[spec_a]\
                            [spec_b][shell_a] = {}
                    self.edge_coeffs[spec_a]\
                            [spec_b][shell_a] = {}
                    for shell_b in range(self.table[spec_b].num_shells):
                        self.hoppings[spec_a]\
                                [spec_b][shell_a][shell_b] = {}
                        self.edge_coeffs[spec_a]\
                                [spec_b][shell_a][shell_b] = {}

                        for ibond in range(10):
                            self.hoppings[spec_a]\
                                    [spec_b][shell_a][shell_b][ibond] = \
                                    np.zeros((self.spin.dim[0],self.spin.dim[1]))
                            self.edge_coeffs[spec_a]\
                                    [spec_b][shell_a][shell_b][ibond] = \
                                    deepcopy(self.edge_coeffs['defaults'])

    def set_non_local_variations(self, **kwargs):
        """
        Defines the allowed variation of one or more Slater-Koster interatomic 
        matrix elements. (i.e. off-diagonal terms of Hamiltonian or Overlap matrices).

        The actual variations are given as keyword arguments.

        Args: (keyword arguments)
            The value assoiated with each keyword sould be a list of tuples.
            In the absence of user defined edge function, the tuple is read as,

                (spec_a, spec_b, shell_a, shell_b, ibond, alpha)
            
            If an edge function has been defined, the tuple is read as, 

                (spec_a, spec_b, shell_a, shell_b, ibond, alpha, radial_params).

            Here, 'spec_a/b' determines the NumercialAtoms involved in the integral.
            'shell_a/b' specifies the shell of the atomic orbitals. 'ibond' determines
            the index of the bond, and 'alpha' is the Slater-Koster interatomic matrix 
            element. 'radial_params' is the dictionary passed to the user defined edge
            function.  
        """
        for key in kwargs:
            self.hoppings_delta[key] = kwargs[key]
            if key not in self.variations:
                self.variations.append(key)

    def apply_non_local_variation(self, key, coeff):
        
        variations = self.hoppings_delta[key]

        #if not isinstance(variations[0],list) or \
        #    not isinstance(variations[0],tuple):
        #    variations = [variations]

        for variation in variations:

            spec_a, spec_b, shell_a, shell_b, ibond, alpha = variation[:6]
            alpha = coeff*alpha

            if len(variation) == 7:
                params = deepcopy(variation[6])                 
                for key in params:
                    params[key] = coeff*params[key]
                self.update_non_local_integrals([(spec_a, spec_b, \
                    shell_a, shell_b, ibond, alpha, params)])
            
            else:
                self.update_non_local_integrals([(spec_a, spec_b, \
                    shell_a, shell_b, ibond, alpha)])

    def init_non_local_variations(self):

        self.hoppings_delta = {}

    def reset_non_local_variations(self):
        """
        Resets all user defined variations of the Slater-Koster interatomic 
        matrix elements.
        """
        for key in self.hoppings_delta:
            self.variations.remove(key)
        self.hoppings_delta = {}

    def precompute_local_blocks(self):

        local_blocks = {}
        for spec in self.table:
            num = self.table[spec].num_orbitals
            mat = np.zeros((num, num, self.spin.dim[0],self.spin.dim[1]),dtype=complex)
            
            for iloc in range(num):
                for jloc in range(num):
                    mat[iloc,jloc,:,:] = self.local_integral(spec, iloc, jloc)

            local_blocks[spec] = mat

        self.local_blocks = local_blocks

    def local_block(self, inode, graph):

        spec = graph.species[inode]
        if hasattr(self, 'local_blocks'):
            return self.local_blocks[spec]

        num = self.table[spec].num_orbitals
        mat = np.zeros((num, num, self.spin.dim[0],self.spin.dim[1]),dtype=complex)

        for iloc in range(num):
            for jloc in range(num):
                mat[iloc,jloc,:,:] = self.local_integral(spec, iloc, jloc)
       
        return mat 
 
    def local_integral(self, spec_a, iorb, jorb):

        return self.onsites[spec_a][iorb][jorb][:,:]

    def set_local_integrals(self, onsites):
        """
        Sets one or more Slater-Koster intra-atomic matrix elements.
        (i.e. diagonal terms of Hamiltonian or Overlap matrices).

        Args:
            onsites: (list of tuples)
                Each tuple is read as, 

                    (spec, iorb, jorb, alpha).

                Here, 'spec' determines the NumercialAtoms involved in the integral.
                'iorb' and 'jorb' determines the indexes of the two atomic orbitals. 
                'alpha' is the Slater-Koster intra-atomic matrix element. 
        """
        #if not hasattr(onsites[0],'__iter__'):
        #    onsites = [onsites]

        for onsite in onsites:
            spec, iorb, jorb, alpha = onsite[:4]

            if not hasattr(alpha,'__iter__'):
                spinmat = self.spin.identity*alpha 
            else:
                spinmat = np.reshape(alpha,self.spin.dim)

            self.onsites[spec][iorb][jorb][:,:] = spinmat
            self.onsites[spec][jorb][iorb][:,:] = np.reshape(spinmat.conj().T,self.spin.dim)

    def update_local_integrals(self, onsites):

        #if not hasattr(onsites[0],'__iter__'):
        #    onsites = [onsites]

        for onsite in onsites:
            spec, iorb, jorb, alpha = onsite[:4]

            if not hasattr(alpha,'__iter__'):
                spinmat = self.spin.identity*alpha 
            else:
                spinmat = np.reshape(alpha,self.spin.dim)

            self.onsites[spec][iorb][jorb][:,:] += spinmat
            if iorb != jorb:
                self.onsites[spec][jorb][iorb][:,:] += np.reshape(spinmat.conj().T,self.spin.dim)


    def init_local_integrals(self): 

        for spec in self.table:
            self.onsites[spec] = {}
            for iorb in range(self.table[spec].num_orbitals):
                self.onsites[spec][iorb] = {}
                for jorb in range(self.table[spec].num_orbitals):
                    self.onsites[spec][iorb][jorb] = \
                         np.zeros((self.spin.dim[0],self.spin.dim[1])).astype(complex)

        if self.overlap:
            for spec in self.table:
                for iorb in range(self.table[spec].num_orbitals):
                    self.onsites[spec][iorb][iorb][:,:] = \
                             self.spin.identity

    def set_local_variations(self, **kwargs):
        """
        Defines the allowed variation of one or more Slater-Koster intra-atomic 
        matrix elements. (i.e. block-diagonal terms of Hamiltonian or Overlap matrices).

        The actual variations are given as keyword arguments.

        Args: (keyword arguments)
            The value assoiated with each keyword sould be a list of tuples.
            Each tuple is read as, 

                (spec, iorb, jorb, alpha).

            Here, 'spec' determines the NumercialAtoms involved in the integral.
            'iorb' and 'jorb' determines the indexes of the two atomic orbitals. 
            'alpha' is the Slater-Koster intra-atomic matrix element. 
        """
        for key in kwargs:
            self.onsites_delta[key] = kwargs[key]
            if key not in self.variations:
                self.variations.append(key)

    def apply_local_variation(self, key, coeff):
        
        variations = self.onsites_delta[key]

        #if not hasattr(variations[0],'__iter__'):
        #    variations = [variations]

        for variation in variations:
            spec, iorb, jorb, alpha = variation[:4]
            alpha = coeff*alpha
             
            self.update_local_integrals([(spec, iorb, jorb, alpha)])

    def init_local_variations(self): 

        self.onsites_delta = {}

    def reset_local_variations(self): 
        """
        Resets all user defined variations of the Slater-Koster intra-atomic 
        matrix elements.
        """
        for key in self.onsites_delta:
            self.variations.remove(key)
        self.onsites_delta = {}

    def set_soc_integrals(self, spinorbs):

        #if not hasattr(spinorbs[0],'__iter__'):
        #    spinorbs = [spinorbs]

        for spinorb in spinorbs: 
            spec, shell, lsorb = spinorb

            soc = []
            if shell == 'p' or shell == 'P':
                for iiorb in range(self.table[spec].num_orbitals):
                    iorb, ishell = self.table[spec].orbitals[iiorb]
                    for jjorb in range(iiorb+1, self.table[spec].num_orbitals):
                        jorb, jshell = self.table[spec].orbitals[jjorb]
                    
                        if (iorb == 1):
                            # x, y
                            if (jorb == 2):
                                alpha = -1j*lsorb*mpauli('z')
                                soc.append((spec, iiorb, jjorb, alpha))
                            # x, z
                            elif (jorb == 3):
                                alpha = 1j*lsorb*mpauli('y')
                                soc.append((spec, iiorb, jjorb, alpha))
                    
                        elif (iorb == 2):
                            # y, z
                            if (jorb == 3):
                                alpha = -1j*lsorb*mpauli('x')
                                soc.append((spec, iiorb, jjorb, alpha))
                
            elif shell == 'd' or shell == 'D':
                for iiorb in range(self.table[spec].num_orbitals):
                    iorb, ishell = self.table[spec].orbitals[iiorb]
                    for jjorb in range(iiorb+1, self.table[spec].num_orbitals):
                        jorb, jshell = self.table[spec].orbitals[jjorb]
               
                        if (iorb == 4):
                            # xy, yz
                            if (jorb == 5):
                                alpha = 1j*lsorb*mpauli('y')
                                soc.append((spec, iiorb, jjorb, alpha))
                            # xy, xz
                            elif (jorb == 6):
                                alpha = -1j*lsorb*mpauli('x')
                                soc.append((spec, iiorb, jjorb, alpha))
                            # xy, x2-y2
                            elif (jorb == 7):
                                alpha = 2j*lsorb*mpauli('z')
                                soc.append((spec, iiorb, jjorb, alpha))
                    
                        elif (iorb == 5):
                            # yz, xz
                            if (jorb == 6):
                                alpha = 1j*lsorb*mpauli('z') 
                                soc.append((spec, iiorb, jjorb, alpha))
                            # yz, x2-y2
                            elif (jorb == 7):
                                alpha = -1j*lsorb*mpauli('x')
                                soc.append((spec, iiorb, jjorb, alpha))
                            # yz, 3z2-r2
                            elif (jorb == 8):
                                alpha = -1j*sqrt(3.)*lsorb*mpauli('x')
                                soc.append((spec, iiorb, jjorb, alpha))
                
                        elif (iorb == 6):
                            # xz, x2-y2
                            if (jorb == 7):
                                alpha = -1j*lsorb*mpauli('y')
                                soc.append((spec, iiorb, jjorb, alpha))
                            # xz, 3z2-r2
                            elif (jorb == 8):
                                alpha = 1j*sqrt(3.)*lsorb*mpauli('y')
                                soc.append((spec, iiorb, jjorb, alpha))
        
            self.update_local_integrals(soc)
 
    def set_soc_variations(self, **kwargs):

        for key in kwargs:
            spec, shell, lsorb = kwargs[key]

            soc_var = []
            if shell == 'p' or shell == 'P':
                for iiorb in range(self.table[spec].num_orbitals):
                    iorb, ishell = self.table[spec].orbitals[iiorb]
                    for jjorb in range(iiorb+1, self.table[spec].num_orbitals):
                        jorb, jshell = self.table[spec].orbitals[jjorb]
                
                        if (iorb == 1):
                            # x, y
                            if (jorb == 2):
                                alpha = -1j*lsorb*mpauli('z')
                                soc_var.append((spec, iiorb, jjorb, alpha))
                            # x, z
                            elif (jorb == 3):
                                alpha = 1j*lsorb*mpauli('y')
                                soc_var.append((spec, iiorb, jjorb, alpha))
                
                        elif (iorb == 2):
                            # y, z
                            if (jorb == 3):
                                alpha = -1j*lsorb*mpauli('x')
                                soc_var.append((spec, iiorb, jjorb, alpha))
                
            elif shell == 'd' or shell == 'D':
                for iiorb in range(self.table[spec].num_orbitals):
                    iorb, ishell = self.table[spec].orbitals[iiorb]
                    for jjorb in range(iiorb+1, self.table[spec].num_orbitals):
                        jorb, jshell = self.table[spec].orbitals[jjorb]
                
                        if (iorb == 4):
                            # xy, yz
                            if (jorb == 5):
                                alpha = 1j*lsorb*mpauli('y')
                                soc_var.append((spec, iiorb, jjorb, alpha))
                            # xy, xz
                            elif (jorb == 6):
                                alpha = -1j*lsorb*mpauli('x')
                                soc_var.append((spec, iiorb, jjorb, alpha))
                            # xy, x2-y2
                            elif (jorb == 7):
                                alpha = 2j*lsorb*mpauli('z')
                                soc_var.append((spec, iiorb, jjorb, alpha))
                    
                        elif (iorb == 5):
                            # yz, xz
                            if (jorb == 6):
                                alpha = 1j*lsorb*mpauli('z') 
                                soc_var.append((spec, iiorb, jjorb, alpha))
                            # yz, x2-y2
                            elif (jorb == 7):
                                alpha = -1j*lsorb*mpauli('x')
                                soc_var.append((spec, iiorb, jjorb, alpha))
                            # yz, 3z2-r2
                            elif (jorb == 8):
                                alpha = -1j*sqrt(3.)*lsorb*mpauli('x')
                                soc_var.append((spec, iiorb, jjorb, alpha))
                
                        elif (iorb == 6):
                            # xz, x2-y2
                            if (jorb == 7):
                                alpha = -1j*lsorb*mpauli('y')
                                soc_var.append((spec, iiorb, jjorb, alpha))
                            # xz, 3z2-r2
                            elif (jorb == 8):
                                alpha = 1j*sqrt(3.)*lsorb*mpauli('y')
                                soc_var.append((spec, iiorb, jjorb, alpha))
        
            dic = {key:soc_var} 
            self.set_local_variations(**dic)


    @property
    def optim_variables(self):
        return self.variations

    def update_optim_variables(self, **variables): 

        for key in variables:
            #print 'Update variable:',  key, variables[key]
            if key in self.onsites_delta:
                self.apply_local_variation(key, variables[key])
            elif key in self.hoppings_delta:
                self.apply_non_local_variation(key, variables[key])


class GeneralizedSlaterKosterParams(SlaterKosterParams):

    def __init__(self, table, **kwargs):
        """   
        Defines a generalized Slater-Koster parametrization of 
        two-center integrals.

        Args:
            table: (AtomicTable Object)
                Lists the NumericalAtoms involved in the two-center integrals
            hamiltonian: (Boolean)
                Wether the parametrized integrals are similar to components of 
                an Hamiltonian matrix
                Default: True
            overlap: (Boolean)
                Wether the parametrized integrals are similar to components of 
                an Overlap matrix
                Default: True
            spinpol: (Boolean)
                Determines wheter the integrals are spin dependent or not.
                Default: False.
            spinorb: (Boolean)
                Determines wheter the integrals involves spinor or scalar
                wavefunctions.
                Default: False.
            edgef: (Callable object)
                A callable object that, given the nominal hopping parameter and 
                the dictionary of bond properties, returns the effective hopping term.
                Simple examples are:

                    def radialf(hop, data):
                        return hop
                
                    def radialf(hop, data):
                        length=data['length']
                        return hop*length**(-1)

                    def radialf(hop, data):
                        length=data['length']
                        order=data['order']
                        
                        if order == 1: 
                            return hop*length**(-1)
                        elif order == 2: 
                            return hop*length**(-2)
                        else:
                            return 0.

                The callable object can be further tuned by passing additional 
                user defined keyword arguments. Note that all the keyword arguments
                have to be explicitely defined!
                E.g.:

                    def radialf(hop, data, a=1., b=2.):
                        length=data['length']
                        return a*hop*length**(-b)

            nodef: (Callable object)
                A callable object that, given the nominal onsite parameter and 
                the dictionary of node properties, returns the effective onsite term.
                E.g.:

                    def nodef(onsite, data):
                        z = data['coords'][2] 
                        if onsite != 0.:
                            return onsite+2*z
                        else:
                            return 0.

                The callable object can be further tuned by passing additional 
                user defined keyword arguments. Note that all the keyword arguments
                have to be explicitely defined!
                E.g.:

                    def nodef(onsite, data, E=2.5):
                        z = data['coords'][2] 
                        if onsite != 0.:
                            return onsite+E*z
                        else:
                            return 0.
        """   
    
        # Initialise general parameters
        ParametrizedInteraction.__init__(self, table, **kwargs) 

        # Initialise internal containers 
        self.onsites = {}
        self.hoppings = {} 
        self.edge_coeffs = {}
        self.node_coeffs = {}
        self.variations = []

        # Initialise geometrical function 
        if 'edgef' in kwargs:
            self.edge_function = kwargs['edgef']
        elif 'radialf' in kwargs:
            self.edge_function = kwargs['radialf']
        else:
            self.edge_function = lambda alpha, edge, graph, **kwargs: alpha

        #all_args, val_args, key_args, defvals = inspect.getargspec(self.edge_function)
        aargs = inspect.getfullargspec(self.edge_function)
        all_args, defvals = aargs.args, aargs.defaults

        edge_coeffs_default = {}
        for iarg, jarg in enumerate(range(3,len(all_args))):
            edge_coeffs_default[all_args[jarg]] = defvals[iarg]
        self.edge_coeffs['defaults'] = edge_coeffs_default 

        # Initialise local function 
        if 'nodef' in kwargs:
            self.node_function = kwargs['nodef']
        else:
            self.node_function = lambda alpha, node, graph, **kwargs: alpha

        #all_args, val_args, key_args, defvals = inspect.getargspec(self.node_function)
        aargs = inspect.getfullargspec(self.node_function)
        all_args, defvals = aargs.args, aargs.defaults

        node_coeffs_default = {}
        for iarg, jarg in enumerate(range(3,len(all_args))):
            node_coeffs_default[all_args[jarg]] = defvals[iarg]
        self.node_coeffs['defaults'] = deepcopy(node_coeffs_default) 

        # Initialise container for onsite parameters 
        self.init_local_integrals()

        # Initialise container for variations of onsite parameters 
        self.init_local_variations()

        # Initialise container for hopping parameters (and radial functions)
        self.init_non_local_integrals()

        # Initialise container for variations of hopping parameters (and radial functions)
        self.init_non_local_variations()

        # Initialise contained for global parameters
        self.global_coeffs = {}
        self.global_delta = {}

    def non_local_block(self, edge, graph):

        i, j, k = edge
        data = graph[i][j][k]
        spec_a = graph.species[i]
        spec_b = graph.species[j]

        num_a = self.table[spec_a].num_orbitals 
        num_b = self.table[spec_b].num_orbitals 
        mat = np.zeros((num_a, num_b, self.spin.dim[0],self.spin.dim[1]),dtype=complex)

        vec = data['vector']
        nvec = np.linalg.norm(vec) 
        rvec = vec/nvec

        for iloc in range(num_a):
            iorb, ishell = self.table[spec_a].orbitals[iloc]
            for jloc in range(num_b):
                jorb, jshell = self.table[spec_b].orbitals[jloc]
              
                if iorb <= jorb:
                    spinmat = self.non_local_integral(spec_a, iorb, ishell,\
                                            spec_b, jorb, jshell, rvec, edge, graph)
                    mat[iloc,jloc,:,:] = spinmat
                else: 
                    spinmat = self.non_local_integral(spec_b, jorb, jshell,\
                                            spec_a, iorb, ishell, -rvec, edge, graph)
                    mat[iloc,jloc,:,:] = np.reshape(np.transpose(spinmat),self.spin.dim)

        return mat
 
    def local_block(self, inode, graph):

        data = graph.nodes[inode]
        spec = graph.species[inode]
        num = self.table[spec].num_orbitals
        mat = np.zeros((num, num, self.spin.dim[0],self.spin.dim[1]),dtype=complex)

        for iloc in range(num):
            for jloc in range(num):
                mat[iloc,jloc,:,:] = self.local_integral(spec, iloc, jloc, inode, graph)
       
        return mat 
 
    def local_integral(self, spec_a, iorb, jorb, node, graph):

        #print spec_a, node, iorb, jorb, self.node_coeffs[spec_a][iorb][jorb]
        onsite = self.onsites[spec_a][iorb][jorb]
        node_coeffs = self.node_coeffs[spec_a][iorb][jorb]
        global_coeffs = self.global_coeffs 
        
        params = dict(list(node_coeffs.items())+list(global_coeffs.items()))
        return self.node_function(onsite, node, graph, **params)

    def set_local_integrals(self, onsites):
        """
        Onsites should be a list of tuples defined as
        (spec, iorb, jorb, alpha)
        """ 
        #if not hasattr(onsites[0],'__iter__'):
        #    onsites = [onsites]

        for onsite in onsites:
            spec, iorb, jorb, alpha = onsite[:4]

            if not hasattr(alpha,'__iter__'):
                spinmat = self.spin.identity*alpha 
            else:
                spinmat = np.reshape(alpha,self.spin.dim)

            self.onsites[spec][iorb][jorb][:,:] = spinmat
            self.onsites[spec][jorb][iorb][:,:] = np.reshape(spinmat.conj().T,self.spin.dim)

            if len(onsite) == 5:
                params = onsite[4]
                for key in params:
                    self.node_coeffs[spec][iorb][jorb][key] = params[key]
                    self.node_coeffs[spec][jorb][iorb][key] = params[key]

    def init_local_integrals(self): 
        """
        Initialise container for onsite parameters 
        """
        for spec in self.table:
            self.onsites[spec] = {}
            self.node_coeffs[spec] = {}
            for iorb in range(self.table[spec].num_orbitals):
                self.onsites[spec][iorb] = {}
                self.node_coeffs[spec][iorb] = {}
                for jorb in range(self.table[spec].num_orbitals):
                    self.onsites[spec][iorb][jorb] = \
                         np.zeros((self.spin.dim[0],self.spin.dim[1])).astype(complex)
                    self.node_coeffs[spec][iorb][jorb] = deepcopy(self.node_coeffs['defaults'])

        if self.overlap:
            for spec in self.table:
                for iorb in range(self.table[spec].num_orbitals):
                    self.onsites[spec][iorb][iorb][:,:] = \
                             self.spin.identity

    def apply_local_variation(self, key, coeff):
        
        variations = self.onsites_delta[key]
        #print ('variations :', variations)

        #if not hasattr(variations[0],'__iter__'):
        #    variations = [variations]

        for variation in variations:
            spec, iorb, jorb, alpha = variation[:4]
            alpha = coeff*alpha
    
            if len(variation) == 5:
                params = deepcopy(variation[4])
                for key in params:
                    params[key] = coeff*params[key]
                #print 'apply local :', spec, iorb, jorb, coeff, alpha, params
                self.update_local_integrals([(spec, iorb, jorb, alpha, params)])
            else:
                self.update_local_integrals([(spec, iorb, jorb, alpha)])

    def update_local_integrals(self, onsites):
        """
        Onsites should be a list of tuples defined as
        (spec, iorb, jorb, alpha)
        """ 
        #if not hasattr(onsites[0],'__iter__'):
        #    onsites = [onsites]

        for onsite in onsites:
            spec, iorb, jorb, alpha = onsite[:4]

            if not hasattr(alpha,'__iter__'):
                spinmat = self.spin.identity*alpha 
            else:
                spinmat = np.reshape(alpha,self.spin.dim)
    
            #print '...before :', self.onsites[spec][iorb][jorb][:,:], self.node_coeffs[spec][iorb][jorb]
            self.onsites[spec][iorb][jorb][:,:] += spinmat
            if iorb != jorb:
                self.onsites[spec][jorb][iorb][:,:] += np.reshape(spinmat.conj().T,self.spin.dim)

            if len(onsite) == 5: 
                params = onsite[4]
                for key in params:
                    self.node_coeffs[spec][iorb][jorb][key] += params[key]
                    if iorb != jorb:
                        self.node_coeffs[spec][jorb][iorb][key] += params[key] 

            #print '...after :', self.onsites[spec][iorb][jorb][:,:], self.node_coeffs[spec][iorb][jorb]


    def non_local_integral(self, spec_a, iorb, ishell, spec_b, jorb, jshell, rvec, edge, graph):

        l, m, n = rvec
        hop = self.hoppings[spec_a][spec_b][ishell][jshell]
        edge_coeffs = self.edge_coeffs[spec_a][spec_b][ishell][jshell]             
        global_coeffs = self.global_coeffs 
        
        def skint(ibond):
            params = dict(list(edge_coeffs[ibond].items())+list(global_coeffs.items()))
            return self.edge_function(hop[ibond], edge, graph, **params)

        # Compute the interatomic matrix element 
        if (iorb == 0):
            if (jorb == 0):
                me = skint(0)
            elif (jorb == 1):
                me = l*skint(1)
            elif (jorb == 2):
                me = m*skint(1)
            elif (jorb == 3):
                me = n*skint(1)
            elif (jorb == 4):
                me = sqrt(3.)*l*m*skint(4) 
            elif (jorb == 5):
                me = sqrt(3.)*m*n*skint(4) 
            elif (jorb == 6):
                me = sqrt(3.)*n*l*skint(4) 
            elif (jorb == 7):
                me = 0.5*sqrt(3.)*(l**2-m**2)*skint(4)
            elif (jorb == 8):
                me = (n**2 - 0.5*(l**2+m**2))*skint(4)

        elif (iorb == 1):
            if (jorb == 1):
                me = l**2*skint(2) + (1-l**2)*skint(3) 
            elif (jorb == 2):
                me = l*m*skint(2) - l*m*skint(3)
            elif (jorb == 3):
                me = l*n*skint(2) - l*n*skint(3)
            elif (jorb == 4):
                me = sqrt(3.)*l**2*m*skint(5) + m*(1.-2*l**2)*skint(6)
            elif (jorb == 5):
                me = sqrt(3.)*l*m*n*skint(5)-2*l*m*n*skint(6)
            elif (jorb == 6):
                me = sqrt(3.)*l**2*n*skint(5) + n*(1.-2*l**2)*skint(6)
            elif (jorb == 7):
                me = 0.5*sqrt(3.)*l*(l**2-m**2)*skint(5) + l*(1.-l**2+m**2)*skint(6)
            elif (jorb == 8):
                me = l*(n**2-0.5*(l**2+m**2))*skint(5) - sqrt(3.)*l*n**2*skint(6)

        elif (iorb == 2):
            if (jorb == 2):
                me = m**2*skint(2) + (1.-m**2)*skint(3) 
            elif (jorb == 3):
                me = m*n*skint(2) - m*n*skint(3)
            elif (jorb == 4):
                me = sqrt(3.)*m**2*l*skint(5) + l*(1.-2*m**2)*skint(6)
            elif (jorb == 5):
                me = sqrt(3.)*m**2*n*skint(5) + n*(1.-2*m**2)*skint(6)
            elif (jorb == 6):
                me = sqrt(3.)*l*m*n*skint(5)-2*l*m*n*skint(6)
            elif (jorb == 7):
                me = 0.5*sqrt(3.)*m*(l**2-m**2)*skint(5) - m*(1+l**2-m**2)*skint(6)
            elif (jorb == 8):
                me = m*(n**2-0.5*(l**2+m**2))*skint(5) - sqrt(3.)*m*n**2*skint(6)

        elif (iorb == 3):
            if (jorb == 3):
                me = n**2*skint(2) + (1.-n**2)*skint(3) 
            elif (jorb == 4):
                me = sqrt(3.)*l*m*n*skint(5)-2*l*m*n*skint(6)
            elif (jorb == 5):
                me = sqrt(3.)*n**2*m*skint(5) + m*(1.-2*n**2)*skint(6)
            elif (jorb == 6):
                me = sqrt(3.)*n**2*l*skint(5) + l*(1.-2*n**2)*skint(6)
            elif (jorb == 7):
                me = 0.5*sqrt(3.)*n*(l**2-m**2)*skint(5) - n*(l**2-m**2)*skint(6)
            elif (jorb == 8):
                me = n*(n**2-0.5*(l**2+m**2))*skint(5) + sqrt(3.)*n*(l**2+m**2)*skint(6)

        elif (iorb == 4):
            if (jorb == 4):
                me = 3.*l**2*m**2*skint(7) + (l**2+m**2-4.*l**2*m**2)*skint(8) + (n**2+l**2*m**2)*skint(9)
            elif (jorb == 5):
                me = 3.*l*m**2*n*skint(7) + l*n*(1.-4.*m**2)*skint(8) + l*n*(m**2-1)*skint(9)
            elif (jorb == 6):
                me = 3.*l**2*m*n*skint(7) + m*n*(1.-4.*l**2)*skint(8) + m*n*(l**2-1)*skint(9)
            elif (jorb == 7):
                me = 1.5*l*m*(l**2-m**2)*skint(7) + 2.*l*m*(m**2-l**2)*skint(8) + 0.5*l*m*(l**2-m**2)*skint(9)
            elif (jorb == 8):
                me = sqrt(3.)*( l*m*(n**2-0.5*(l**2+m**2))*skint(7) - 2.*l*m*n**2*skint(8) + 0.5*l*m*(1.+n**2)*skint(9) )

        elif (iorb == 5):
            if (jorb == 5):
                me = 3.*m**2*n**2*skint(7) + (m**2+n**2-4.*m**2*n**2)*skint(8) + (l**2+m**2*n**2)*skint(9)
            elif (jorb == 6):
                me = 3.*l*n**2*m*skint(7) + l*m*(1.-4.*n**2)*skint(8) + l*m*(n**2-1)*skint(9)
            elif (jorb == 7):
                me = 1.5*m*n*(l**2-m**2)*skint(7) - m*n*(1.+2.*(l**2-m**2))*skint(8) + m*n*(1.+0.5*(l**2-m**2))*skint(9)
            elif (jorb == 8):
                me = sqrt(3.)*( m*n*(n**2-0.5*(l**2+m**2))*skint(7) + m*n*(l**2+m**2-n**2)*skint(8) - m*n*0.5*(l**2+m**2)*skint(9) )

        elif (iorb == 6):
            if (jorb == 6):
                me = 3*l**2*n**2*skint(7) + (l**2+n**2-4.*l**2*n**2)*skint(8) + (m**2+n**2*l**2)*skint(9)
            elif (jorb == 7):
                me = 1.5*n*l*(l**2-m**2)*skint(7) + n*l*(1.-2.*(l**2-m**2))*skint(8) - n*l*(1.-0.5*(l**2-m**2))*skint(9)
            elif (jorb == 8):
                me = sqrt(3.)*( l*n*(n**2-0.5*(l**2+m**2))*skint(7) + l*n*(l**2+m**2-n**2)*skint(8) - l*n*0.5*(l**2+m**2)*skint(9) )

        elif (iorb == 7):
            if (jorb == 7):
                me = 0.75*(l**2-m**2)**2*skint(7) + (l**2 + m**2 - (l**2-m**2)**2)*skint(8) + (n**2+0.25*(l**2-m**2)**2)*skint(9) 
            elif (jorb == 8):
                me = sqrt(3.)*( (l**2-m**2)*(n**2-0.5*(l**2+m**2))*0.5*skint(7) + n**2*(m**2-l**2)*skint(8) + 0.25*(1+n**2)*(l**2-m**2)*skint(9) )

        elif (iorb == 8):
            if (jorb == 8):
                me = (n**2-0.5*(l**2+m**2))**2*skint(7) + 3.*n**2*(l**2+m**2)*skint(8) + 0.75*(l**2+m**2)**2*skint(9)

        return me


    def set_global_parameters(self, **params):

        for key in params:
            self.global_coeffs[key] = params[key]

    def update_global_parameters(self, **params):
  
        for key in params:
            self.global_coeffs[key] += params[key]

    def set_global_variations(self, **kwargs):

        for key in kwargs:
            self.global_delta[key] = kwargs[key]
            if key not in self.variations:
                self.variations.append(key)

    def apply_global_variation(self, key, coeff):

        variations = self.global_delta[key]

        for variation in variations:
            for vkey in variation:
                alpha = variation[vkey]*coeff
                var_dic = {vkey:alpha}
                self.update_global_parameters(**var_dic)

    def reset_global_variations(self): 

        for key in self.global_delta:
            self.variations.remove(key)
        self.global_delta = {}

    @property
    def optim_variables(self):
        return self.variations

    def update_optim_variables(self, **variables): 

        for key in variables:
            #print 'Update variable:',  key, variables[key]
            if key in self.onsites_delta:
                self.apply_local_variation(key, variables[key])
            elif key in self.hoppings_delta:
                self.apply_non_local_variation(key, variables[key])
            elif key in self.global_delta:
                self.apply_global_variation(key, variables[key])

   
class BlochPhase(ParametrizedInteraction):

    def __init__(self, table, **kwargs):
        
        ParametrizedInteraction.__init__(self, table, \
            hamiltonian=False, overlap=False, peierls=True, **kwargs) 

    def non_local_block(self, vec, kpt):
 
        return np.exp(1j*np.dot(kpt,vec))

