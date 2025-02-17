
from math import exp, log, sin, cos, pi, sqrt, trunc, ceil
from copy import copy, deepcopy

import numpy as np
import pickle
import scipy
import scipy.linalg as linalg
import warnings


class Optimable(dict):

    def __init__(self):

        super(dict,self).__init__()

        self._external_variables = []
        self._internal_variables = []

        self.penalty_functions = {}
        self.penalty_functions['bands'] = Optimable.bands_rms
        self.penalty_functions['spins'] = Optimable.spin_rms
        self.penalty_functions['dspins'] = Optimable.dspin_rms
        self.penalty_functions['dbs'] = Optimable.dbands_spin_rms
        self.penalty_functions['dbands'] = Optimable.delta_bands_rms
        self.penalty_functions['ibands'] = Optimable.bands_inverse
        #self.penalty_functions['rspins'] = Optimable.relative_spins_rms
        self.penalty_functions['rbands'] = Optimable.relative_bands_rms
        self.penalty_functions['wfermi'] = Optimable.energy_window
        

    @classmethod
    def dbands_spin_rms(cls, optim, ref, include=[], compute=[], deltas=[], \
                  cbands=1., cspins=1.,\
                  rkpt=0, rband=0, rspin=0, fac=1., default=10., **kwargs):

        # Initialise smearing as Gauss-like function with parameters
        # sigma and mu 
        if 'mu' in kwargs and 'sigma' in kwargs:
            mu = kwargs['mu']
            sigma = kwargs['sigma']
            smear = lambda e : 0.5*scipy.special.erfc((e-mu)/sigma) 
        # Use the smearing function provided as argument 
        elif 'smear' in kwargs:
            smear = kwargs['smear']
        # No smearing
        else:
            smear = lambda e : 1. 

        optim.compute_integrals(*compute)
        try:
            eig = optim.eigenvalues(kpts=ref.kpts,wkpts=ref.wkpts,include=include, eigvec=True)
            eig.evaluate_spin_direction3(flat=True)
        except:
            return default

        rms1 = 0.
        rms2 = 0.
        for ikp in ref:
            wk = ref.wkpts[ikp]
            
            for iband, ispin, jband, jspin in deltas:

                if iband in ref[ikp] and \
                   jband in ref[ikp]: 

                    e = (ref[ikp][jband][jspin]['E'] + \
                         ref[ikp][iband][ispin]['E'] )/2. \
                         - ref[rkpt][rband][rspin]['E'] 

                    delta_ref = ref[ikp][jband][jspin]['E'] - \
                                ref[ikp][iband][ispin]['E']
                    delta = eig[ikp][jband][jspin]['E'] - \
                            eig[ikp][iband][ispin]['E']

                    rms1 += wk*smear(e)*(delta-delta_ref)**2

                    delta_spin = np.linalg.norm(np.array(eig[ikp][jband][jspin]['S']) - np.array(ref[ikp][jband][jspin]['S'])) + \
                                     np.linalg.norm(np.array(eig[ikp][iband][ispin]['S']) - np.array(ref[ikp][iband][ispin]['S']))

                    rms2 += wk*smear(e)*delta_spin**2

        return (cbands*np.sqrt(rms1)+cspins*np.sqrt(rms2))*fac

    @classmethod
    def dspin_rms(cls, optim, ref, include=[], compute=[], deltas=[], \
                  cbands=1., cspins=1.,\
                  rkpt=0, rband=0, rspin=0, fac=1., default=10., **kwargs):

        # Initialise smearing as Gauss-like function with parameters
        # sigma and mu 
        if 'mu' in kwargs and 'sigma' in kwargs:
            mu = kwargs['mu']
            sigma = kwargs['sigma']
            smear = lambda e : 0.5*scipy.special.erfc((e-mu)/sigma) 
        # Use the smearing function provided as argument 
        elif 'smear' in kwargs:
            smear = kwargs['smear']
        # No smearing
        else:
            smear = lambda e : 1. 

        optim.compute_integrals(*compute)
        try:
            eig = optim.eigenvalues(kpts=ref.kpts,wkpts=ref.wkpts,include=include, eigvec=True)
            eig.evaluate_spin_direction3(flat=True)
        except:
            return default

        rms1 = 0.
        rms2 = 0.
        for ikp in ref:
            wk = ref.wkpts[ikp]
            
            for iband, ispin, jband, jspin in deltas:

                if iband in ref[ikp] and \
                   jband in ref[ikp]: 

                    e = (ref[ikp][jband][jspin]['E'] + \
                         ref[ikp][iband][ispin]['E'] )/2. \
                         - ref[rkpt][rband][rspin]['E'] 

                    delta_ref = ref[ikp][jband][jspin]['E'] - \
                                ref[ikp][iband][ispin]['E']
                    delta = eig[ikp][jband][jspin]['E'] - \
                            eig[ikp][iband][ispin]['E']

                    rms1 += wk*smear(e)*(delta-delta_ref)**2

                    delta_spin_ref = np.array(ref[ikp][jband][jspin]['S']) - \
                                     np.array(ref[ikp][iband][jspin]['S'])
                    delta_spin = np.array(eig[ikp][jband][jspin]['S']) - \
                                 np.array(eig[ikp][iband][jspin]['S'])

                    rms2 += wk*smear(e)*np.linalg.norm(delta_spin-delta_spin_ref)**2

        return (cbands*np.sqrt(rms1)+cspins*np.sqrt(rms2))*fac

    @classmethod
    def spin_rms(cls, optim, ref, include=[], compute=[], rkpt=0, rband=0, rspin=0, fac=1., default=10., **kwargs):

        # Initialise smearing as Gauss-like function with parameters
        # sigma and mu 
        if 'mu' in kwargs and 'sigma' in kwargs:
            mu = kwargs['mu']
            sigma = kwargs['sigma']
            smear = lambda e : 0.5*scipy.special.erfc((e-mu)/sigma) 
        # Use the smearing function provided as argument 
        elif 'smear' in kwargs:
            smear = kwargs['smear']
        # No smearing
        else:
            smear = lambda e : 1. 

        optim.compute_integrals(*compute)
        try:
            eig = optim.eigenvalues(kpts=ref.kpts,wkpts=ref.wkpts,include=include, eigvec=True)
            eig.evaluate_spin_direction3(flat=True)
        except:
            return default

        rms = 0.
        for ikp in ref:
            wk = ref.wkpts[ikp]
            for iband in ref[ikp]:
                for ispin in ref[ikp][iband]:
                    e = ref[ikp][iband][ispin]['E'] - ref[rkpt][rband][rspin]['E'] 
                    delta_spin = np.linalg.norm(np.array(ref[ikp][iband][ispin]['S']) - \
                                 np.array(eig[ikp][iband][ispin]['S']))
                    rms += wk*smear(e)*(delta_spin)**2
      
        return np.sqrt(rms)*fac
#    def relative_spins_rms(cls, optim, ref, include=[], compute=[], fac=1.):
#
#        optim.compute_integrals(*compute)
#        eig = optim.eigenvalues(kpts=ref.kpts,wkpts=ref.wkpts,include=include)
#
#        rms = 0.
#        for ikp in ref:
#            wk = ref.wkpts[ikp]
#            for iband in ref[ikp]:
#                delta_ref = ref[ikp][iband][1]['E'] - \
#                             ref[ikp][iband][0]['E']
#                delta_spin = eig[ikp][iband][1]['E'] - \
#                             eig[ikp][iband][0]['E']
#                rms += wk*(delta_spin-delta_ref)**2
#
#        return np.sqrt(rms)*fac


    @classmethod
    def delta_bands_rms(cls, optim, ref, include=[], compute=[], deltas=[],fac=1., \
                        rkpt=0, rband=0, rspin=0,
                        default=10., spin_split=False, **kwargs):

        # Initialise smearing as Gauss-like function with parameters
        # sigma and mu 
        if 'mu' in kwargs and 'sigma' in kwargs:
            mu = kwargs['mu']
            sigma = kwargs['sigma']
            smear = lambda e : 0.5*scipy.special.erfc((e-mu)/sigma) 
        # Use the smearing function provided as argument 
        elif 'smear' in kwargs:
            smear = kwargs['smear']
        # No smearing
        else:
            smear = lambda e : 1. 

        optim.compute_integrals(*compute)
        if not spin_split:
            try: 
                eig = optim.eigenvalues(kpts=ref.kpts,wkpts=ref.wkpts,include=include)
            except:
                return default
        else:
            try: 
                eig = optim.eigenvalues(kpts=ref.kpts,wkpts=ref.wkpts,include=include, eigvec=True)
                eig.evaluate_spin_direction3(flat=True)
            except:
                return default

        rms = 0.
        for ikp in ref:
            wk = ref.wkpts[ikp]
            for iband, ispin, jband, jspin in deltas:

                if iband in ref[ikp] and \
                   jband in ref[ikp]: 

                    e = (ref[ikp][jband][jspin]['E'] + \
                         ref[ikp][iband][ispin]['E'] )/2. \
                         - ref[rkpt][rband][rspin]['E'] 

                    if not spin_split:
                        delta_ref = ref[ikp][jband][jspin]['E'] - \
                                    ref[ikp][iband][ispin]['E']
                        delta = eig[ikp][jband][jspin]['E'] - \
                                eig[ikp][iband][ispin]['E']

                        rms += wk*smear(e)*(delta-delta_ref)**2
                    else:
                        if ref[ikp][jband][jspin]['S'][2] >= 0.:
                            delta_ref = ref[ikp][jband][jspin]['E'] - \
                                        ref[ikp][iband][ispin]['E']
                        else:
                            delta_ref = ref[ikp][iband][ispin]['E'] - \
                                        ref[ikp][jband][jspin]['E']

                        if eig[ikp][jband][jspin]['S'][2] >= 0.:
                            delta = eig[ikp][jband][jspin]['E'] - \
                                    eig[ikp][iband][ispin]['E']
                        else:
                            delta = eig[ikp][iband][ispin]['E'] - \
                                    eig[ikp][jband][jspin]['E']
                        rms += wk*smear(e)*(delta-delta_ref)**2

        return np.sqrt(rms)*fac

    @classmethod
    def bands_inverse(cls, optim, ref, include=[], compute=[], fac=1.):

        optim.compute_integrals(*compute)
        eig = optim.eigenvalues(kpts=ref.kpts,wkpts=ref.wkpts,include=include)

        rms = 0.
        for ikp in ref:
            wk = ref.wkpts[ikp]
            for iband in ref[ikp]:

                for jband in [iband-1,iband+1]:
                    if jband in ref[ikp]:

                        for ispin in ref[ikp][iband]:
                   
                            delta_ref = 1./(ref[ikp][iband][ispin]['E'] - \
                                         ref[ikp][jband][ispin]['E'])
                            delta = 1./(eig[ikp][iband][ispin]['E'] - \
                                         eig[ikp][jband][ispin]['E'])

                            rms += wk*(delta-delta_ref)**2

      
        return np.sqrt(rms)*fac


    @classmethod
    def energy_window(cls, optim, ref, include=[], compute=[], rkpt=0, rband=[0,0], rspin=0, emin=-6., emax=6., sigma=2., fac=1., default=10.): 

        # Initialise smearing function 
        def smear(x):

            if x >= 0.:
                sf = 0.5*scipy.special.erfc((x-emax)/sigma)
            elif x < 0.:
                sf = 0.5*scipy.special.erfc(-(x-emin)/sigma)

            return sf

        optim.compute_integrals(*compute)
        try:
            eig = optim.eigenvalues(kpts=ref.kpts,wkpts=ref.wkpts,include=include)
        except:
            return default

        rms = 0.
        eref0 = (ref[rkpt][rband[0]][rspin]['E'] + ref[rkpt][rband[1]][rspin]['E'])/2
        eref1 = (eig[rkpt][rband[0]][rspin]['E'] + eig[rkpt][rband[1]][rspin]['E'])/2
        

        for ikp in ref:
            wk = ref.wkpts[ikp]
            for iband in ref[ikp]:
                for ispin in ref[ikp][iband]:
                    delta_ref = ref[ikp][iband][ispin]['E'] - eref0
                    delta = eig[ikp][iband][ispin]['E'] - eref1
                    rms += wk*smear(delta_ref)*(delta-delta_ref)**2
 
        return np.sqrt(rms)*fac


    @classmethod
    def relative_bands_rms(cls, optim, ref, include=[], compute=[], rkpt=0, rband=0, rspin=0, fac=1., default=10., **kwargs):

        # Initialise smearing as Gauss-like function with parameters
        # sigma and mu 
        if 'mu' in kwargs and 'sigma' in kwargs:
            mu = kwargs['mu']
            sigma = kwargs['sigma']
            smear = lambda e : 0.5*scipy.special.erfc((e-mu)/sigma) 
        # Use the smearing function provided as argument 
        elif 'smear' in kwargs:
            smear = kwargs['smear']
        # No smearing
        else:
            smear = lambda e : 1. 

        optim.compute_integrals(*compute)
        try:
            eig = optim.eigenvalues(kpts=ref.kpts,wkpts=ref.wkpts,include=include)
        except:
            return default

        rms = 0.
        for ikp in ref:
            wk = ref.wkpts[ikp]
            for iband in ref[ikp]:
                for ispin in ref[ikp][iband]:
                    delta_ref = ref[ikp][iband][ispin]['E'] - \
                                 ref[rkpt][rband][rspin]['E']
                    delta = eig[ikp][iband][ispin]['E'] - \
                                 eig[rkpt][rband][rspin]['E']
                    rms += wk*smear(delta_ref)*(delta-delta_ref)**2

      
        return np.sqrt(rms)*fac


    @classmethod
    def bands_rms(cls, optim, ref, include=[], compute=[], fac=1.):

        optim.compute_integrals(*compute)
        eig = optim.eigenvalues(kpts=ref.kpts,wkpts=ref.wkpts,include=include)

        rms = 0.
        for ikp in ref:
            wk = ref.wkpts[ikp]
            for iband in ref[ikp]:
                for ispin in ref[ikp][iband]:
                    delta = eig[ikp][iband][ispin]['E'] - \
                            ref[ikp][iband][ispin]['E']
                    rms += wk*(delta)**2

        return np.sqrt(rms)*fac


    def add_item(self, label, optim, ref, penalty, *args, **kwargs):
        """
        Add an item (i.e. a term to the optimization function).
       
        Args:
	    label (string): 
                Name given to the item.
            optim (optimisable object):
                Optimisable objects should have a method 'optim_variables'
                and a method update_optim_variables
            ref (object):
                Any object used as reference for the optimisation
            penalty (string): 
                Name of the user defined penalty function to be used for
                the optimization (see Optimable.define_new_penalty_function)
        """
        self[label] = {}
        self[label]['optim'] = optim
        self[label]['ref'] = ref
        self[label]['penalty'] = penalty
        self[label]['args'] = args
        self[label]['kwargs'] = kwargs

        for key in self[label]['optim'].optim_variables:
            if key not in self._external_variables:
                self._external_variables.append(key)
                self._internal_variables.append(key)
        
    def define_new_penalty_function(self, label, pfunc):
        """
        Add a user defined penalty function to the set
        of available penalty functions.
       
        Args:
	    label (string): 
                Name given to the penatly function.
            pfunc (function): 
                For a given item, the penalty function is called as
                pfunc(optim, ref, *args, **kwargs) where optim, ref,
                args, and kwargs are the arguments associated with
                the definition of the item (see Optimable.add_item)
        """
        self.penalty_functions[label] = pfunc
             

    def add_constraint(self, label, vars_in, vars_out, cfunc):
        """
        Add a constraint to the optimization i.e. internal relations 
        in between the optimization parameters. Here the constraints
        have to be expressed in terms of internal degrees of freedom 
        (internal variables).
 
        For example, the optimization variables {var_a, var_b, var_c}
        can be constrained by means of an internal relation expressed 
        as
         
        var_a = f(beta), var_b = g(beta), var_c = h(beta)

        In this example, {var_a, var_b, var_c} are the external 
        variables and beta is the internal variable.
 
        Args: 
	    label (string): 
                Name given to the constraint. 
            vars_in (list of strings): 
                The list of internal optimization variables.
            vars_out (list of strings): 
                The list of external optimization variables.
            cfunc (function):
                Upon application of the constraint, the function cfunc 
                receives as argument a dictionary whose keys and values 
                are the labels and actual values of the "internal"
                optimization variables. And should return a dictionary
                whose keys and values correspond to the "external"
                optimization variables.
        """

        if not hasattr(self,'_constraints'):
            self._constraints = {}

        #print '....before constraint :', self._internal_variables
        self._constraints[label] = cfunc
        for var in vars_out:
            self._internal_variables.remove(var)
        for var in vars_in:
            self._internal_variables.append(var)        
        #print '....after constraint :', self._internal_variables
         
    def init_optim_variables(self): 
   
        return [0.]*len(self._internal_variables)
 
    def optim_variables(self, values): 

        internal_dict = dict((key,values[i]) for i, key in enumerate(self._internal_variables))
        external_dict = deepcopy(internal_dict)        

        if hasattr(self,'_constraints'):
            for label in self._constraints:
                external_dict = self._constraints[label](**external_dict)

        return external_dict

    def add_to_history(self, coeffs, *args):

        if not hasattr(self,'optim_history'):
            self.optim_history = []

        step = self.optim_variables(coeffs)
        hist = [step]+[args]
        self.optim_history.append(hist) 

    def add_to_backup(self, filename, coeffs, *args):

        io = open(filename,"aw")
        step = self.optim_variables(coeffs)
        hist = [step]+[args]
        pickle.dump(hist,io)
        io.close()

    def penalty(self, values, regularized=[], keep_history=True, backup_file=None, verbose=False, decompose=False):

        step_fwd = self.optim_variables(values)
        step_bwd = dict((key,-step_fwd[key]) for key in step_fwd)

        pen = 0.
        dpen = []
        for label in self:
            ref = self[label]['ref']
            optim = self[label]['optim']
            plabel = self[label]['penalty']
            pfunc = self.penalty_functions[plabel]
            args = self[label]['args']
            kwargs = self[label]['kwargs']

            optim.update_optim_variables(**step_fwd)
            delta = pfunc(optim, ref, *args, **kwargs) 
            dpen.append(delta)
            pen += delta
            optim.update_optim_variables(**step_bwd) 

            if verbose:
                print ('{:20} : {:.6E}     ({:.6E})'.format(label, delta, pen))

        if backup_file != None:
            self.add_to_backup(backup_file, values, pen)

        if keep_history:
            self.add_to_history(values, pen)

        if len(regularized) > 0:

            if regularized[0] > 0.:
                pen += regularized[0]*np.linalg.norm(values,ord=1)

            if len(regularized) > 1: 
                if regularized[1] > 0.:
                    pen += regularized[1]*np.linalg.norm(values)**2

        if decompose:
            return pen, dpen
        else:
            return pen 
