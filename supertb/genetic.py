"""
This module defines the classes and functions that enables the
optimization of tight-binding parameters by means of simple
genetic algorythms (or more accurately evolution algorithms).
"""

import numpy as np
import numpy.random as npr
import bottleneck 
from copy import deepcopy
import scipy.optimize as opt

class Individual():

    def __init__(self, genotype):

       self.genotype = np.array(genotype)
       self.fitness = None

    @classmethod
    def biological_crossover(cls, *parents):
        
        nparents = len(parents)
        ng = len(parents[0].genotype)
        biocross = np.random.randint(0,nparents,size=ng)
        genotype = np.array([parents[biocross[i]].genotype[i] for i in range(ng)])
        return cls(genotype)

    @classmethod
    def blending_crossover(cls, *parents):

        nparents = len(parents)
        blend = npr.sample(nparents)
        blend = blend/np.sum(blend)
        genotype = np.sum(np.array([blend[i]*parents[i].genotype for i in range(nparents)]),axis=0) 
        return cls(genotype)

    @classmethod
    def averaging_crossover(cls, *parents):

        nparents = len(parents)
        genotype = np.average(np.array([parents[i].genotype for i in range(nparents)]),axis=0)
        return cls(genotype)

    def evaluate_fitness(self,fitness_function):

        self.fitness = fitness_function(*self.genotype)
        return self.fitness

    @classmethod
    def normal_generation(cls, means, spreads):

        genotype = np.zeros(means.shape)
        for idx, mean in np.ndenumerate(means):
            spread = spreads[idx]
            genotype[idx] = np.random.normal(loc=mean,scale=spread)
        return cls(genotype)

    @classmethod
    def uniform_generation(cls, means, spreads):

        genotype = np.zeros(means.shape)
        for idx, mean in np.ndenumerate(means):
            a = mean - spreads[idx]/2.
            b = mean + spreads[idx]/2.
            genotype[idx] = (b - a) * np.random.random_sample() + a
        return cls(genotype)

    def normal_mutation(self, prob, spreads):

        for idx, spread in np.ndenumerate(spreads):
            if np.random.random_sample() <= prob:
                self.genotype[idx] += np.random.normal(loc=0.,scale=spread)

    def uniform_mutation(self, prob, spreads):

        for idx, spread in np.ndenumerate(spreads):
            if np.random.random_sample() <= prob:
                self.genotype[idx] += spread * np.random.random_sample() - spread/2.

    def __str__(self):
       
        return self.genotype.__str__()

    def __eq__(self, other):
    	if isinstance(other, Individual):
        	return (self.genotype == other.genotype).all()
    	return False


    def optimize(self, fitness_function, prob, method='BFGS', tol=0.0000001, options={'maxiter':8,'eps':0.0000001}):
        
        optflag = False
        delta = np.zeros(self.genotype.shape)
        for idx in range(len(self.genotype)):
            if np.random.random_sample() <= prob:
                delta[idx] = 1.
                optflag = True
               
        if optflag:

            def penalty(args):
                return 1./fitness_function(*args)
            res = opt.minimize(fun,coeffs_init,method=method,tol=tol,options=options) 
            self.genotype = res.x

class Population(list):

    @classmethod
    def normal_generation(cls, size, means, spreads):

        population = cls([])
        for idx in range(size):
            individual = Individual.normal_generation(means, spreads)
            population.append(individual)

        return population

    @classmethod
    def uniform_generation(cls, size, means, spreads):

        population = cls([])
        for idx in range(size):
            individual = Individual.uniform_generation(means, spreads)
            population.append(individual)

        return population

    
    def remove(self, b):
        population = Population()
        for individual in self:
            if individual not in b:
            	population.append(individual)
        return deepcopy(population)

    @classmethod
    def join(cls, a, b):
        population = cls([])
        for individual in a:
            population.append(individual)
        for individual in b:
            population.append(individual)

        return deepcopy(population)

    def compute_fitness(self, fitness_function):

        #print '...compute fitness...'

        fit_vector = []
        for idx, individual in enumerate(self):
            fitness = individual.evaluate_fitness(fitness_function)
            fit_vector.append(fitness)
            #print idx, individual.genotype, fitness
        self._fit_vector = np.array(fit_vector)  

    def report_on_best(self, nbest, fitness_function=None):

        if fitness_function != None:
            self.compute_fitness(fitness_function)

        elite = bottleneck.argpartition(-self._fit_vector, nbest)[:nbest]   
        report = {}
        for el in elite:
             report[el] = self[el]

        return report

    def elite_selection(self, nselect, fitness_function=None):
        
        if fitness_function != None:
            self.compute_fitness(fitness_function)

        elite = bottleneck.argpartition(-self._fit_vector, nselect)[:nselect]   
        population = Population()
        for el in elite:
            population.append(self[el])
 
        return deepcopy(population)

    def fitness_proportionate_selection(self, nselect, fitness_function=None):

        if fitness_function != None:
            self.compute_fitness(fitness_function)
        population = Population()
        max = sum([c.fitness for c in self])
        selection_probs = [c.fitness/max for c in self]
        for i in npr.choice(len(self),size=nselect,p=selection_probs,replace=False):
            population.append(self[i])

        return deepcopy(population)



#    def fitness_proportionate_selection(self, nselect, fitness_function=None):
#        
#        if fitness_function != None:
#            self.compute_fitness(fitness_function)
#    
#        J, q = alias_setup(self._fit_vector)
#
#        population = Population() 
#        for nn in xrange(N):
#            population.append(self[alias_draw(J, q)])
#
#        return deepcopy(population)
#
#    def stochastic_universal_sampling(self, nselect, fitness_function=None):
#
#        if fitness_function != None:
#            self.compute_fitness(fitness_function)
#    
#        l = np.sum(self._fit_vector)/nselect
#        root = l*np.random.random_sample() 
#      
#        population = Population() 
#        j = 0
#        for idx, csum in enumerate(np.cumsum(self._fit_vector)):
#            #print idx, csum
#            if csum >= root + j*l :
#                population.append(self[idx])
#                j += 1  
#
#        return deepcopy(population)

    def sampling(self, nsampling, nelite, method='proportionate', fitness_function=None):

        if nelite > 0 :
            pop1 = self.elite_selection(nelite, fitness_function=fitness_function)
        else:
            pop1 = Population()

        if nsampling > 0:
            #if method == 'stochastic':
            #    pop2 = self.stochastic_universal_sampling(nsampling, fitness_function=fitness_function)
            if method == 'proportionate':
                poptmp = self.remove(pop1)
                pop2 = poptmp.fitness_proportionate_selection(nsampling, fitness_function=fitness_function)
            else:
                pop2 = Population()

            return deepcopy(Population.join(pop1, pop2))

        return deepcopy(pop1)

    #def tournament_selection(self):

    #def truncation_selection(self):

    def blending_crossover(self, nparents, nchildrens):

        population = Population() 

        size = len(self)-1
        for ichild in range(nchildrens):
            parents = [self[i] for i in npr.random_integers(0,size,nparents)]
            population.append(Individual.blending_crossover(*parents)) 
    
        return deepcopy(population)

    def averaging_crossover(self, nparents, nchildrens):

        population = Population() 

        size = len(self)-1
        for ichild in range(nchildrens):
            parents = [self[i] for i in npr.random_integers(0,size,nparents)]
            population.append(Individual.averaging_crossover(*parents)) 
    
        return deepcopy(population)

    def biological_crossover(self, nparents, nchildrens):

        population = Population() 

        size = len(self)-1
        for ichild in range(nchildrens):
            parents = [self[i] for i in npr.random_integers(0,size,nparents)]
            population.append(Individual.biological_crossover(*parents)) 
    
        return deepcopy(population)

    def crossover(self, nchildrens, nparents, method='blending'):

        if method == 'blending':
            return self.blending_crossover(nparents, nchildrens)
        elif method == 'averaging':
            return self.averaging_crossover(nparents, nchildrens)
        elif method == 'biological':
            return self.biological_crossover(nparents, nchildrens)
        else:
            return Population()

    def mutate(self, prob, spreads, method='normal'):

        if method == 'normal':
            for idx in range(len(self)):
                self[idx].normal_mutation(prob,spreads)
        elif method == 'uniform':
            for idx in range(len(self)):
                self[idx].uniform_mutation(prob,spreads)

    def optimize(self, fitness_func, prob,  dmax, method='BFGS', tol=0.0000001, options={'maxiter':8,'eps':0.0000001}):

        for idx in range(len(self)):
            self[idx].optimize(fitness_func, prob,  dmax, method='BFGS', tol=0.0000001, options={'maxiter':8,'eps':0.0000001})

class GeneticAlgo(list):

    @classmethod
    def normal_generation(cls, size, means, spreads):
        
        populations = cls([])
        init = Population.normal_generation(size, means, spreads)
        populations.append(init)
        
        return populations

    @classmethod
    def uniform_generation(cls, size, means, spreads):
        
        populations = cls([])
        init = Population.uniform_generation(size, means, spreads)
        populations.append(init)
        
        return populations

    @property
    def fitness(self):
        return self._fitness
 
    @fitness.setter
    def fitness(self, value):
        self._fitness = value
 
    @property
    def sampling(self):
        if not hasattr(self,'_sampling'):
            return 'stochastic'
        else:
            return self._sampling
 
    @sampling.setter
    def sampling(self, value):
        self._sampling = value
 
    @property
    def crossover(self):
        if not hasattr(self,'_crossover'):
            return 'blending'
        else:
            return self._crossover
 
    @crossover.setter
    def crossover(self, value):
        self._crossover = value
 
    @property
    def mutation(self):
        if not hasattr(self,'_mutation'):
            return 'normal'
        else:
            return self._mutation
 
    @mutation.setter
    def mutation(self, value):
        self._mutation = value

    def propagate(self, ngeneration, nsampling, nelite, prob, spreads, \
                  superelite=0, order=2, report=False, oprob=0.1, omethod=None, otol=0.000001, oopt={'maxiter':8,'eps':0.0000001}):

        # Evaluate fitness
        size = len(self[-1])
        self[-1].compute_fitness(self.fitness)
        #print self[-1]._fit_vector

        # Report 
        if report:
            report = self[-1].report_on_best(5)
            print ('-- Initial population :')
            for idx in report:
                print ("{:8d}".format(idx),  "{:8.3e}".format(report[idx].fitness),  ["{:+8.2e}".format(i) for i in report[idx].genotype])
            

        for igen in range(ngeneration):

            # Sample
            parents = self[-1].sampling(nsampling, nelite, method=self.sampling)
 
            # Crossover
            children = parents.crossover(size-superelite, order, method=self.crossover)
     
            # Mutation
            children.mutate(prob, spreads, method=self.mutation)

            # Optimization
            if omethod != None :
                children.optimize(fitness_function, oprob, method=omethod, tol=otol, options=oopt)

            #children.compute_fitness(self.fitness)
            #if report:
            #    print 'Children :', np.amax(children._fit_vector), len(children)

            # Elite selection
            if superelite > 0:
                elite = self[-1].elite_selection(superelite) 
                #elite.compute_fitness(self.fitness)
                #if report:
                #    print 'Superelite :', np.amax(elite._fit_vector), len(elite)

                new_gen = Population.join(children, elite)
                new_gen.compute_fitness(self.fitness)
                #if report:
                #    print 'Total :', np.amax(new_gen._fit_vector), len(new_gen)
                self.append(new_gen)
            else:
                self.append(children)

            # Evaluate fitness
            self[-1].compute_fitness(self.fitness)
            #if report:
            #    print '-- Generation ', len(self), len(self[-1]), ' -- ', np.amax(self[-1]._fit_vector)

            # Report 
            if report:
                report = self[-1].report_on_best(5)
                print ('-- Generation ', len(self), len(self[-1]))
                #print  'max =', np.amax(self[-1]._fit_vector)
                for idx in report:
                    print ("{:8d}".format(idx),  "{:8.3e}".format(report[idx].fitness),  ["{:+8.2e}".format(i) for i in report[idx].genotype])

    def report_on_bests(self, igen, nbest):

        return self[igen].report_on_best(nbest)


def steepest_descent(func, x0, delta):

    f0 = func(x0)
    f1 = func(x0-delta)
    f2 = func(x0+delta)
    
    df = (f2-f1)/(2.*delta)
    ddf = (f2 - 2.*f0 + f1)/(delta**2)

    return -df/ddf


## Alias Method from Devroye (1986)
## Implementation from https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

def alias_setup(probs):
    K       = len(probs)
    q       = np.zeros(K)
    J       = np.zeros(K, dtype=np.int)
 
    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger  = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
 
    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
 
        J[small] = large
        q[large] = q[large] - (1.0 - q[small])
 
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
 
    return J, q
 
def alias_draw(J, q):
    K  = len(J)
 
    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand()*K))
 
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]
