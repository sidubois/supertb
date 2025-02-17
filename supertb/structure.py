import numpy as np
import pymatgen as pm
import itertools
from supertb import CoverTree
from copy import deepcopy
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial import cKDTree
import networkx as nx
import os

#from memory_profiler import profile

def super_vertices(n):

    nx, ny, nz = n
    return  np.array([[ix, iy, iz] for ix in range(nx+1) \
                                   for iy in range(ny+1) \
                                   for iz in range(nz+1)],dtype=int)

def super_vertices_centered(n):

    nx, ny, nz = n
    return  np.array([[ix, iy, iz] for ix in list(range(nx+1))+list(range(-nx,0)) \
                                   for iy in list(range(ny+1))+list(range(-ny,0)) \
                                   for iz in list(range(nz+1))+list(range(-nz,0))],dtype=int)

def cartesian_vertices():

    cv  = super_vertices_centered([1,1,1])
    cv = np.zeros((7,3))
    cv[0,:] = [ 0, 0, 0]
    cv[1,:] = [ 1, 0, 0]
    cv[2,:] = [-1, 0, 0]
    cv[3,:] = [ 0, 1, 0]
    cv[4,:] = [ 0,-1, 0]
    cv[5,:] = [ 0, 0, 1]
    cv[6,:] = [ 0, 0,-1]
    return cv

class Lattice(pm.core.Lattice):
    """
    A lattice object.  Essentially a matrix with conversion matrices.
    This class emulates the Lattice class from Pymatgen.
    """
    def __init__(self,*args,**kwargs):
        """
        Creates a lattice from any sequence of 9 numbers. Note that the sequence
        is assumed to be read one row at a time. Each row represents one
        lattice vector.

        Args:
            matrix: Sequence of numbers in any form. Examples of acceptable
                input.
                i) An actual numpy array.
                ii) [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                iii) [1, 0, 0 , 0, 1, 0, 0, 0, 1]
                iv) (1, 0, 0, 0, 1, 0, 0, 0, 1)
                Each row should correspond to a lattice vector.
                E.g., [[10, 0, 0], [20, 10, 0], [0, 0, 30]] specifies a lattice
                with lattice vectors [10, 0, 0], [20, 10, 0] and [0, 0, 30].
        """

        if isinstance(args[0],pm.core.Lattice):
            super(Lattice,self).__init__(args[0].matrix)
        else:
            super(Lattice,self).__init__(*args,**kwargs)

        self._cart_vertices = np.array([x for x in cartesian_vertices()],dtype=int)
        self._cart_vectors = np.array([np.dot(x,self.matrix) for x in self._cart_vertices])

    @classmethod
    def from_pymatgen(cls,pm_lattice):
        """
        Creates a Lattice object from a pymatgen lattice object         

        Args:
            pm_lattice: pymatgen lattice object
        """

        return cls(pm_lattice.matrix)

    @classmethod
    def from_parameters(cls,*args,**kwargs):
        """
        Creates a Lattice using unit cell lengths and angles (in degrees).

        Args:
            a (float): *a* lattice parameter.
            b (float): *b* lattice parameter.
            c (float): *c* lattice parameter.
            alpha (float): *alpha* angle in degrees.
            beta (float): *beta* angle in degrees.
            gamma (float): *gamma* angle in degrees.

        Returns:
            Lattice with the specified lattice parameters.
        """

        return cls.from_pymatgen(pm.core.Lattice.from_parameters(*args,**kwargs))

    @classmethod
    def from_lengths_and_angles(cls,*args,**kwargs):
        """
        Creates a Lattice using unit cell lengths and angles (in degrees).

        Args:
            abc (3x1 array): Lattice parameters, e.g. (4, 4, 5).
            ang (3x1 array): Lattice angles in degrees, e.g., (90,90,120).

        Returns:
            A Lattice with the specified lattice parameters.
        """

        return cls.from_pymatgen(pm.core.Lattice.from_lengths_and_angles(*args,**kwargs))

    def cdist(self, p):
        """
        Replicates the coordinate given as argument in neighboring cartesian cell 
        and identifies the replica associated with the smallest distance to the origin.

        Returns the indices of the selected replica and the computed distance.
        """

        dist = np.linalg.norm(p-self._cart_vectors, axis=-1)
        argmin = dist.argmin()

        return argmin, dist[argmin]

    def periodic_data(self, v, frac_coords=False):

        if frac_coords:
            vec = self.get_cartesian_coords(np.reshape(v,(3)))
        else:
            vec = np.reshape(v,(3))

        argmin = 1
        period = np.zeros(3)
        image = np.zeros(3,dtype=int)

        while argmin != 0:
            argmin, dist = self.cdist(vec)
            vertex = self._cart_vertices[argmin]
            dv = self._cart_vectors[argmin]
            image += vertex
            period += dv
            vec -= dv

        return dist, vec, image, period


class Structure(pm.core.Structure):
    
    def __init__(self, lattice, species, coords, validate_proximity=False,
                 coords_are_cartesian=False, site_properties=None):
        """
        Create a periodic structure.

        Args:
            lattice: The lattice, either as a pymatgen.core.lattice.Lattice or
                simply as any 2D array. Each row should correspond to a lattice
                vector. E.g., [[10,0,0], [20,10,0], [0,0,30]] specifies a
                lattice with lattice vectors [10,0,0], [20,10,0] and [0,0,30].
            species: List of species on each site. Can take in flexible input,
                including:

                i.  A sequence of element / specie specified either as string
                    symbols, e.g. ["Li", "Fe2+", "P", ...] or atomic numbers,
                    e.g., (3, 56, ...) or actual Element or Specie objects.

                ii. List of dict of elements/species and occupancies, e.g.,
                    [{"Fe" : 0.5, "Mn":0.5}, ...]. This allows the setup of
                    disordered structures.
            coords: list of fractional coordinates of each species.
            validate_proximity (bool): Whether to check if there are sites
                that are less than 0.01 Ang apart. Defaults to False.
            coords_are_cartesian (bool): Set to True if you are providing
                coordinates in cartesian coordinates. Defaults to False.
            site_properties (dict): Properties associated with the sites as a
                dict of sequences, e.g., {"magmom":[5,5,5,5]}. The sequences
                have to be the same length as the atomic species and
                fractional_coords. Defaults to None for no properties.
        """
        super(Structure,self).__init__(lattice, species, coords, \
            validate_proximity=validate_proximity, to_unit_cell=True,\
            coords_are_cartesian=coords_are_cartesian, \
            site_properties=site_properties)

class StructureGraph(nx.MultiDiGraph):
    """
    A StructureGraph object designed to perform tight-binding calculations.  
    This class is based on the MultiDiGraph class from Networkx, and extends
    the latter by adding methods to automatically create a Graph from a
    Structure, with appropriate nodes and edges attribute.  
    """

    def __init__(self, lattice, species, coords, rcut,\
                 coords_are_cartesian=False):

        ### Lattice
        if isinstance(lattice, Lattice):
            glattice = lattice
        else:
            glattice = Lattice(lattice)

        ### Fractional and cartesian coordinates
        if coords_are_cartesian:
            fracs = glattice.get_fractional_coords(coords)
            gfrac_coords = fracs
            #gfrac_coords = fracs%1.
        else:
            gfrac_coords = coords%1.
        gcart_coords = glattice.get_cartesian_coords(gfrac_coords)

        ### Species
        gspecies = species

        ### MultiDiGraph
        super(StructureGraph,self).__init__(lattice=glattice, species=gspecies, \
            cart_coords=gcart_coords, frac_coords=gfrac_coords)

        ### Nodes
        for i in range(len(gcart_coords)):
            self.add_node(i)

        ### Edges 
        if rcut > 0.:

            # Internal edges
            kdi = cKDTree(gcart_coords)
            for i, j in kdi.query_pairs(rcut):
                v = gcart_coords[j] - gcart_coords[i]
                self.add_edge(i, j, vector=deepcopy(v), order=1)
                self.add_edge(j, i, vector=-deepcopy(v), order=1)
                
            # External edges
            ortho = np.array([v/np.linalg.norm(v) for v in glattice.inv_matrix.T])*rcut
            delta = np.array([np.dot(ortho[i],glattice.inv_matrix)[i] for i in range(3)])
            nsc = np.ceil(delta).astype(int)
            self.graph['nsc'] = nsc
            vertices = super_vertices_centered(nsc)[1:]
            
            tmp_coords = []
            extra_sites = []
            for isite, ifrac in enumerate(gfrac_coords):
                p = ifrac + vertices
                for ix in range(3):
                    ptmp = p[np.logical_and(p[:,ix]>-delta[ix], p[:,ix]<1.+delta[ix])]   
                    p = ptmp
                for frac in p:
                    tmp_coords.append(frac)
                    extra_sites.append(isite)
            
            extra_cart_coords = np.dot(np.array(tmp_coords),glattice.matrix) 
            del tmp_coords
            
            kdo = cKDTree(extra_cart_coords)
            
            for isite, neighbors in enumerate(kdi.query_ball_tree(kdo,rcut)):
                icart = gcart_coords[isite]
                for jn in neighbors:
                    jcart = extra_cart_coords[jn]
                    jsite = extra_sites[jn]
                    v = jcart - icart
                    self.add_edge(isite, jsite, vector=deepcopy(v), order=1)
            
            #for i, j, k in self.edges_iter(keys=True):
            for i, j, k in self.edges(keys=True):
                self[i][j][k]['path'] = [(i,j,k)]


    def subcell(self, nodes, scalings, centered=False, rtol=1.e-6):

        # Sub lattice
        if centered:
            iscalings = np.array([1./(2*(x-1)+1) for x in scalings])
        else:
            iscalings = np.array([1./x for x in scalings])
        sub_mat = (self.lattice.matrix.T*iscalings).T

        # Coordinates and species
        nnodes = len(nodes)
        sub_cart_coords = np.empty((nnodes,3))
        sub_species = []
        mapping = {}
        for idx, inode in enumerate(nodes):
            sub_cart_coords[idx,:] = self.cart_coords[inode,:]
            sub_species.append(self.species[inode])
            mapping[inode] = idx
        
        # StructureGraph
        sub_graph = StructureGraph(sub_mat, sub_species, sub_cart_coords, 0.,\
                 coords_are_cartesian=True)
        sub_lattice = sub_graph.graph['lattice']


        # Edges preliminaries
        sc_mapping = {}
        #for i in self.nodes_iter():
        for i in self.nodes():
            icoords = self.cart_coords[i]
            for j in nodes:
                ijf = sub_lattice.get_fractional_coords(self.coords[j] - icoords) 
                if np.linalg.norm(sub_lattice.get_cartesian_coords(ijf-np.rint(ijf))) < rtol:
                    sc_mapping[i] = j
        
        # Edges
        #for i,j,k in self.edges_iter(keys=True,data=False):
        for i,j,k in self.edges(keys=True,data=False):

            if i not in nodes:
                continue

            ii = mapping[i]
            vec = self[i][j][k]['vector']
            order = self[i][j][k]['order']

            if j in nodes:
                jj = mapping[j]    
                sub_graph.add_edge(ii,jj,vector=deepcopy(vec), order=order)
            elif j in sc_mapping:
                jj = sc_mapping[j]
                sub_graph.add_edge(ii,jj,vector=deepcopy(vec), order=order)
                    
        #for i,j,k in sub_graph.edges_iter(keys=True):
        for i,j,k in sub_graph.edges(keys=True):
            vec = sub_graph[i][j][k]['vector']
            order = sub_graph[i][j][k]['order']
            path = sub_graph.underlying_path(i,j,vec,omax=1,itermax=order, rtol=rtol)
            sub_graph[i][j][k]['path'] = path

        return sub_graph


    def supercell_mapping(self, scalings, centered=False):

        nx, ny, nz = scalings
        if centered:
            sc_vertices = super_vertices_centered((nx-1,ny-1,nz-1))
        else:
            sc_vertices = super_vertices((nx-1,ny-1,nz-1))
        sc_num = len(sc_vertices)
        nsites = len(self.coords)
        
        mapping = {}
        #for inode in self.nodes_iter(data=False):
        for inode in self.nodes(data=False):
            mapping[inode] = [inode+isc*nsites for isc in range(sc_num)]

        return mapping


    def supercell(self, scalings, centered=False, rtol=1.e-6, periodic=True):
        """
        WARNING, this method only work if the supercell is a principal-cell 
        (i.e. interacts only with its nearest neighbor image)!
        """

        nx, ny, nz = scalings
        if centered:
            sc_vertices = super_vertices_centered((nx-1,ny-1,nz-1))
        else:
            sc_vertices = super_vertices((nx-1,ny-1,nz-1))
        sc_num = len(sc_vertices)
        nsites = len(self.coords)

        # Super lattice
        if centered:
            vscalings = np.array([(2*(x-1)+1) for x in scalings])
        else:
            vscalings = np.array([x for x in scalings])
        sc_mat = (self.lattice.matrix.T*vscalings).T

        # Coordinates        
        sc_cart_coords = np.empty((sc_num*nsites,3))
        for iv, vv in enumerate(sc_vertices):
            sc_cart_coords[iv*nsites:(iv+1)*nsites] = self.cart_coords+np.dot(vv,self.lattice.matrix)
        
        # Species
        sc_species = []
        for isc in range(sc_num):
            for spec in self.species:
                sc_species.append(spec)

        # StructureGraph
        sc_graph = StructureGraph(sc_mat, sc_species, sc_cart_coords, 0.,\
                 coords_are_cartesian=True)
        sc_lattice = sc_graph.graph['lattice']
        
        # Edges preliminaries
        rcut = 0.
        #for i,j,k in self.edges_iter(keys=True,data=False):
        for i,j,k in self.edges(keys=True,data=False):
            l = np.linalg.norm(self[i][j][k]['vector'])
            if l > rcut:
                rcut = l

        ortho = np.array([v/np.linalg.norm(v) for v in sc_lattice.inv_matrix.T])*rcut
        delta = np.array([np.dot(ortho[i],sc_lattice.inv_matrix)[i] for i in range(3)])
        nsc = np.ceil(delta).astype(int) 
        vertices = super_vertices_centered(nsc)[1:]
        #print vertices

        tmp_coords = []
        tmp_sites = []
        for isite, ifrac in enumerate(sc_graph.frac_coords):
            p = ifrac + vertices
            for ix in range(3):
                ptmp = p[np.logical_and(p[:,ix]>-delta[ix], p[:,ix]<1.+delta[ix])]
                p = ptmp
           
            for frac in p:
                tmp_coords.append(frac)
                tmp_sites.append(isite)
        extra_sites = np.array(tmp_sites)
        extra_cart_coords = np.dot(np.array(tmp_coords),sc_lattice.matrix)
        del tmp_coords
        del tmp_sites
        #print extra_cart_coords
        #print extra_sites

        # Edges
        #for i,j,k in self.edges_iter(keys=True,data=False):
        for i,j,k in self.edges(keys=True,data=False):
            vec = self[i][j][k]['vector']
            order = self[i][j][k]['order']
            #print "Start {: 8.6f} {: 8.6f} {: 8.6f}".format(*self.coords[i]), " vec {: 8.6f} {: 8.6f} {: 8.6f}".format(*vec)

            for isc in range(sc_num):
                ii = i+isc*nsites
                iicoords = sc_graph.coords[ii]
                #print '...starting from ', ii, " {: 8.6f} {: 8.6f} {: 8.6f}".format(*iicoords)
                ijcoords = iicoords + vec
                #print "......looking for site on {: 8.6f} {: 8.6f} {: 8.6f}".format(*ijcoords) 

                for jsc in range(sc_num):
                    jj = j+jsc*nsites   
                    jjcoords = sc_graph.coords[jj]
                    if np.linalg.norm(ijcoords-jjcoords) < rtol: 
                        sc_graph.add_edge(ii,jj,vector=deepcopy(vec), order=order) 

                    if periodic:
                        for iextra in np.where(extra_sites==jj)[0]:
                            jjcoords = extra_cart_coords[iextra]
                            if np.linalg.norm(ijcoords-jjcoords) < rtol:
                                sc_graph.add_edge(ii,jj,vector=deepcopy(vec), order=order) 

        #for i,j,k in sc_graph.edges_iter(keys=True):
        for i,j,k in sc_graph.edges(keys=True):
            vec = sc_graph[i][j][k]['vector']
            order = sc_graph[i][j][k]['order']
            path = sc_graph.underlying_path(i,j,vec,omax=1,itermax=order, rtol=rtol)
            sc_graph[i][j][k]['path'] = path

        return sc_graph

    @property
    def lattice(self):
        return self.graph['lattice']

    @property
    def coords(self):
        return self.graph['cart_coords']

    @property
    def cart_coords(self):
        return self.graph['cart_coords']

    @property
    def frac_coords(self):
        return self.graph['frac_coords']

    @property
    def species(self):
        return self.graph['species']

    def nodes_of_species_iter(self, species):
        for inode, spec in enumerate(self.species):
            if spec == species:
                yield inode

    def is_an_edge(self, inode, jnode, vec, rtol=1.e-6):

        found = False
        duplicata = []
        if inode in self:
           if jnode in self[inode]: 
               for k in self[inode][jnode]:
                   v = self[inode][jnode][k]['vector']
                   if np.linalg.norm(vec-v) < rtol:
                       duplicata.append((inode,jnode,k))                   

        return duplicata


    def underlying_path(self, inode, jnode, vec, omax=1, itermax=8, rtol=1.e-6, except_species=[]):

        found = False
        iiter = 0

        total_vec = vec
        total_path = []
        tmp_node = inode
        tmp_vec = np.zeros(3)

        #print 'underlying path :', inode, jnode, vec
        while (not found and iiter < itermax):

            vecs = []
            edges = []   
            for j in self[tmp_node]:
                #print 'test ', j
                if self.graph['species'][j] in except_species:
                    next
                for m in self[tmp_node][j]:
                    if 'order' in self[tmp_node][j][m]:
                        if self[tmp_node][j][m]['order'] <= omax:
                            vecs.append(self[tmp_node][j][m]['vector'])
                            edges.append((tmp_node,j,m))
            idx = np.linalg.norm(np.array(vecs)+tmp_vec-total_vec, axis=-1).argmin()
            total_path.append(edges[idx])
           
            if np.linalg.norm(vecs[idx]+tmp_vec-total_vec) < rtol:
                found = True

            tmp_vec += vecs[idx]
            tmp_node = edges[idx][1]
            iiter += 1
            
        if found:
            return total_path
        else:
            return []
        


    def empty_graph(self):

        G = nx.MultiDiGraph()

        #for node in self.nodes_iter():
        for node in self.nodes():
            G.add_node(node)

        #for i, j, k in self.edges_iter(keys=True):
        for i, j, k in self.edges(keys=True):
            G.add_edge(i,j)

        return G

    def remove_nodes_by_species(self, species):

        new_graph = deepcopy(self)
        new_cart_coords = []
        new_frac_coords = []
        new_species = []

        i=0
        j=0
        mapping = {}
        for spec, cart, frac in itertools.izip(self.graph['species'],\
                                               self.graph['cart_coords'],\
                                               self.graph['frac_coords']):
            if spec not in species:
                new_cart_coords.append(cart)
                new_frac_coords.append(frac)
                new_species.append(spec)

                mapping[i] = j
                j+=1
            else:
                new_graph.remove_node(i)    

            i+=1


        new_graph = nx.relabel_nodes(new_graph,mapping,copy=False)
        new_graph.graph['species'] = new_species
        new_graph.graph['cart_coords'] = new_cart_coords
        new_graph.graph['frac_coords'] = new_frac_coords

        return new_graph, mapping

    @classmethod
    def init_from_structure(cls, struct, rcut):
        """
        Creates a StructureGraph from a Structure object.

        Args:
            struct: 
                Structure object. The nodes of the created StructureGraph
                are associated with the atomic sites of the structure.  
            rcut: 
                Cutoff-radius used to determine the graph connectivity.
                An edge is associated with each air of atomic coordinates 
                whose distance is smaller than rcut.
        """	

        lattice = struct.lattice
        frac_coords = struct.frac_coords
        species = [x.symbol for x in struct.species]
        return StructureGraph(lattice, species, frac_coords, rcut)


    def edges_olen(self, minstep=0.1, tol=0.00001, **kwargs):

        lengths = []
        for i,j,k,data in self.edges(keys=True, data=True):
            
            if 'species' in kwargs and \
                (self.graph['species'][i] != kwargs['species'] or \
                 self.graph['species'][j] != kwargs['species']):
                 continue

            n = np.linalg.norm(np.array(data['vector']))
            lengths.append(n)

        lsort = np.sort(lengths)
        ilsort = np.argsort(lengths)

        #print lsort       
 
        lstep = [0.]
        for newl in lsort:
            if newl > lstep[-1]+minstep:
                lstep.append(newl+tol)
        #print lstep 
        
        for i,j,k,data in self.edges(keys=True, data=True):
            n = np.linalg.norm(np.array(data['vector']))

            if 'species' in kwargs and \
                (self.graph['species'][i] != kwargs['species'] or \
                 self.graph['species'][j] != kwargs['species']):
                 continue

            #order = np.argmin(lstep<n) 
            self[i][j][k]['order'] = np.argmin(lstep<n)
            #print i,j,k, n, self[i][j][k]['order']
            


    def color_edges(self, omax=8):

        if not hasattr(self,'clock_loops') or \
           not hasattr(self,'anti_clock_loops'):
            self.simple_2D_loops(omax=omax)
  
        #for edge in self.edges_iter(keys=True):
        for edge in self.edges(keys=True):

            color = [0,0]
            for loop in self.clock_loops:
                if edge in loop:
                    color[0] = len(loop)

            for loop in self.anti_clock_loops:
                if edge in loop:
                    color[1] = -len(loop)

            i, j ,k = edge
            self[i][j][k]['colors'] = color

    def color_nodes(self, clockwise=True, omax=8):

        if clockwise and not hasattr(self,'clock_loops'):
            self.simple_2D_loops(omax=omax)
        elif not clockwise and not hasattr(self,'anti_clock_loops'):
            self.simple_2D_loops(omax=omax)
  
        #for node in self.nodes_iter(data=False):
        for node in self.nodes(data=False):

            color = []
            if clockwise:
                for loop in self.clock_loops:
                    for i,j,k in loop:
                        if i == node:
                            color.append(len(loop))
                            break
            else:
                for loop in self.anti_clock_loops:
                    for i, j, k in loop:
                        if i == node:
                            color.append(-len(loop))
                            break

            self.nodes[node]['colors'] = color

    def simple_2D_loops(self, omax=8):

        self.clock_loops = []
        self.anti_clock_loops = []
        #for i,j,k in self.edges_iter(keys=True):
        for i,j,k in self.edges(keys=True):

            edge = (i,j,k)

            found = False
            for loop in self.clock_loops:
                if edge in loop:
                    found = True
                    break
            if not found:
                loop = self.clockwise_loop(edge, omax=omax)
                self.clock_loops.append(loop)
             
            found = False
            for loop in self.anti_clock_loops:
                if edge in loop:
                    found = True
                    break
            if not found:
                loop = self.clockwise_loop(edge, omax=omax, anti=True)
                self.anti_clock_loops.append(loop)

    def clockwise_loop(self, edge, omax=8, anti=False):

        start_node, inode, iedge = edge
        vec = deepcopy(self[start_node][inode][iedge]['vector'])
        nvec = vec/np.linalg.norm(vec)
        vectot = deepcopy(vec)

        loop = [edge]
        for order in range(omax):
            best_dot_product = 1.
            found = False
            for knode in self[inode]:
                for kedge in self[inode][knode]:
                    vec_trial = self[inode][knode][kedge]['vector']
                    nvec_trial = vec_trial/np.linalg.norm(vec_trial)

                    cross_product = np.cross(nvec, nvec_trial)[2]
                    dot_product = np.dot(nvec, nvec_trial)

                    if (cross_product < -0.00001 and not anti) or \
                       (cross_product > 0.00001 and anti):
                        if dot_product < best_dot_product:
                            best_edge = (inode, knode, kedge)
                            best_nvec = nvec_trial
                            best_vec = vec_trial
                            best_node = knode
                            best_dot_product = dot_product
                            found = True

            if found :
                nvec = best_nvec
                inode = best_node
                vectot += best_vec 
                loop.append(best_edge)

            if inode == start_node:
                if np.linalg.norm(vectot) < 0.01:
                    break     

        return loop

    def increase_graph_order(self,order, colors = False):

        G = deepcopy(self)
        for k in range(order):
            K = G.add_neighbors_to_graph(colors = colors)
            G = deepcopy(K)            

        return G

    def add_neighbors_to_graph(self, colors = False):
        
        G = deepcopy(self)
        for inode in self.nodes():
            coords = self.graph['cart_coords'][inode]

            for jnode in self.successors(inode):
                for jedge in self[inode][jnode]:
                    jv = self[inode][jnode][jedge]['vector']
                    jp = self[inode][jnode][jedge]['path'] 

                    for knode in self.successors(jnode):
                        for kedge in self[jnode][knode]:

                            if self[jnode][knode][kedge]['order'] > 1:
                                continue

                            kv = self[jnode][knode][kedge]['vector']
                            newv = jv + kv                            

                            accept = True
                            if knode in G[inode]:
                                for ledge in G[inode][knode]:
                                    lv = G[inode][knode][ledge]['vector']
                                    if np.linalg.norm(lv-newv) < 0.01:
                                        accept = False
                                        break
                            if np.linalg.norm(newv) < 0.01:
                                accept = False

                            if accept:
                                newp = deepcopy(jp)
                                newp.append((jnode,knode,kedge))

                                neworder = len(newp)
                                length = np.linalg.norm(newv)

                                G.add_edge(inode, knode, vector=newv, \
                                          order=neworder, path=newp)

        if colors:
            G.propagate_edge_colors()

        return G

    def propagate_edge_colors(self):

        #for i, j, k, data in self.edges_iter(keys=True, data=True):
        for i, j, k, data in self.edges(keys=True, data=True):
            if 'colors' not in self[i][j][k]:
                path = self[i][j][k]['path'] 

                color = []
                for loop in self.clock_loops:
                    inloop = True

                    try:
                        iedge = loop.index(path[0])
                    except:
                        inloop = False                    

                    if inloop:
                        for idx, edge in enumerate(path[1:]):
                            try:
                                jedge = loop.index(edge)
                            except:
                                inloop = False
                                break
                            if jedge != (iedge+idx+1)%len(loop):
                                inloop = False
                                break

                    if inloop:
                        color.append(len(loop))


                for loop in self.anti_clock_loops:
                    inloop = True

                    try:
                        iedge = loop.index(path[0])
                    except:
                        inloop = False                    

                    if inloop:
                        for idx, edge in enumerate(path[1:]):
                            try:
                                jedge = loop.index(edge)
                            except:
                                inloop = False
                                break
                            if jedge != (iedge+idx+1)%len(loop):
                                inloop = False
                                break

                    if inloop:
                        color.append(-len(loop))

                self[i][j][k]['colors'] = color

    def add_edges_to_graph(self, species, rmin, rmax):

        # Creates a temporary array containings all potential external coordinates 
        lattice = self.graph['lattice']
        ortho = np.array([v/np.linalg.norm(v) for v in lattice.inv_matrix.T])*rmax
        delta = np.array([np.dot(ortho[i],lattice.inv_matrix)[i] for i in range(3)])
        nsc = np.ceil(delta).astype(int)
        vertices = close_vertices(nsc)[1:]

        tmp_intra_coords = []
        tmp_extra_coords = []
        intra_sites = []
        extra_sites = []

        #for isite in self.nodes_iter(data=False):
        for isite in self.nodes(data=False):
            ifrac = self.graph['frac_coords'][isite]
            ispec = self.graph['species'][isite]

            if ispec in species:
                intra_sites.append(isite)
                tmp_intra_coords.append(ifrac)
                
                p = ifrac + vertices
                for ix in range(3):
                    ptmp = p[np.logical_and(p[:,i]>-delta[i], p[:,i]<1.+delta[i])]   
                    p = ptmp
                for frac in p:
                    tmp_extra_coords.append(frac)
                    extra_sites.append(isite)
        
        intra_cart_coords = np.dot(np.array(tmp_intra_coords),lattice.matrix) 
        extra_cart_coords = np.dot(np.array(tmp_extra_coords),lattice.matrix) 
        del tmp_intra_coords          
        del tmp_extra_coords          

        # Create kdtree with all the internal coordinates
        kdi = cKDTree(intra_cart_coords)
        kdo = cKDTree(extra_cart_coords)

        # Add internal edges to graph
        for i, j in kdi.query_pairs(rmax):

            isite = intra_sites[i]
            jsite = intra_sites[j]
            ispec = self.graph['species'][isite]
            jspec = self.graph['species'][jsite]

            if (ispec == species[0] and jspec == species[1]) or \
               (ispec == species[1] and jspec == species[0]) :
 
                icart = intra_cart_coords[i]
                jcart = intra_cart_coords[j]
                vec = jcart - icart
                length = np.linalg.norm(vec)

                if length >= rmin:
                    self.add_edge(isite, jsite, vector=deepcopy(vec), order=1)
                    self.add_edge(jsite, isite, vector=-deepcopy(vec), order=1)

        # Add external edges to graph
        for i, neighbors in enumerate(kdi.query_ball_tree(kdo,rmax)):

            isite = intra_sites[i]
            icart = intra_cart_coords[i]
            ispec = self.graph['species'][isite]

            if ispec in species:
            
                for jn in neighbors:
              
                    jsite = extra_sites[jn] 
                    jspec = self.graph['species'][jsite]

                    if (ispec == species[0] and jspec == species[1]) or \
                       (ispec == species[1] and jspec == species[0]) :

                        jcart = extra_cart_coords[jn]
                        vec = jcart - icart
                        length = np.linalg.norm(vec)
                        
                        if length >= rmin:
                            self.add_edge(isite, jsite, vector=vec, order=1)
        
        #for i, j, k in self.edges_iter(keys=True):
        for i, j, k in self.edges(keys=True):
            if 'path' not in self[i][j][k]:
                self[i][j][k]['path'] = [(i,j,k)] 

def show_vesta(s,data=None,atoms=None,supercell=[0,1,0,1,0,1],bonds=None):
    
    current_dir = os.getcwd()
    os.system("rm -r ./.tmp_vesta")
    os.system("mkdir .tmp_vesta")
    os.chdir(current_dir+"/.tmp_vesta")
    print (os.getcwd())
    afile = 'struct.vesta'

    # Vesta directory
    vesta_dir = '/Applications/VESTA.app/Contents/Resources/'
    
    # Create a vesta file 
    f = open(afile, 'a')

    # Header
    f.write('#VESTA_FORMAT_VERSION 3.3.0\n\n')
    f.write('CRYSTAL\n\n')
    f.write('TITLE\n')
    f.write('Structure generated from SuperTB\n\n')
    
    # Density
    if data != None:
        f.write('IMPORT_DENSITY 1\n')
        f.write('+1.000000 '+current_dir+'/'+data+'\n\n')
    
    # Cell
    f.write('CELLP \n')
    f.write('%10.6f %10.6f %10.6f' % tuple(s.lattice.abc))
    f.write('%10.6f %10.6f %10.6f\n' % tuple(s.lattice.angles))
    f.write('%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' % tuple([0.,0.,0.,0.,0.,0.,]))
    
    # Structure
    f.write('\nSTRUC \n')
    snum = {x:s.species.count(x) for x in s.species}
    scount = {x:0 for x in s.species}
    for isite, site in enumerate(s.sites):
        f.write('{:>3d}'.format(isite+1))# (site.specie, site.frac_coords)
        f.write(' {:>2s}  '.format(site.specie))
        scount[site.specie] += 1
        sitename = '{:s}{:d}'.format(site.specie, scount[site.specie])
        f.write(' {:>6s}'.format(sitename))
        f.write(' {:6.4f}'.format(1.))
        f.write(' {: 10.6f} {: 10.6f} {: 10.6f}  '.format(*site.frac_coords))
        f.write(' {:s}\n'.format(' 1a      1'))
        f.write('                       {: 10.6f} {: 10.6f} {: 10.6f} {: 4.2f} \n'.format(0.,0.,0.,0.))
    f.write('  0 0 0 0 0 0 0\n')
                
    if supercell!=None:
        f.write('BOUND\n')
        f.write('    {:>6f} {:>6f} {:>6f} {:>6f} {:>6f} {:>6f}\n'.format(*supercell))
        f.write(' 0  0  0  0  0\n\n')
    
    if bonds!=None:
        f.write('SBOND\n')
        for ibond, bond in enumerate(bonds):
            f.write('{:>3d}'.format(ibond+1))
            f.write('     {:>2s}  {:>2s}   {:6.4f}  {:6.4f}  '.format(*bond))
            f.write('0  0  1  0  1  0.250  1.000 180 180 180\n')
            f.write('  0 0 0 0\n\n')

    
    if atoms!=None:
        f.write('ATOMT\n')
        for iatom, atom in enumerate(atoms):
            f.write('{:>3d}'.format(iatom+1))
            f.write('     {:>2s}'.format(atom['specie']))
            f.write(' {:6.4f}'.format(atom['radius']))
            f.write(' {:>3d} {:>3d} {:>3d}'.format(*[int(x*255) for x in colors.to_rgb(atom['color'])]))
            f.write(' {:>3d} {:>3d} {:>3d}'.format(*[int(x*255) for x in colors.to_rgb(atom['color'])]))
            f.write(' 204  0\n')
            
    # Style
    fstyle = open(vesta_dir+'style.ini', 'r')
    f.write('\nSTYLE \n')
    for il,l in enumerate(fstyle):
        if il > 11:
            f.write(l)
    fstyle.close()
    
    f.close()

    os.system("/Applications/VESTA.app/Contents/MacOS/VESTA "+afile)

    # Delete the tmp directory
    os.chdir(current_dir)
    os.system("rm -r ./.tmp_vesta")
