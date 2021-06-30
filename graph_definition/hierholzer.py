# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:36:11 2021

@author: andub
"""
import itertools

def hierholzer(g):
    """Find an Euler circuit on the given undirected graph if one exists.
    Args:
        g:  Undirected graph.
    Returns:
        List of vertices on the circuit or None if a circuit does not exist.
    # http://www.austincc.edu/powens/+Topics/HTML/06-1/ham2.gif
    # two triangles ABC and CDE
    >>> V = ['A', 'B', 'C', 'D', 'E']
    >>> E = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C','D'), ('C', 'E'), ('D', 'E')]
    >>> g = Graph(nodes=V, edges=E)
    >>> print(hierholzer(g))
    ['A', 'B', 'C', 'D', 'E', 'C', 'A']
    # V shape graph
    >>> print(hierholzer(Graph(nodes=[1,2,3], edges=[(1,2), (2,3)])))
    None
    """
    # Check if the graph has an Euler circuit: All vertices have even degrees.
    # for u in g:
    #     if len(list(g[u])) % 2 == 1:
    #         return None

    # Create necessary data structures.
    start = next(g.__iter__())  # choose the start vertex to be the first vertex in the graph
    circuit = [start]           # can use a linked list for better performance here
    traversed = {}
    ptr = 0
    while len(traversed) // 2 < g.number_of_edges() and ptr < len(circuit):
        subpath = []            # vertices on subpath
        __dfs_hierholzer(g, circuit[ptr], circuit[ptr], subpath, traversed)
        if len(subpath) != 0:   # insert subpath vertices into circuit
            circuit = list(itertools.chain(circuit[:ptr+1], subpath, circuit[ptr+1:]))
        ptr += 1

    return circuit

def __dfs_hierholzer(g, u, root, subpath, traversed):
    """Dfs on vertex u until get back to u. The argument vertices is a list and
    contains the vertices traversed. If all adjacent edges of starting vertex
    are already traversed, 'vertices' is empty after the call.
    """
    for v in g[u]:
        if (u,v) not in traversed or (v,u) not in traversed:
            traversed[(u,v)] = traversed[(v,u)] = True
            subpath.append(v)
            if v == root:
                return
            else:
                __dfs_hierholzer(g, v, root, subpath, traversed)