'''
Created on Mar 4, 2018

@author: evan
'''

from crnpy.crn import CRN, from_react_strings
from pulp import *
from itertools import product
import numpy as np

# Want to compute the minimal deficiency possible over all linearly conjugate
# networks to crn that obey eps <= [A]_ij <= ubound.
def min_deficiency(crn, eps):
    
    m = crn.n_complexes
    n = crn.n_species
    # Y is n by m
    Y = np.array(crn.complex_matrix).astype(np.float64)
    Ak = np.array(crn.kinetic_matrix).astype(np.float64)
    M = Y.dot(Ak)
    
    # Convert the sympy matrix to numpy float64 matrix and compute rank
    s = np.linalg.matrix_rank(np.array(crn.stoich_matrix).astype(np.float64))

    prob = LpProblem("Minimal Deficiency", LpMinimize)
    
    # Get a list of all off-diagonal entries to use later to greatly
    # simplify loops
    off_diag = [(i, j) for i,j in product(range(m), repeat=2) if i != j]
    
    # Decision variables for matrix A, only need off-diagonal since A
    # has zero-sum columns
    A = LpVariable.dicts("A", [(i, j) for i,j in product(range(m), repeat=2)])
    
    # Decision variables for the diagonal of T
    T = LpVariable.dicts("T", range(n), eps, 1/eps)
    
    # Kernel variables
    Phi = LpVariable.dicts("Phi", off_diag)
    
    # Binary variables
    theta = LpVariable.dicts("theta", range(m-s), 0, 1, "Integer")
    gamma = LpVariable.dicts("gamma", [(i,j) for i in range(m) for j in range(m-s)], 0, 1, "Integer")
    
    # Objective function: minimize -sum theta[k] for k = 1..partitions
    prob += lpSum( theta[k] for k in range(m-s) )
    
    ########### Constraints ###########
    
    prob += T[0] == 1
    
    # Y*A = T*M
    for i in range(n):
        for j in range(m):
            prob += lpSum(( Y[i, k]*A[k, j] for k in range(m))) == T[i]*M[i, j]
    

    # A has zero-sum columns
    for j in range(m):
        prob += lpSum((A[i,j] for i in range(n))) == 0
    
    # Off-diagonal constraints on A
    for (i,j) in off_diag:
        prob += A[i, j] >= 0
        prob += A[i, j] <= 1/eps
    
    # Diagonal entries must be nonpositive
    for i in range(m):
        prob += A[i, i] <= 0 
    
    # Partitioning
    for i in range(m):
        prob += lpSum((gamma[i, k] for k in range(m-s))) == 1
        
        # Uniqueness constraint
        for k in range(m-s):
            if k <= i:
                prob += lpSum( (gamma[j, k] for j in range(i-1)) ) >= lpSum( (gamma[i, L] for L in range(k+1, m-s)) )
        
    for k in range(m-s):
        prob += lpSum(( gamma[i, k] for i in range(m) )) >= eps*theta[k]
        prob += lpSum(( gamma[i, k] for i in range(m) )) <= 1/eps*theta[k]
        
    # Kernel constraints
    for (i,j) in off_diag:
        prob += lpSum(( Phi[i, L] - Phi[L, i] for L in range(i-1) if L != i)) == 0
    
        for k in range(m-s):
            prob += eps*Phi[i,j] <= (gamma[i, k] - gamma[j, k] + 1)
            prob += Phi[i, j] >= eps*A[i, j]
            prob += Phi[i, j] <= 1/eps*A[i, j]
           
    prob.solve() 

    # Get solutions to problem
    gammasol = np.ndarray((m, m-s))
    Tsol = np.zeros((n, n))
    Asol = np.ndarray((m,m))
    
    for i in range(m):
        for j in range(m):
            Asol[i, j] = value(A[i, j])
        for k in range(m-s):
            gammasol[i, k] = value(gamma[i, k])
    
    for i in range(n):
        Tsol[i, i] = value(T[i])
    
    print("A =", Asol)
    print("gamma = ", gammasol)
    print("T = ", Tsol)

crn = from_react_strings(["0 ->(1) A1", "A1 ->(1) 0", "A1 ->(1) 2 A1", "2 A1 ->(1) A1", "A1 ->(1) A2 + A3", "A2 + A3 ->(2) A1", "A2 + A3 ->(1) 2 A2", "2 A2 ->(4) A2 + A3", "X2 + X3 ->(1) 2 X3", "2 X3 ->(1) X2 + X3"])

min_deficiency(crn, 0.333)