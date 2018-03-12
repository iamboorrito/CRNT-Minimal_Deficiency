'''
Created on Mar 4, 2018

@author: evan
'''

from crnpy.crn import CRN, from_react_strings
from pulp import *
from itertools import product
import numpy as np
from pulp.solvers import CPLEX

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

    col_range = range(1, m+1)
    row_range = range(1, n+1)
    part_range = range(1, m-s+1)

    prob = LpProblem("Minimal Deficiency", LpMinimize)
    
    # Get a list of all off-diagonal entries to use later to greatly
    # simplify loops
    off_diag = [(i, j) for i,j in product(col_range, repeat=2) if i != j]
    
    # Decision variables for matrix A, only need off-diagonal since A
    # has zero-sum columns
    A = LpVariable.dicts("A", [(i, j) for i,j in product(col_range, repeat=2)])
    
    # Decision variables for the diagonal of T
    T = LpVariable.dicts("T", row_range, eps, 1/eps)
    
    # Kernel variables
    Phi = LpVariable.dicts("Phi", off_diag)
    
    # Binary variables
    theta = LpVariable.dicts("theta", part_range, 0, 1, "Integer")
    gamma = LpVariable.dicts("gamma", [(i,j) for i in col_range for j in part_range], 0, 1, "Integer")
    
    # Objective function: minimize -sum theta[k] for k = 1..partitions
    prob += m-s-lpSum( theta[k] for k in part_range )
    
    ########### Constraints ###########
    
    # Y*A = T*M
    for i in range(n):
        for j in range(m):
            prob += lpSum( Y[i, k]*A[k+1, j+1] for k in range(m) ) == T[i+1]*M[i, j]
    

    # A has zero-sum columns
    for j in col_range:
        prob += lpSum( A[i,j] for i in row_range ) == 0
    
    # Off-diagonal constraints on A
    for (i,j) in off_diag:
        prob += A[i, j] >= 0
        prob += A[i, j] <= 1/eps
    
    # Diagonal entries must be nonpositive
    for i in col_range:
        prob += A[i, i] <= 0 
    
    # Partitioning
    for i in col_range:
        prob += lpSum((gamma[i, k] for k in part_range)) == 1
        
        # Uniqueness constraint
        for k in part_range:
            if k <= i:
                prob += lpSum( gamma[j, k] for j in range(1, i) ) >= lpSum( gamma[i, L] for L in range(k+1, m-s+1) )
        
    for k in part_range:
        prob += lpSum( gamma[i, k] for i in col_range ) >= eps*theta[k]
        prob += lpSum( gamma[i, k] for i in col_range ) <= 1/eps*theta[k]
        
    # Kernel constraints
    for (i,j) in off_diag:
        prob += lpSum(( Phi[i, L] - Phi[L, i] for L in range(1, i) if L != i)) == 0
    
        for k in part_range:
            prob += eps*Phi[i,j] <= (gamma[i, k] - gamma[j, k] + 1)
            prob += Phi[i, j] >= eps*A[i, j]
            prob += Phi[i, j] <= 1/eps*A[i, j]
           
    status = prob.solve(solver=CPLEX()) 

    #print(prob)

    print(status)
    
    # Get solutions to problem
    gammasol = np.ndarray((m, m-s))
    Tsol = np.zeros((n, n))
    Asol = np.ndarray((m,m))
    
    for i in col_range:
        for j in row_range:
            Asol[i-1, j-1] = value(A[i, j])
        for k in part_range:
            gammasol[i-1, k-1] = value(gamma[i, k])
    
    for i in row_range:
        Tsol[i-1, i-1] = value(T[i])
    
    print("A =", Asol)
    print("gamma = ", gammasol)
    print("T = ", Tsol)
    print("deficiency = ", value(prob.objective))

#crn = from_react_strings(["0 ->(1) A1", "A1 ->(1) 0", "A1 ->(1) 2 A1", "2 A1 ->(1) A1", "A1 ->(1) A2 + A3", "A2 + A3 ->(2) A1", "A2 + A3 ->(1) 2 A2", "2 A2 ->(4) A2 + A3", "X2 + X3 ->(1) 2 X3", "2 X3 ->(1) X2 + X3"])
crn = from_react_strings(["2 T100 (1)<->(1) 2 T001", "2 T100 (1)<->(1) 2 T010",
                          "2 T001 (1)<->(1) 2 T010", "T100 + T001 (1)<->(1) T100 + T010", "T100 + T001 (1)<->(1) T010 + T001", "T100 + T010 (1)<->(1) T010 + T001"])

min_deficiency(crn, 0.01)