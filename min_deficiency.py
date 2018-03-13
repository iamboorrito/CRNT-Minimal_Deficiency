'''
Created on Mar 4, 2018

Based on the paper 'Computing Weakly Reversible Linearly Conjugate 
Chemical Reaction Networks with Minimal Deficiency' by Matthew Johnston,
David Siegel, and Gabor Szederkenyi (2012) arXiv:1203.5140

@author: Evan Burton
'''

from crnpy.crn import from_react_strings
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
from itertools import product
import numpy as np
from pulp.solvers import CPLEX

# Want to compute the minimal deficiency possible over all linearly conjugate
# networks to crn that obey eps <= [A]_ij <= 1/eps.
def min_deficiency(crn, eps):
    
    m = crn.n_complexes
    n = crn.n_species
    # Y is n by m
    Y = np.array(crn.complex_matrix).astype(np.float64)
    Ak = np.array(crn.kinetic_matrix).astype(np.float64)
    M = Y.dot(Ak)
    
    # Convert the sympy matrix to numpy float64 matrix and compute rank
    s = np.linalg.matrix_rank(np.array(crn.stoich_matrix).astype(np.float64))

    print("CRN defined by\nY =\n", Y)
    print("\nM =\n", M)    
    print("\nAk =\n", Ak)

    col_range = range(1, m+1)
    row_range = range(1, n+1)
    part_range = range(1, m-s+1)

    prob = LpProblem("Minimal Deficiency", LpMinimize)
    
    # Get a list of all off-diagonal entries to use later to greatly
    # simplify loops
    off_diag = [(i, j) for i,j in product(col_range, repeat=2) if i != j]
    
    # Decision variables for matrix A, only need off-diagonal since A
    # has zero-sum columns
    A = LpVariable.dicts("A", [(i, j) for i in col_range for j in col_range])
    
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
    for i in row_range:
        for j in col_range:
            prob += lpSum( Y[i-1, k-1]*A[k, j] for k in col_range ) == M[i-1, j-1]*T[i]
    
    # A has zero-sum columns
    for j in col_range:
        prob += lpSum( A[i,j] for i in col_range ) == 0
    
    # Off-diagonal constraints on A
    for (i,j) in off_diag:
        prob += A[i, j] >= 0
        prob += A[i, j] <= 1/eps
    
    # Diagonal entries must be non-positive
    for i in col_range:
        prob += A[i, i] <= 0 
    
    # Partitioning
    for i in col_range:
        prob += lpSum( gamma[i, k] for k in part_range ) == 1
        
        # Uniqueness constraint
        for k in part_range:
            if k <= i:
                prob += lpSum( gamma[j, k] for j in range(1, i) ) >= lpSum( gamma[i, L] for L in range(k+1, m-s+1) )

    for k in part_range:
        prob += lpSum( gamma[i, k] for i in col_range ) >= eps*theta[k]
        prob += lpSum( gamma[i, k] for i in col_range ) <= 1/eps*theta[k]
        
    # Kernel constraints
    for (i,j) in off_diag:
        prob += lpSum( Phi[i, L] - Phi[L, i] for L in range(1, i)) == 0
        prob += lpSum( Phi[i, L] - Phi[L, i] for L in range(i+1, m+1)) == 0
    
        for k in part_range:
            prob += Phi[i,j] <= (1/eps*gamma[i, k] - 1/eps*gamma[j, k] + 1/eps)
            prob += Phi[i, j] >= eps*A[i, j]
            prob += Phi[i, j] <= 1/eps*A[i, j]
           
    status = prob.solve(solver=CPLEX()) 

    #print(prob)

    if status == 1:
        # Get solutions to problem
        gammasol = np.ndarray((m, m-s))
        Tsol = np.zeros((n, n))
        Asol = np.zeros((m,m))
        
        for i in col_range:
            for j in col_range:
                Asol[i-1, j-1] = value(A[i, j])
            for k in part_range:
                gammasol[i-1, k-1] = value(gamma[i, k])
        
        for i in row_range:
            Tsol[i-1, i-1] = value(T[i])
        
        print("\nA =\n", Asol)
        print("\ngamma =\n", gammasol)
        print("\nT =\n", Tsol)
        print("\nmin deficiency = ", value(prob.objective))
    else:
        print("No solution found")


crn1 = from_react_strings(["0 ->(1) A1", "A1 ->(1) 0", "A1 ->(1) 2 A1", "2 A1 ->(1) A1", 
                          "A1 ->(1) A2 + A3", "A2 + A3 ->(2) A1", "A2 + A3 ->(1) 2 A2",
                           "2 A2 ->(4) A2 + A3", "X2 + X3 ->(1) 2 X3", "2 X3 ->(1) X2 + X3"])


crn2 = from_react_strings(["2 T100 (1)<->(1) 2 T001", "2 T100 (1)<->(1) 2 T010",
                          "2 T001 (1)<->(1) 2 T010", "T100 + T001 (1)<->(1) T100 + T010", "T100 + T001 (1)<->(1) T010 + T001", "T100 + T010 (1)<->(1) T010 + T001"])

crn3 = from_react_strings(["A (1)<->(8.5) 2 A", "A + B (1)<->(1) C", "C (1)<->(0.2) B"])

crn4 = from_react_strings(["3 A ->(0.1) A + 2 B", "A + 2 B ->(1) 3 B", "3 B ->(0.1) 2 A + B", "2 A + B ->(1) 3 A"])

min_deficiency(crn2, 0.1)

