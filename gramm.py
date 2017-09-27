"""
File   gramm.py
Author Ernesto P. Adorio, Ph.D.
	   U.P. Clarkfield, Pampanga
Version 0.0.1  2009.09.28 first version.
"""
from math import sqrt

def gramm(X,inplace = False):
    # Returns the Gramm-Schmidt orthogonalization of matrix X
    if not inplace:
        V = [row[:] for row in X]  # make a copy.
    else:
        V = X
    k = len(X[0])		  # number of columns.
    n = len(X)			 # number of rows.
    for j in range(k):
        for i in range(j):
            D = sum([V[p][i]*V[p][j] for p in range(n)])
            for p in range(n):
                # Note that the Vi's already have length one!
                # Vj = Vj - <Vi,Vj> Vi/< Vi,Vi >
                V[p][j] -= (D * V[p][i])
        # Normalize column V[j]
        invnorm = 1.0 / sqrt(sum([(V[p][j])**2 for p in range(n)]))
        for p in range(n):
            V[p][j] *= invnorm
    return V
