from math import exp, log, sin, cos, pi, sqrt, trunc, ceil
import numpy as np

def factors(n):
    factors = []
    for x in range(1, int(sqrt(n)) + 1):
        if n % x == 0:
            factors.append([x,n//x])
    return factors

def ratio_factor(n,r):
    fac = factors(n)
    res = float("inf")
    if r <= 1.:
        for a,b in fac:
            ares = abs(float(a)/float(b)-r)
            if ares < res:
                res = ares
                out = [a,b]
    else:
        for a,b in fac:
            ares = abs(float(b)/float(a)-r)
            if ares < res:
                res = ares
                out = [b,a]
       
    return out

def largest_factor(n):
    factors = []
    for x in range(int(sqrt(n)) + 1,0,-1):
        if n % x == 0:
            factors.append(x)
            factors.append(n//x)
            break
    return sorted(factors)

