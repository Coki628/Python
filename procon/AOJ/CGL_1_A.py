# -*- coding: utf-8 -*-

"""
・射影
"""

import sys

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

def dot(a, b):
    x1, y1 = a
    x2, y2 = b
    return x1*x2 + y1*y2

def norm(a):
    x, y = a
    return x**2 + y**2

def add(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x1+x2, y1+y2)

def sub(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x1-x2, y1-y2)

def mul(a, b):
    x1, y1 = a
    if not isinstance(b, tuple):
        return (x1*b, y1*b)
    x2, y2 = b 
    return (x1*x2, y1*y2)

# 線分segに対する点pの射影
def project(seg, p):
    p1, p2 = seg
    base = sub(p2, p1)
    r = dot(sub(p, p1), base) / norm(base)
    return add(p1, mul(base, r))

x1, y1, x2, y2 = MAP()
seg = ((x1, y1), (x2, y2))
Q = INT()
for i in range(Q):
    x, y  = MAP()
    res = project(seg, (x, y))
    print(*res)
