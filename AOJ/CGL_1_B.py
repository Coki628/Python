# -*- coding: utf-8 -*-

"""
・反射
"""

import sys

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

class Geometry:

    def add(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return (x1+x2, y1+y2)

    def sub(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return (x1-x2, y1-y2)

    def mul(self, a, b):
        x1, y1 = a
        if not isinstance(b, tuple):
            return (x1*b, y1*b)
        x2, y2 = b 
        return (x1*x2, y1*y2)

    def dot(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return x1*x2 + y1*y2

    def norm(self, a):
        x, y = a
        return x**2 + y**2

    def project(self, seg, p):
        """ 線分segに対する点pの射影 """
        p1, p2 = seg
        base = self.sub(p2, p1)
        r = self.dot(self.sub(p, p1), base) / self.norm(base)
        return self.add(p1, self.mul(base, r))

    def reflect(self, seg, p):
        """ 線分segを対称軸とした点pの線対称の点 """
        return self.add(p, self.mul(self.sub(self.project(seg, p), p), 2))

gm = Geometry()
x1, y1, x2, y2 = MAP()
seg = ((x1, y1), (x2, y2))
Q = INT()
for i in range(Q):
    x, y  = MAP()
    res = gm.reflect(seg, (x, y))
    print(*res)
