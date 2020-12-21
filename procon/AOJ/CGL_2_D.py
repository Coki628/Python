# -*- coding: utf-8 -*-

"""
・線分と線分の距離
"""

import sys

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

class Geometry:

    EPS = 10 ** -9

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

    def abs(self, a):
        from math import hypot
        x1, y1 = a
        return hypot(x1, y1)

    def norm(self, a):
        x, y = a
        return x**2 + y**2

    def dot(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return x1*x2 + y1*y2

    def cross(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return x1*y2 - y1*x2

    def project(self, seg, p):
        """ 線分segに対する点pの射影 """
        p1, p2 = seg
        base = self.sub(p2, p1)
        r = self.dot(self.sub(p, p1), base) / self.norm(base)
        return self.add(p1, self.mul(base, r))

    def reflect(self, seg, p):
        """ 線分segを対称軸とした点pの線対称の点 """
        return self.add(p, self.mul(self.sub(self.project(seg, p), p), 2))

    def ccw(self, p0, p1, p2):
        """ 線分p0,p1から線分p0,p2への回転方向 """
        a = self.sub(p1, p0)
        b = self.sub(p2, p0)
        # 反時計回り
        if self.cross(a, b) > self.EPS: return 1
        # 時計回り
        if self.cross(a, b) < -self.EPS: return -1
        # 直線上(p2 => p0 => p1)
        if self.dot(a, b) < -self.EPS: return 2
        # 直線上(p0 => p1 => p2)
        if self.norm(a) < self.norm(b): return -2
        # 直線上(p0 => p2 => p1)
        return 0

    def intersect(self, seg1, seg2):
        """ 線分seg1と線分seg2の交差判定 """
        p1, p2 = seg1
        p3, p4 = seg2
        return (
            self.ccw(p1, p2, p3) * self.ccw(p1, p2, p4) <= 0
            and self.ccw(p3, p4, p1) * self.ccw(p3, p4, p2) <= 0
        )

    def get_distance_LP(self, line, p):
        """ 直線lineと点pの距離 """
        p1, p2 = line
        return abs(self.cross(self.sub(p2, p1), self.sub(p, p1)) / self.abs(self.sub(p2, p1)))

    def get_distance_SP(self, seg, p):
        """ 線分segと点pの距離 """
        p1, p2 = seg
        if self.dot(self.sub(p2, p1), self.sub(p, p1)) < 0: return self.abs(self.sub(p, p1))
        if self.dot(self.sub(p1, p2), self.sub(p, p2)) < 0: return self.abs(self.sub(p, p2))
        return self.get_distance_LP(seg, p)

    def get_distance_SS(self, seg1, seg2):
        """ 線分seg1と線分seg2の距離 """
        p1, p2 = seg1
        p3, p4 = seg2
        if self.intersect(seg1, seg2): return 0
        return min(
            self.get_distance_SP(seg1, p3), self.get_distance_SP(seg1, p4),
            self.get_distance_SP(seg2, p1), self.get_distance_SP(seg2, p2),
        )

gm = Geometry()
for _ in range(INT()):
    x1, y1, x2, y2, x3, y3, x4, y4 = MAP()
    seg1 = ((x1, y1), (x2, y2))
    seg2 = ((x3, y3), (x4, y4))
    print(gm.get_distance_SS(seg1, seg2))
