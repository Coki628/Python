# -*- coding: utf-8 -*-

import sys
from math import hypot

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

x1,y1,r=MAP()
x2,y2,x3,y3=MAP()

def get_dist(v1, v2):
    return hypot(v1[0]-v2[0], v1[1]-v2[1])

def check_red():
    # 円周の最大最小xyの位置が四角形より内側にある
    if x1-r>=x2 \
            and x1+r<=x3 \
            and y1-r>=y2 \
            and y1+r<=y3:
        return False
    else:
        return True

def check_blue():
    # 四角形の4隅が円の中心から見て円周より近くにある
    if get_dist((x1, y1), (x2, y2))<=r \
            and get_dist((x1, y1), (x2, y3))<=r \
            and get_dist((x1, y1), (x3, y2))<=r \
            and get_dist((x1, y1), (x3, y3))<=r:
        return False
    else:
        return True

YES() if check_red() else NO()
YES() if check_blue() else NO()
