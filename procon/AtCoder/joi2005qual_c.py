# -*- coding: utf-8 -*-

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

d = [0] * 6
for i in range(6):
    d[i] = i + 1

def east():
    tmp = d[0]
    d[0] = d[3]
    d[3] = d[5]
    d[5] = d[2]
    d[2] = tmp

def left():
    tmp = d[1]
    d[1] = d[3]
    d[3] = d[4]
    d[4] = d[2]
    d[2] = tmp

def north():
    tmp = d[0]
    d[0] = d[1]
    d[1] = d[5]
    d[5] = d[4]
    d[4] = tmp

def right():
    tmp = d[1]
    d[1] = d[2]
    d[2] = d[4]
    d[4] = d[3]
    d[3] = tmp

def south():
    tmp = d[0]
    d[0] = d[4]
    d[4] = d[5]
    d[5] = d[1]
    d[1] = tmp

def west():
    tmp = d[0]
    d[0] = d[2]
    d[2] = d[5]
    d[5] = d[3]
    d[3] = tmp

ans = 1
for i in range(INT()):
    s = input()
    if s == 'East':
        east()
    elif s == 'Left':
        left()
    elif s == 'North':
        north()
    elif s == 'Right':
        right()
    elif s == 'South':
        south()
    else:
        west()
    ans += d[0]
print(ans)
