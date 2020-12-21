# -*- coding: utf-8 -*-

"""
・木の復元
"""

import sys
sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))

N = INT()
preo = LIST()
ino = LIST()
posto = []

def rec(l):
    if len(l) == 0:
        return
    # preorderから次の値を取り出す
    x = preo.pop(0)
    for i in range(len(l)):
        if l[i] == x:
            # preorderの値と一致したら、そこから左右に分割
            rec(l[:i])
            rec(l[i+1:])
    posto.append(x)
    return

# inorderを再帰的に分割していく
rec(ino)
print(*posto)
