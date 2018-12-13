# -*- coding: utf-8 -*-

"""
再帰
"""

N, X = map(int, input().split())

def dfs(cur, L):
    if L == N:
        return cur
    return dfs('B' + cur + 'P' + cur + 'B', L + 1)

burger = dfs('P', 0)

print(burger[:X].count('P'))
