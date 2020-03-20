# -*- coding: utf-8 -*-

"""
・なんとか自力AC
・とはいえ重実装になり過ぎた。。
"""

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    S = input()
    N = len(S)
    L = []
    i = 0
    j = N - 1
    while i < j:
        if S[i] == S[j]:
            L.append(S[i])
            i += 1
            j -= 1
        else:
            break
    else:
        print(S)
        continue
    R = L[::-1]
    k = j
    mx = 0
    L2 = []
    while i < k:
        tmp = []
        i2 = i
        k2 = k
        while i2 < k2:
            if S[i2] == S[k2]:
                tmp.append(S[i2])
                i2 += 1
                k2 -= 1
            else:
                break
        else:
            tmp2 = tmp[::-1]
            if i2 == k2:
                tmp.append(S[i2])
            tmp += tmp2
            if len(tmp) > mx:
                mx = len(tmp)
                L2 = tmp
        k -= 1
    if not L2:
        L2.append(S[i])
    k = i
    mx = 0
    R2 = []
    while k < j:
        tmp = []
        k2 = k
        j2 = j
        while k2 < j2:
            if S[k2] == S[j2]:
                tmp.append(S[k2])
                k2 += 1
                j2 -= 1
            else:
                break
        else:
            tmp2 = tmp[::-1]
            if k2 == j2:
                tmp.append(S[j2])
            tmp += tmp2
            if len(tmp) > mx:
                mx = len(tmp)
                R2 = tmp
        k += 1
    if not R2:
        R2.append(S[j])
    if len(L2) > len(R2):
        ans = L + L2 + R
    else:
        ans = L + R2 + R
    print(''.join(ans))
