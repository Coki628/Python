"""
・自力AC
・そこが確定している場合、K個先も同じものを使わないといけない。
　まずはこのことに従って?を埋める。
・それがクリアできたら、後は好きに?を使えるので、長さKの区間について
　0か1が過半数になってしまう場所がないかチェック。あればNG。
　なければ?を使ってうまく調節できるはずなのでOKとする。
"""

import sys
from collections import Counter

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
INF = 10**19
MOD = 10**9 + 7

def check(C, K):
    if C['0'] > K//2 or C['1'] > K//2:
        return False
    else:
        return True

for _ in range(INT()):
    N, K = MAP()
    S = list(input())

    ok = True
    for i in range(N-K):
        if S[i+K] == '?':
            S[i+K] = S[i]
        elif S[i] != '?' and S[i] != S[i+K]:
            ok = False
            break
    if not ok:
        NO()
        continue

    C = Counter()
    for i in range(K):
        C[S[i]] += 1
    if not check(C, K):
        NO()
        continue
    for i in range(K, N):
        C[S[i-K]] -= 1
        C[S[i]] += 1
        if not check(C, K):
            ok = False
            break
    if ok:
        YES()
    else:
        NO()
