# -*- coding: utf-8 -*-

"""
・結構すんなり自力AC
・累積和を工夫して前計算
"""

import sys
from collections import defaultdict

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

N = INT()
S = '*' + input() + '*'

# 初期状態の数を確認
D = defaultdict(int)
for s in S[1:-1]:
    if s == 'J':
        D[s] += 1
    elif s == 'O':
        D[s] += D['J']
    else:
        D[s] += D['O']
ans = D['I']

# Iの前計算
cntj = [0] * (N+2)
for i in range(1, N+2):
    cntj[i] += cntj[i-1] + (1 if S[i] == 'J' else 0)
cntjo = [0] * (N+2)
for i in range(1, N+2):
    cntjo[i] += cntjo[i-1] + (1 if S[i] == 'O' else 0) * cntj[i]
# Jの前計算
cnti = [0] * (N+2)
for i in range(N, -1, -1):
    cnti[i] += cnti[i+1] + (1 if S[i] == 'I' else 0)
cntoi = [0] * (N+2)
for i in range(N, -1, -1):
    cntoi[i] += cntoi[i+1] + (1 if S[i] == 'O' else 0) * cnti[i]
# Oの前計算
cntji = 0
for i in range(1, N+2):
    cntji = max(cntji, cntj[i-1] * cnti[i])

# J,O,Iをそれぞれ置く場合の最大から最大を取る
ans += max(cntoi[0], cntji, cntjo[-1])
print(ans)
