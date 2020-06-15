"""
・自力AC
・エラトステネスっぽく倍数の情報を更新すれば線形*logで前計算が済む。
・これだけだと同値が2つ以上ある場合をカバーできないのでそこは別処理。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7

N = INT()
A = LIST()
MAX = max(A)

# 同値で何回もやらないようにsetにしておく
S = set(A)
ok = [1] * (MAX+1)
for a in S:
    # エラトステネスっぽく倍数の情報を更新
    for x in range(a+a, MAX+1, a):
        ok[x] = 0

# 同値が2つ以上ある場合もNGにする
C = Counter(A)
for k, v in C.items():
    if v >= 2:
        ok[k] = 0

# 全部チェックが済んだらあとは見るだけ
ans = 0
for a in A:
    if ok[a]:
        ans += 1
print(ans)
