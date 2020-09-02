"""
・自力AC！
・うまいことシミュレーションする。
・最初、10万の内側でsetのマージとかやってしまってTLE。
　Counterで人毎に最大値をいくつ持っているかを管理したら無事AC。
　内側ループにTLE感があるけど、実際は各能力で各人は1回最大値から外れたらもうそこに加わることがないので、
　ならし計算量で内側ループの分は無視できる。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, M = MAP()
A = [[] for i in range(N)]
for i in range(N):
    A[i] = LIST()

# mx[j] := 能力jの現在の最大値
mx = [0] * M
# se[j] := 能力jについて最大値を持っている人の集合
se = [set() for i in range(M)]
# C[i] := 人iが最大値を持っている数
C = Counter()
for i in range(N):
    for j in range(M):
        # 能力jの最大値更新
        if A[i][j] > mx[j]:
            mx[j] = A[i][j]
            # 能力jの最大値保持者たちから1ずつ減らす
            for a in se[j]:
                C[a] -= 1
                # 最大値が1つもなくなった人は削除
                if C[a] == 0:
                    del C[a]
            se[j] = set([i])
            C[i] += 1
        # 能力jの最大値保持者を追加
        elif A[i][j] == mx[j]:
            se[j].add(i)
            C[i] += 1
    # dictの長さを見るだけなのでここに計算量はかからない
    print(len(C))
