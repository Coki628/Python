"""
・自力AC
・高さiのピラミッドを作るコスト、を可能な範囲で全探索する。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()
A = LIST()

sm = sum(A)
ans = INF
i = 1
while 1:
    # 高さiのピラミッドの完成形を構築
    B1 = []
    for j in range(1, i+1):
        B1.append(j)
    B2 = []
    for j in range(i-1, 0, -1):
        B2.append(j)
    B = B1 + B2
    M = len(B)
    total = sum(B)
    # 数が足りなければこの先はやる必要ない
    if total > sm:
        break
    # 現在の状態との差分を求めていく
    cnt1 = cnt2 = 0
    for j in range(max(N, M)):
        if j >= N:
            cnt1 += B[j]
        elif j >= M:
            cnt2 += A[j]
        else:
            if A[j] < B[j]:
                cnt1 += B[j] - A[j]
            elif A[j] > B[j]:
                cnt2 += A[j] - B[j]
    # 交換回数
    mv = min(cnt1, cnt2)
    # 削除回数
    rm = abs(cnt1-cnt2)
    ans = min(ans, mv+rm)
    i += 1
print(ans)
