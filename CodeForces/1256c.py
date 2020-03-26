# -*- coding: utf-8 -*-

"""
・自力AC！バチャ終了5分前くらいで通した！
・実装重め、結構みんな通してなかったので嬉しい。これでBも通せてれば。。
"""
 
import sys
from itertools import accumulate
 
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M, d = MAP()
A = LIST()

# とりあえずなるべく遠くまで行ってみる
cur = 0
A2 = []
for a in A:
    cur += d
    A2.append((cur, cur+a))
    cur += a - 1
cur += d
A2.append((cur, cur+1))

# それでも届かなければNG
if cur < N+1:
    NO()
    exit()

B = [0] * (M+1)
for i in range(M, -1, -1):
    l, r = A2[i]
    # 余ってしまう分、間を潰していく
    if cur - (d - 1) >= N+1:
        cur -= d - 1
        B[i] = d - 1
    else:
        # 最後は帳尻が合う量だけ縮める
        dist = cur - (N+1)
        cur -= dist
        B[i] = dist
        break
# 前でずらした分は後ろの位置にも影響するので、累積和を取っておく
B = list(accumulate(B))

ans = [0] * N
for i, (l, r) in enumerate(A2[:-1]):
    # 必要な分、区間の位置をずらす
    l -= B[i]
    r -= B[i]
    for j in range(l-1, r-1):
        ans[j] = i + 1
YES()
print(*ans)
