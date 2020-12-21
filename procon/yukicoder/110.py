"""
・自力AC
・制約小さいのでDPで。
・でもこれ10万とかで出されたら貪欲詰めるしかないんだよなー。。
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

N1 = INT()
A = LIST()
N2 = INT()
B = LIST()

AB = []
MAXW = 0
for a in A:
    AB.append((a, 0))
    MAXW = max(MAXW, a)
for b in B:
    AB.append((b, 1))
    MAXW = max(MAXW, b)
AB.sort(reverse=1)
N = len(AB)

# dp*[i][j] := i個目のブロックまで見て、一番上の幅がj、最後に積んだ色が白or黒(*=0/1)の時に積める最大高さ
dp0 = list2d(N+1, MAXW+2, -INF)
dp1 = list2d(N+1, MAXW+2, -INF)
# 最初は1番長いブロックも置けるようにしておく
dp0[0][MAXW+1] = dp1[0][MAXW+1] = 0
for i, (w, c) in enumerate(AB):
    for j in range(MAXW+2):
        # このブロックを置かない
        dp0[i+1][j] = max(dp0[i+1][j], dp0[i][j])
        dp1[i+1][j] = max(dp1[i+1][j], dp1[i][j])
        # このブロックを置く
        if j > w:
            if c == 0:
                dp0[i+1][w] = max(dp0[i+1][w], dp1[i][j] + 1)
            else:
                dp1[i+1][w] = max(dp1[i+1][w], dp0[i][j] + 1)
ans = max(max(dp0[N]), max(dp1[N]))
print(ans)
