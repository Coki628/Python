"""
・なんとか自力AC。バグり散らかしてひどかった。コンテストなら絶対間に合ってない。。
・座圧、いもす、座圧した分の重み付けをして累積和
・なんとなく書いたら盛大に合わないので、cf1029cを参考に調整。
・まず右端は、半開区間的に見た時のものがあればいいので、r+1だけ使う。
・あとは添字ズレ地獄を丁寧に丁寧にデバッグして修正。ひどいムーブだった。。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N, K = MAP()
S = set()
AB = []
for _ in range(N):
    a, b = MAP()
    S.add(a)
    S.add(b+1)
    AB.append((a, b))

S.add(0)
zipped, unzipped = compress(S)
M = len(zipped)
imos = [0] * M
for i in range(N):
    a, b = AB[i]
    imos[zipped[a]] += 1
    imos[zipped[b+1]] -= 1
imos = list(accumulate(imos))

# 座圧した後の各値に重み付けする
W = [0] * M
for i in range(M-1):
    W[i] = unzipped[i+1] - unzipped[i]

sm = 0
ans = -1
for i in range(1, M):
    # 次に増える予定の枚数
    nxt = imos[i] * W[i]
    if sm + nxt >= K:
        need = K - sm
        # 直前までの値 + 今回必要な分を足して増える値
        ans = unzipped[i]-1 + ceil(need, imos[i])
        break
    sm += nxt
print(ans)
