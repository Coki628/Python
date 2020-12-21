# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc032
・セグ木は要らなかったけど、比較関数は役に立った。
・辞書順最小系、後ろからDP、前から貪欲
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def compress(A):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(set(A))):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N = INT()
AB = []
S = set()
for i in range(N):
    a, b = MAP()
    AB.append((a, b))
    S.add(a)
    S.add(b)

# 座標圧縮
zipped, _ = compress(S)
M = len(zipped)
for i, (a, b) in enumerate(AB):
    AB[i] = (zipped[a], zipped[b])

# まず仕事数、同数なら仕事番号の辞書順を比較する
def compare(a, b):
    if a[0] > b[0]:
        return a
    if a[0] < b[0]:
        return b
    if a[1] < b[1]:
        return a
    if a[1] > b[1]:
        return b
    return a

# DP更新のため、左端毎にまとめる
AB2 = [[] for i in range(M)]
for i, (a, b) in enumerate(AB):
    AB2[a].append((b, i))

# dp[i] := 時点iから達成できる(最大仕事数, 次に取る仕事の最小番号)
dp = [(0, INF) for i in range(M+1)]
for a in range(M-1, -1, -1):
    dp[a] = compare(dp[a], dp[a+1])
    for b, i in AB2[a]:
        cnt, _ = dp[b]
        dp[a] = compare(dp[a], (cnt+1, i))

ans = []
cnt, i = dp[0]
while i < INF:
    ans.append(i+1)
    a, b = AB[i]
    mn = INF
    # 右端以降で次の最適な仕事を探す
    for j in range(b, M):
        nxtcnt, nxti = dp[j]
        # 仕事数が現在-1より小さくなったら、それ以降は使えない
        if nxtcnt < cnt - 1:
            break
        # 最大仕事数な中で、仕事番号の最小を取る
        if nxti < mn:
            mn = nxti
    i = mn
    cnt -= 1
print(len(ans))
print(*ans)
