# -*- coding: utf-8 -*-

"""
参考：http://mayokoex.hatenablog.com/entry/2016/03/30/001735
　　　https://qiita.com/drken/items/a207e5ae3ea2cf17f4bd
・蟻本演習2-3-15
・部分文字列DP、復元、辞書順系
・復元パートしんどい。これ自分で導けるようになるだろうか。。
・pythonTLE、pypyTLE、文字列100万*26つらい。
・数が10**18+1以下で収まるようにしたらpypyTLE取れたけどMLE。
・C++でAC確認、0.2秒。
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
INF = float('inf')
MOD = 10 ** 9 + 7

S = [ord(s)-97 for s in input()]
K = INT()
N = len(S)

# nxt[i][c] := i文字目以降で最初に文字cが登場するindex(存在しないときはN)
nxt = list2d(N+1, 26, N)
for i in range(N-1, -1, -1):
    for c in range(26):
        nxt[i][c] = nxt[i+1][c]
    nxt[i][S[i]] = i

# dp[i] := i文字目以降について部分文字列となる通り数
dp = [1] * (N+1)
for i in range(N-1, -1, -1):
    for c in range(26):
        # 次の文字がない時
        if nxt[i][c] == N:
            continue
        # 次の文字がある時
        else:
            # 次回出現位置の次のindexから貰う
            dp[i] += dp[nxt[i][c]+1]
            # 数が大きくなりすぎないようにする(K+1以下が分かれば問題ない)
            dp[i] = min(dp[i], K+1)

# 全通り-空文字列 < K なら不可
if dp[0] - 1 < K:
    print('Eel')
    exit()

# 復元
curi = curk = 0
ans = []
# ぴったりK通りに来るまで文字を追加していく
while curk != K:
    for c in range(26):
        # 文字cが次に出現する位置
        i = nxt[curi][c]
        # この文字はもうない
        if i == N:
            continue
        # 文字cから始まる部分文字列の通り数k(先頭はcで確定なので、+1した位置の通り数)
        k = dp[i+1]
        # K以上になるまでは該当位置のkを足していく
        if curk + k < K:
            curk += k
            continue
        else:
            ans.append(chr(c+97))
            # 文字cが増えた分1通り追加
            curk += 1
            # indexは今回のc出現位置の次に
            curi = i + 1
            break
print(''.join(ans))
