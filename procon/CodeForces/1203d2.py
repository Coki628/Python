# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/69108
・自力ならず。
・解説、やっぱり何言ってるかよく分かんなくて、コード動かして理解。
・何をやったかはソース内コメント参照。
・こういう、前からと後ろからで累積取って、i番目まで前でi+1番目から後ろ使って
　それのi=0~Nまでで一番いいところを取る、みたいな流れは制約10万でよくあると思う。
・この辺も典型の範疇に入れて解けるようにしたいねー。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

# 文字列Tのi文字目がSで最初に出現するインデックスjを見つける
def check(S, T):
    N = len(S)
    M = len(T)
    res = [N] * M
    i = j = 0
    # 両方のindexを並行で進める
    while i < M:
        while j < N and T[i] != S[j]:
            j += 1
        res[i] = j
        j += 1
        i += 1
    return res

S = [ord(s)-97 for s in input()]
T = [ord(t)-97 for t in input()]
N = len(S)
M = len(T)

# res1[i] := 文字列Tのi文字目がSで一番左に出現するインデックス
res1 = check(S, T)
# res2[i] := 文字列Tのi文字目がSで一番右に出現するインデックス
res2 = check(S[::-1], T[::-1])
res2 = [N-a-1 for a in res2[::-1]]
# それぞれ番兵を入れて位置調整
res1.insert(0, -1)
res2.append(N)

ans = 0
for i in range(M+1):
    # i文字目より前を左に寄せてi文字目以降を右に寄せた時の間の文字数、から最大を取る
    ans = max(ans, res2[i] - res1[i] - 1)
print(ans)
