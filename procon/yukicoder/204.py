"""
・自力AC
・ランレングス圧縮
・最初、連続させる位置を決め打ちして2重ループ全探索したんだけど、
　連続した1箇所の平日しか連休にできない制約を最初考慮してなくて作り直し。
　(普通にやると、週末を跨いだ平日も休みにできてしまう)
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

def RLE(data):
    """ ランレングス圧縮 """
    from itertools import groupby

    return [(x, len(list(grp))) for x, grp in groupby(data)]

D = INT()
N = 42
S = 'x'*14
S += input()
S += input()
S += 'x'*14

# 実装の楽のため前後に番兵入れとく
rle = [('o', 0)] + RLE(S) + [('o', 0)]
M = len(rle)
ans = 0
for i in range(1, M-1, 2):
    s0, ln0 = rle[i-1]
    s1, ln1 = rle[i]
    s2, ln2 = rle[i+1]
    if ln1 <= D:
        # この平日を全部休みにできるなら3区間繋ぐ
        ans = max(ans, ln0+ln1+ln2)
    else:
        # できないなら前後それぞれとDで繋ぐ
        ans = max(ans, ln0+D, D+ln2)
print(ans)
