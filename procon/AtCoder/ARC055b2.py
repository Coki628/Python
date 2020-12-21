# -*- coding: utf-8 -*-

"""
参考：http://arc055.contest.atcoder.jp/data/arc/055/editorial.pdf
　　　http://mayokoex.hatenablog.com/entry/2016/06/04/102854
　　　http://baitop.hatenadiary.jp/entry/2018/09/11/015125
・確率系
・メモ化再帰
・添字的には、2次元DP+フラグの3次元
・今までで最大のものが出る確率は、実はNの大きさに関わらず1/i、
　これを戻り値に適用するので、1つ先に当たる1/(i+1)をかける。
・速度改善版。3次元メモを2次元2つに分けた。
・pythonTLE->1.8秒AC、pypy1.2->0.9秒ACと両方結構速くなった。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()

# memo1[i][j] := i枚目まで考えて、j枚食べている時、その時点で最大のものを取っていない状態での、全体で最大のものを取れる確率
# memo2[i][j] := i枚目まで考えて、j枚食べている時、その時点で最大のものを取っている状態での、全体で最大のものを取れる確率
memo1 = list2d(N+1, K+1, -1)
memo2 = list2d(N+1, K+1, -1)
def rec(i, j, is_max):
    if j > K:
        return 0
    # N枚終了時に、最大の物を取っているかどうか
    if i == N:
        return is_max
    if not is_max and memo1[i][j] != -1:
            return memo1[i][j]
    elif is_max and memo2[i][j] != -1:
            return memo2[i][j]

    res = 0
    # 次のiが最大値ではない時
    res += rec(i+1, j, is_max) * (i / (i+1))
    # 次のiが最大値の時
    res += max(
        # 取る
        rec(i+1, j+1, 1),
        # 取らない
        rec(i+1, j, 0),
    ) * (1 / (i+1))

    if not is_max:
        memo1[i][j] = res
    else:
        memo2[i][j] = res
    return res

print(rec(0, 0, 0))
