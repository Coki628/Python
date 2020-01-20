# -*- coding: utf-8 -*-

"""
参考：https://maspypy.com/atcoder-%E5%8F%82%E5%8A%A0%E6%84%9F%E6%83%B3-2020-01-18%E3%82%AD%E3%83%BC%E3%82%A8%E3%83%B3%E3%82%B9%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0%E3%82%B3%E3%83%B3%E3%83%86
・別解、裏表決め打ちビット全探索
・各カードについて裏表を決め打ちすると、偶奇位置が定まり、その中ではソート済位置への移動が自由にできる。
・元の1つの数列に戻した時、ソート済が維持されていれば、条件を満たすので、後は移動回数の確認。
・隣同士のスワップによる移動回数は転倒数数え上げに帰着。
・計算量はビット全探索*転倒数数え上げで2^N*N^2=約8500万だけど、条件を満たすケースしか数え上げのN^2しないからか大分速い。
　pypyAC0.8秒。
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

def pop_count(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

N = INT()
cards = [[]] * 2
cards[0] = LIST()
cards[1] = LIST()

ans = INF
for S in range(1<<N):
    # 無理なケース(奇数枚裏返す)を除外
    if pop_count(S) % 2 != 0:
        continue
    # 奇数位置と偶数位置で分けて保持する
    A1 = [[] for i in range(2)]
    for i in range(N):
        # カードiを青くする
        if S & 1<<i:
            A1[(i&1)^1].append((cards[1][i], i))
        # 赤のまま
        else:
            A1[i&1].append((cards[0][i], i))
    # 無理なケース(置きたい偶奇位置の数がおかしい)を除外
    if len(A1[0]) != ceil(N, 2) or len(A1[1]) != N // 2:
        continue
    # 偶奇が同じ位置同士は自由に動けるのでソートする
    A1[0].sort(reverse=1)
    A1[1].sort(reverse=1)
    cnt = 0
    A2 = [0] * N
    idxs = [0] * N
    for i in range(N):
        # 偶奇位置を交互に取り出す
        A2[i], idxs[i] = A1[i&1].pop()
    # ソート済と一致すればOK
    if A2 == sorted(A2):
        # 移動距離を転倒数数え上げで求める(Nが小さいのでN^2かけて大丈夫)
        cnt = 0
        for i, idx1 in enumerate(idxs):
            for j, idx2 in enumerate(idxs[i+1:], i+1):
                if idx1 > idx2:
                    cnt += 1
        ans = min(ans, cnt)
if ans == INF:
    print(-1)
else:
    print(ans)
