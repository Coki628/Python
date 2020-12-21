# -*- coding: utf-8 -*-

"""
・終了後500点(ほぼ)自力AC。
・ツイートでちらっと順番関係ないみたいな話は見てしまっていた。けど、それは自分でも予測はできたはず。
・同じ数字はまとめてから、1つになるまで半分にしていく。
・半分にした余りの処理と、残った最後の処理は愚直シミュレーションする。
・公式解の発想はなかなか出づらそう。。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

# 同じ数字2つをマージした時の行き先と増える操作回数
# (nxt, cnt)
D = {
    0: (0, 1),
    1: (2, 1),
    2: (4, 1),
    3: (6, 1),
    4: (8, 1),
    5: (1, 2),
    6: (3, 2),
    7: (5, 2),
    8: (7, 2),
    9: (9, 2),
}

M = INT()
C = Counter()
for i in range(M):
    d, c = MAP()
    # 同じ数字の桁は全て一纏めにする
    C[d] += c

# 数字aがcnt個あった時に、全てマージした後の数字と費やした操作回数を返す
def calc(a, cnt):
    sm = 0
    odd = []
    # 1つになるまでやる
    while cnt > 1:
        # 奇数だと半分にした時1つ余るので後で別処理するように分けておく
        if cnt % 2 == 1:
            odd.append(a)
            cnt -= 1
        cnt //= 2
        sm += cnt * D[a][1]
        a = D[a][0]
    # 奇数の時に分けておいた分を愚直シミュレーション
    for b in odd:
        a += b
        sm += 1
        if a >= 10:
            a = int(str(a)[0]) + int(str(a)[1])
            sm += 1
    return (a, sm)

ans = 0
A = []
for k, v in C.items():
    # 数字毎にマージ、これで残り最大10桁まで減るはず
    a, sm = calc(k, v)
    ans += sm
    A.append(a)

# 残りを愚直シミュレーション
sm = A[0]
for a in A[1:]:
    sm += a
    ans += 1
    if sm >= 10:
        sm = int(str(sm)[0]) + int(str(sm)[1])
        ans += 1
print(ans)
