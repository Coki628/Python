# -*- coding: utf-8 -*-

"""
・自力AC！
・これは考察もスムーズに正しい方針に進んでいい感じだった。
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

N, L = MAP()
XD = []
for i in range(N):
    x, d = input().split()
    x = int(x)
    XD.append((x, d))

# 前後に番兵を配置
if XD[0][1] == 'L':
    XD.insert(0, (0, 'R'))
if XD[-1][1] == 'R':
    XD.append((L+1, 'L'))

# 同じ所に収束するもの毎にまとめる
curd = 'R'
L = []
R = []
tmp = []
for x, d in XD:
    if curd == d:
        tmp.append(x)
    else:
        if curd =='L':
            L.append(tmp)
        else:
            R.append(tmp)
        curd = d
        tmp = [x]
L.append(tmp)

M = len(L)
ans = 0
for i in range(M):
    # 左右とも先頭(末尾)まで詰める
    lcnt = 0
    for j, l in enumerate(L[i]):
        # 自分 - 先頭 - 前にいる数
        lcnt += l - L[i][0] - j
    rcnt = 0
    for j, r in enumerate(R[i][::-1]):
        # 末尾 - 自分 - 後ろにいる数
        rcnt += R[i][-1] - r - j
    # 右向き末尾と左向き先頭の間の空間
    space = L[i][0] - R[i][-1] - 1
    # 数が多い方を動かす
    move = space * max(len(L[i]), len(R[i]))
    # 動いた分全部を答えに追加
    ans += lcnt + rcnt + move
print(ans)
