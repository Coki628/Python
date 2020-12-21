# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-4
・反転(ライツアウト)、ビット演算
・1列目を決め打ちすれば、後は一意に定まるのでそこから最大を取る。
・最初、行列逆に勘違いしててTLEしまくりだった。。
・気づいた後、実装し直し面倒だったので、最初に行列入れ替えの処理だけ挟んだ。
・計算量は2^10*10000=1000万、pypyAC0.4秒。
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

def bit_count(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

R, C = MAP()
grid = [None] * R
for i in range(R):
    grid[i] = LIST()
# 行列入れ替え
grid = list(zip(*grid))
R, C = C, R

org = int(''.join(map(str, grid[0])), 2)
# 各行をビットで持つ
bit = [0] * (R-1)
for i in range(R-1):
    bit[i] = int(''.join(map(str, grid[i+1])), 2)

ans = 0
# 1行目を全通り試す
for row1 in range(1<<C):
    cnt = bit_count(row1)
    # 行は単体で自由に切り替えられるので、反転と比較して大きい方を使う
    sm = max(cnt, C-cnt)
    change = org ^ row1
    for i in range(R-1):
        # 決め打ちした1行目と同じ変更を、他の行にも適用する
        cnt = bit_count(bit[i] ^ change)
        sm += max(cnt, C-cnt)
    ans = max(ans, sm)
print(ans)
