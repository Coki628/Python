# -*- coding: utf-8 -*-

"""
参考：http://kyopro.hateblo.jp/entry/2019/02/06/010004
・蟻本演習3-2-2
・尺取法、幾何、角度ソート
・3点動かす尺取法。前に尺取バグらせそうだしにぶたんでやったやつ。なんとか作れた。
・log乗らない分速くなると思ったけど、意外とにぶたんのと大差なし。pypyAC5.0秒。
"""

import sys
from math import atan2, degrees

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
EPS = 10 ** -9

def sub(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x1-x2, y1-y2)

def get_dig(a, b, c):
    # aから見たb,cへ向かうベクトル
    v1 = sub(b, a)
    x1, y1 = v1
    v2 = sub(c, a)
    x2, y2 = v2
    # atan2がx=0から見た角度を返すので、差分を取る
    res = degrees(atan2(y2, x2) - atan2(y1, x1))
    if res < 0:
        res += 360
    return res

N = INT()
XY = []
for _ in range(N):
    x, y = MAP()
    XY.append((x, y))

ans = [0] * 3
for i in range(N):
    j = (i+1) % N
    a = XY[i]
    b = XY[j]
    digs = []
    for k in range(N):
        if i == j or j == k or k == i:
            continue
        c = XY[k]
        # 角bacの角度
        dig = get_dig(a, b, c)
        digs.append(dig)
    digs += [dig+360 for dig in digs]
    # 線分abの位置を追加
    digs += [0] + [360]
    digs.sort()

    # 尺取法(3点動かす)
    l = m = r = 0
    cnt1 = cnt2 = 0
    while digs[l] < 360:
        # EPSでmを少しだけ左に大きく取る
        while m < len(digs) and digs[m] - digs[l] < 90 - EPS:
            m += 1
        # 直角三角形
        if abs((digs[m] - digs[l]) - 90) <= EPS:
            cnt1 += 1
            m += 1
        # EPSでrを少しだけ右に大きく取る
        while r < len(digs) and digs[r] - digs[l] < 180 + EPS:
            r += 1
        # 鈍角三角形
        cnt2 += r - m
        # 今回はここはなくてもいい
        # if l == m:
        #     m += 1
        # if l == r:
        #     # 左が右に追いついたら、右も左に合わせて+1する
        #     r += 1
        l += 1
    ans[1] += cnt1
    ans[2] += cnt2
total = N * (N-1) * (N-2) // 6
ans[0] = total - ans[1] - ans[2]
print(*ans)
