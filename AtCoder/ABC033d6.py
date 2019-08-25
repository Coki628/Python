# -*- coding: utf-8 -*-

"""
・numpy実装、以下、得られた知見を列挙
　・.hstackと.vstackで配列連結、hは横、vは縦に繋ぐ。
　・.transposeで行列入れ替え。
　・np.searchsortedは配列まとめて二分探索できる。
　・atan2とかdegreesみたいな数学系の関数もnp.～でまとめていける。
　・np.whereで条件の場合分けも配列まとめていける。
・pythonで3.5秒AC！pypyより速い。
"""

import sys
import numpy as np

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
EPS = 10 ** -9

N = INT()
XY = np.zeros((N, 2))
for i in range(N):
    x, y = MAP()
    XY[i] = (x, y)
# この後の切り出しがやりやすいように2つ繫げておく
XY = np.vstack((XY, XY))

# ∠bacの時計回り側の角度を計算する
def calc(A, B, C):
    # aから見たb,cへ向かうベクトルにする
    V1 = B - A
    V2 = C - A
    # 行列入れ替え
    V1 = V1.transpose()
    V2 = V2.transpose()
    # 角度も配列全部まとめて処理
    res = np.degrees(np.arctan2(V2[1], V2[0])-np.arctan2(V1[1], V1[0]))
    # 負の数なら時計回り側にあるので逆向きにする
    res = np.where(res<0, res+360, res)
    return res

    # # aから見たb,cへ向かうベクトルにする
    # v1=sub(b, a)
    # v2=sub(c, a)
    # x1,y1=v1
    # x2,y2=v2
    # # atan2を使うと座標上の角度が出しやすい(atan2(y, x)でy=0から見た角度を返す)
    # res=degrees(atan2(y2, x2)-atan2(y1, x1))
    # if res<0:
    #     # 負の数なら時計回り側にあるので逆向きにする
    #     res+=360
    # return res

ans = [0]*3
for i in range(N):
    # まとめて処理するために必要な部分を切り出す
    A = np.array([XY[i]]*(N-2))
    B = np.array([XY[i+1]]*(N-2))
    C = XY[i+2:i+2+(N-2)]
    degs = calc(A, B, C)

    # degs=[]
    # j=(i+1)%N
    # for k in range(i+2, i+N):
    #     # j,kはNで循環させる
    #     k%=N
    #     print(i, j, k)
    #     degs.append(calc(XY[i], XY[j], XY[k]))

    # 二分探索用にソートして番兵と2周目を追加
    degs.sort()
    # numpy配列の連結
    degs = np.hstack((0, degs, 360, degs+360))

    # numpyで二分探索をまとめて処理する
    idx1 = np.searchsorted(degs, degs[:N-1]+90-EPS, side='left')
    idx2 = np.searchsorted(degs, degs[:N-1]+90+EPS, side='right')
    idx3 = np.searchsorted(degs, degs[:N-1]+180-EPS, side='left')
    ans[1] += (idx2-idx1).sum()
    ans[2] += (idx3-idx2).sum()

    # for j in range(N-1):
    #     # degs[j]から見て90の左端、90の右端、180の左端のindexを取る(EPSで誤差をカバーする)
    #     idx1=bisect_left(degs, degs[j]+90-EPS)
    #     idx2=bisect_right(degs, degs[j]+90+EPS)
    #     idx3=bisect_left(degs, degs[j]+180-EPS)
    #     # それぞれ差分で直角と鈍角の個数を出す
    #     ans[1]+=idx2-idx1
    #     ans[2]+=idx3-idx2

# 鋭角三角形の個数 = 全体 - 直角の数 - 鈍角の数
ans[0] = (N*(N-1)*(N-2)//6)-ans[1]-ans[2]
print(*ans)
