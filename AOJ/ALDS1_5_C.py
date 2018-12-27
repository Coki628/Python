# -*- coding: utf-8 -*-

"""
コッホ曲線
参考：https://www.mynote-jp.com/entry/2016/04/30/201249
　　　https://atarimae.biz/archives/23930
・回転行列を使うと、向きの変わった座標が出せる。
・そのために行列の計算方法も確認。
　→m行n列成分は、「左の行列のm行目」と「右の行列のn列目」の内積
"""

from math import sin, cos, radians

N = int(input())

def dfs(p1x, p1y, p2x, p2y, depth):
    if depth == N:
        return
    sx = p1x + (p2x-p1x)/3
    tx = p1x + (p2x-p1x)/3*2
    sy = p1y + (p2y-p1y)/3
    ty = p1y + (p2y-p1y)/3*2
    # s,tを基準の位置として左回りに60度回転させる
    ux = sx + (tx-sx) * cos(radians(60)) - (ty-sy) * sin(radians(60))
    uy = sy + (tx-sx) * sin(radians(60)) + (ty-sy) * cos(radians(60))
    # 再帰が終了した場所から出力していけば、線分上の各頂点の順番になる
    dfs(p1x, p1y, sx, sy, depth+1)
    print(sx, sy)
    dfs(sx, sy, ux, uy, depth+1)
    print(ux, uy)
    dfs(ux, uy, tx, ty, depth+1)
    print(tx, ty) 
    dfs(tx, ty, p2x, p2y, depth+1)

print(0, 0)
dfs(0, 0, 100, 0, 0)
print(100, 0)
