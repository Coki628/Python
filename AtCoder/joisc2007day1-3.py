# -*- coding: utf-8 -*-

"""
参考：https://babcs2035.hateblo.jp/entry/2019/03/15/164056
・自力はならず。。
・結果的に、愚直シミュレーションで良かったという。
・K=100の内側で、今操作する頂点があって、その内側で50^2=2500の格子を確認しに行く。
・よく考えると、操作する頂点のループは全部合わせて1000が最大なので、Kのループ1回につき
　多くても平均10ずつくらいしかない。こうなると100*10*2500=250万なのでいける量になる。
・問題文にある頂点の近くにはせいぜい10個しか頂点ないとは書いてあったけど、
　それでも回によってはかなりの数(10頂点それぞれが10見つけたら100だし)捌かないといけないような
　気がしてしまって、あーダメだこれ間に合わないだろってなってしまっていた。
・愚直が通ると分かってからは、メモリとの戦いだった。
　計算量250万は大抵pythonじゃきついし今回もTLE、普通にやったらpypyはギリギリでメモリ超過。。
　pypyACやれば一番乗りだったのもあってちょっと頑張った。
・pythonでガベージコレクションの機能とか使うことになるとは。
　pypyにどの程度効果があるかは分からなかったけど、結果的には、多少効果はあるけど、速度とトレードオフって感じ。
・ループ内でやる頻度を微調整して、pypyAC0.7秒。
"""

import sys
import gc
from math import hypot

def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())

N = INT()
M = INT()
D = INT()
K = INT()
cur = []
x, y = MAP()
cur.append((x, y))
XY = set()
for _ in range(N-1):
    x, y = MAP()
    XY.add((x, y))

# 日毎に感染拡大した数をカウント
cnt = [0] * (K+1)
cnt[0] = 1
for k in range(1, K+1):
    nxt = []
    # 今回感染拡大させる都市
    for x, y in cur:
        # 影響範囲Dは狭いので、四方の格子を全探索しても間に合う
        for i in range(x-D, x+D+1):
            for j in range(y-D, y+D+1):
                # 未感染の都市が存在して、距離がD以内なら感染拡大
                if (i, j) in XY and hypot(x-i, y-j) <= D:
                    cnt[k] += 1
                    # 次の回では拡大させる方に使う
                    nxt.append((i, j))
                    # 次回以降要らないので集合からは消す
                    XY.remove((i, j))
    cur = nxt
    # 5回に1回だけメモリ開放(毎回やると遅い)
    if k % 5 == 0:
        gc.collect()
# M日前のものまで遡って合計
ans = sum(cnt[-M:])
print(ans)
