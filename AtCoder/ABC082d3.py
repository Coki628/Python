# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc082/editorial.pdf
・DPをxとyの2つに分けて行う。
　[step][x] = その時xはその状態になれるか
　[step][y] = その時yはその状態になれるか
・x方向とy方向は独立に考えることができる。
・S2最大約4000*座標の範囲8000*2(-8000～8000)=6400万がきついらしくてpypyでもTLE…
"""

from collections import defaultdict

S = input()
X, Y = map(int, input().split())

S2 = []
tmp = 0
# 連続するF毎にまとまるように文字列Sを変形させる
for i in range(len(S)):
    if S[i] == 'F':
        tmp += 1
    # T毎に区切る
    else:
        S2.append(tmp)
        tmp = 0
S2.append(tmp)

# 文字列がFのみだとこの後のDPできないので別処理
if len(S2) == 1:
    if X == S2[0] and Y == 0:
        print('Yes')
    else:
        print('No')
    exit()

# dictならマイナスの添字もそのまま使える
dpx = [defaultdict(int) for _ in range(len(S2))]
dpy = [defaultdict(int) for _ in range(len(S2))]
# xの初期化：step0は右移動、step1ではそのまま
dpx[0][S2[0]] = dpx[1][S2[0]] = 1
# yの初期化：step0は何もなし、step1で上下どちらかに移動
dpy[1][S2[1]] = dpy[1][-S2[1]] = 1
for i in range(2, len(S2)):
    for j in range(-len(S), len(S)+1):
        # 左右に移動する
        if i % 2 == 0:
            # 前回stepに今回step分を+-して到達可能か見る
            dpx[i][j] = dpx[i-1][j-S2[i]] | dpx[i-1][j+S2[i]]
            # xをやる時はyはそのまま
            dpy[i][j] = dpy[i-1][j]
        # 上下に移動する
        else:
            dpx[i][j] = dpx[i-1][j]
            dpy[i][j] = dpy[i-1][j-S2[i]] | dpy[i-1][j+S2[i]]
# 最後のstepでXとYが到達可能ならOK
if dpx[len(S2)-1][X] and dpy[len(S2)-1][Y]:
    print('Yes')
else:
    print('No')
