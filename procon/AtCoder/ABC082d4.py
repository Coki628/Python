# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc082/editorial.pdf
　　　https://atcoder.jp/contests/abc082/submissions/3930644
・計算量改善。-8000～8000全部回してたループを値があるkeyだけ回るようにした。
・それに合わせて、貰うDPを配るDPにした。(自分的にはこっちのが分かりやすいし)
　→TLEしない代わりになんとこれはメモリ超過…
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
    # 左右に移動する
    for k, v in dpx[i-1].items():
        if i % 2 == 0:
            # 到達可能な位置から+-した位置に送る
            dpx[i][k-S2[i]] = 1
            dpx[i][k+S2[i]] = 1
        else:
            # yをやる時はxはそのまま
            dpx[i][k] = dpx[i-1][k]
    # 上下に移動する
    for k, v in dpy[i-1].items():
        if i % 2 == 1:
            dpy[i][k-S2[i]] = 1
            dpy[i][k+S2[i]] = 1
        else:
            dpy[i][k] = dpy[i-1][k]
# 最後のstepでXとYが到達可能ならOK
if dpx[len(S2)-1][X] and dpy[len(S2)-1][Y]:
    print('Yes')
else:
    print('No')
