# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc082/editorial.pdf
　　　https://atcoder.jp/contests/abc082/submissions/3930644
・メモリ節約のため、dictをsetにした。
・さらに使い終わった1つ前はもういらないので消すことにした。
　→やっとACした。
・あとちょっと初期化とか色々調整
"""

S = input()
X, Y = map(int, input().split())

S2 = []
tmp = 0
# 連続するF毎にまとまるように文字列Sを変形させる
for i in range(len(S)):
    if S[i] == 'F':
        tmp += 1
        if i == len(S)-1:
            S2.append(tmp)
    # T毎に区切る
    else:
        S2.append(tmp)
        tmp = 0

# あるかないか分かればいいので、dictじゃなくてsetにする
dpx = [set() for _ in range(len(S2))]
dpy = [set() for _ in range(len(S2))]
# xの初期化：step0は右移動固定
dpx[0].add(S2[0])
# yの初期化：step0は何も動かないので座標0
dpy[0].add(0)
for i in range(1, len(S2)):
    # 左右に移動する
    for v in dpx[i-1]:
        if i % 2 == 0:
            # 到達可能な位置から+-した位置に送る
            dpx[i].add(v-S2[i])
            dpx[i].add(v+S2[i])
        else:
            # yをやる時はxはそのまま
            dpx[i].add(v)
    # 1つ前は使い終わったので削除
    dpx[i-1] = set()
    # 上下に移動する
    for v in dpy[i-1]:
        if i % 2 == 1:
            dpy[i].add(v-S2[i])
            dpy[i].add(v+S2[i])
        else:
            dpy[i].add(v)
    dpy[i-1] = set()

# 最後のstepでXとYが到達可能ならOK
if X in dpx[-1] and Y in dpy[-1]:
    print('Yes')
else:
    print('No')
