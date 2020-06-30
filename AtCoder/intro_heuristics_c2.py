"""
・マラソン、局所探索法(山登り法)の準備。
・愚直を部分更新に最適化。実装疲れた。。
・とりあえず変更は他の日付には影響するけど、他のコンテストには影響しないので、
　変更のある2つのコンテストに絞って更新する方針を取った。
・last配列は2次元で各日付毎も管理、コンテスト毎のが独立なので日付の方を2次元目に。
・重みなしコストと日付を2次元で縦軸と横軸にしてみると、開催日が変わった時のコスト変更は
　ちょうど長方形領域を作ることが分かる。(自分ノート参照)
・これに各コンテスト毎の重みを乗せれば、コストの変動はO(1)で処理できる。
・last配列を更新する時にどのみちO(D)はかかってしまうけど、全部やる訳じゃないので定数倍はだいぶ軽いし、
　26個見ていたコンテストを2つしか見ないので、log1個分くらいは計算量自体も改善してる。
・これならクエリの10万と合わせても10万*365=3650万くらいで済むのでpypyで通る。pypyAC0.20秒。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

D = INT()
C = LIST()
# last配列との兼ね合いで、日付は1-indexedで管理
S = [[]] * (D+1)
for i in range(1, D+1):
    S[i]  = LIST()
T = [0] + [t-1 for t in LIST(D)]
M = 26

# last[i][j] := j番目のコンテストについてi日目時点での重み付け前のコスト
last = list2d(M, D+2, 0)
def check(T):
    score = 0
    for i, t in enumerate(T[1:], 1):
        score += S[i][t]
        for j in range(M):
            last[j][i] = last[j][i-1] + 1
        last[t][i] = 0
        for j in range(M):
            score -= C[j] * last[j][i]
    return score

# day日目のコンテストをaからbに変更する
def change(day, a, b):
    # コンテストaのコストを再計算
    nxtday = last[a].index(0, day+1)
    w = nxtday - day
    h = last[a][day-1] + 1
    a_change = C[a] * h * w
    # last配列の更新
    for d in range(day, nxtday):
        last[a][d] = last[a][d-1] + 1

    # コンテストbのコストを再計算
    nxtday = last[b].index(0, day+1)
    w = nxtday - day
    h = last[b][day-1] + 1
    b_change = C[b] * h * w
    last[b][day] = 0
    for d in range(day+1, nxtday):
        last[b][d] = last[b][d-1] + 1

    res = -a_change + b_change - S[day][a] + S[day][b]
    return res

score = check(T)
Q = INT()
for i in range(Q):
    d, q = MAP()
    q -= 1

    score += change(d, T[d], q)
    print(score)
    T[d] = q
