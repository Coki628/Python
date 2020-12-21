# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc106/editorial.pdf
　　　https://www.hamayanhamayan.com/entry/2018/08/18/234325
・累積和
・O(QN+N^2)(10万*500+25万=5025万)がきついらしく、pythonでTLEしてpypyでACして2秒弱くらい。
　この問題制限3secだからいいけど。
"""

N, M, Q = map(int, input().split())
pQ, qQ = [0] * Q, [0] * Q
LR = [[0] * N for i in range(N)]
# 都市番号を0-indexed
for i in range(M):
    l, r = map(int, input().split())
    LR[l-1][r-1] += 1
for i in range(Q):
    p, q = map(int, input().split())
    pQ[i], qQ[i] = p-1, q-1

# 始点固定で終点を動かした累積和を出す
cnt = [[0] * N for i in range(N)]
for i in range(N):
    cnt[i][i] = LR[i][i]
    for j in range(1, N):
        cnt[i][j] = cnt[i][j-1] + LR[i][j]

for i in range(Q):
    ans = 0
    for j in range(pQ[i], qQ[i]+1):
        # ある終点についてもう和が出てるので、
        # ここでは始点を動かして最終的な合計にする
        ans += cnt[j][qQ[i]]
    print(ans)
