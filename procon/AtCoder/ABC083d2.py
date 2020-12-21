# -*- coding: utf-8 -*-

"""
尺取法の応用
・4点を持ちながら3つの区間を確認していく。
・まあこれは問題の答えとしてはWAなんだけども。
"""

S = input()

l,r,r2 = 0,0,0
ans,prev = 0,0
while l < len(S):
    # prev,l,r,r2の4点で、3つの区間の情報を保持する
    if l == r == r2:
        while r < len(S) and S[l] == S[r]:
            r += 1
        r2 = r
    while r2 < len(S) and S[r] == S[r2]:
        r2 += 1
    # 今回区間
    sm = r - l
    # 前回区間
    prev_sm = l - prev
    # 次回区間
    nxt_sm = r2 - r
    # 両側に区間があれば追加
    sm += min(prev_sm, nxt_sm)
    ans = max(sm, ans)
    # 点を動かす
    prev = l
    l = r
    r = r2
print(ans)
