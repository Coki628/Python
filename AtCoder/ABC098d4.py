# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/05/31/015618
しゃくとり法
・ちょっとだけ整理(smとxorは一致させるから1つでよかった)
"""
 
N = int(input())
aN = list(map(int, input().split()))

l,r,ans = 0,0,0
sm = 0
# 外ループで左を動かす
while l < N:
    # 内ループは条件を満たす限り右を動かす
    while r < N and sm + aN[r] == sm ^ aN[r]:
        sm += aN[r]
        r += 1
    # 左右の差分が、今の左位置からうまくいった合計になる
    ans += r - l
    # 左を動かす分、合計から引く
    sm -= aN[l]
    l += 1
print(ans)
