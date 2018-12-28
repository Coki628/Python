# -*- coding: utf-8 -*-

"""
しゃくとり法
・前にやったのとか参考にして自力実装
"""
 
N = int(input())
aN = list(map(int, input().split()))

l,r,ans = 0,0,0
sm = xor = 0
# 外ループで左を動かす
while l < N:
    # 内ループは条件を満たす限り右を動かす
    while r < N and sm + aN[r] == xor ^ aN[r]:
        sm += aN[r]
        xor ^= aN[r]
        r += 1
    # 左右の差分が、今の左位置からうまくいった合計になる
    ans += r - l
    # ※今回はl==rの時whileの条件を確実に満たすので、
    # 　ここでl==rになることはなく、このif文は本当は必要ないはず
    if l == r:
        # 左が右に追いついたら、右も左に合わせて+1する
        r += 1
    else:
        # 左を動かす分、合計から引く
        sm -= aN[l]
        xor ^= aN[l]
    l += 1
print(ans)
