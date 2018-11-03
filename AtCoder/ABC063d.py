# -*- coding: utf-8 -*-

N,A,B = map(int, input().split())
hN = [int(input()) for i in range(N)]

# 特定の爆破数で全滅させられるかを判定する関数
def enough(T):
    cnt = 0
    # 各魔物について確認していく
    for i in range(N):
        hp = hN[i]
        # 全体への共通ダメージ分
        hp -= B*T
        # 倒せていなければ追加ダメージを与える
        if hp > 0:
            cnt += (hp-1) // (A-B) +1
    # 追加ダメージに使用した合計爆破数がT回以内かどうか判定
    if cnt <= T:
        return True
    else:
        return False

# 2分探索で最初にTrueになる場所を見つける
hi = 10 ** 9
low = 0
# low+1 == hi -> lowとhiが隣り合った状態なので、境界が確定する
while low+1 < hi:
    mid = (hi+low) // 2
    if enough(mid):
        hi = mid
    else:
        low = mid
# hiの位置が最初のTrue
print(hi)

# こっちでも大丈夫だった。でも上のやつのが分かりやすいかな。
# hi = 10 ** 9 + 1
# low = 0
# mid = 10 ** 9 // 2
# while low+1 < hi:
#     if enough(mid):
#         hi = mid
#     else:
#         low = mid
#     mid = (hi+low+1) // 2
# print(mid)