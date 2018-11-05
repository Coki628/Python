# -*- coding: utf-8 -*-

N = int(input())

# まず深さ(決着までのターン数)を調べる
x = N
depth = 0
while x > 1:
    x //= 2
    depth += 1

# 深さの偶奇で二人の戦略が決まる(シミュレーションの二分木を参照)
if depth % 2:
    # 奇数なら常にtakaが2xでaokiが2x+1
    def taka(x):
        return x * 2
    def aoki(x):
        return x * 2 + 1
else:
    # 偶数なら常にtakaが2x+1でaokiが2x
    def taka(x):
        return x * 2 + 1
    def aoki(x):
        return x * 2

# 設定した戦略でゲーム実施
turn = 1
x = 1
while x <= N:
    if turn % 2:
        x = taka(x)
    else:
        x = aoki(x)
    turn += 1

# 終了時のターン数が奇数になっていたら、偶数回目(aokiのターン)で決着が着いたのでtakaの勝ち
if turn % 2:
    print('Takahashi')
else:
    print('Aoki')