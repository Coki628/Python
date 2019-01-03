# -*- coding: utf-8 -*-

"""
全然ダメだ…。
"""

# MAX：階乗に使う数値の最大以上まで作る
MAX = 101
# 階乗テーブル
factorial = [1] * (MAX)
factorial[0] = factorial[1] = 1
for i in range(2, MAX):
	factorial[i] = factorial[i-1] * i
# 逆元テーブル
inverse = [1] * (MAX)
# powに第三引数入れると冪乗のmod付計算を高速にやってくれる
inverse[MAX-1] = 1 / factorial[MAX-1]
for i in range(MAX-2, 0, -1):
	# 最後から戻っていくこのループならMAX回powするより処理が速い
	inverse[i] = inverse[i+1] * (i+1)

# 組み合わせの数(必要な階乗と逆元のテーブルを事前に作っておく)
def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = inverse[r] * inverse[n-r]
    return int(numerator * denominator)

N = int(input())
aN = list(map(int, input().split()))
aN.sort(reverse=True)

ans = None
mx = -1
for i in range(N-1):
    for j in range(i+1, N):
        if nCr(aN[i], aN[j]) >= mx:
            mx = nCr(aN[i], aN[j])
            ans = (aN[i], aN[j])
print(*(ans))
