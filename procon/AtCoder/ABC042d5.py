# -*- coding: utf-8 -*-

H,W,A,B = map(int, input().split())
mod = (10 ** 9 + 7)

# 予め組み合わせ計算に必要な階乗と逆元のテーブルを作っておく
factorial = [1] * (H+W)
factorial[0] = 1
factorial[1] = 1
for i in range(2,H+W):
    factorial[i] = factorial[i-1] * i % mod

inverse = [1] * (H+W)
# フェルマーの小定理から(x! ** mod-2 % mod == x! ** -1 % mod)
# powに第三引数入れると冪乗のmod付計算を高速にやってくれる
inverse[H+W-1] = pow(factorial[H+W-1], mod-2, mod)
for i in range(H+W-2, 0, -1):
    # 最後から戻っていくこのループならH+W回powするより処理が速い
    inverse[i] = inverse[i+1] * (i+1) % mod

# 組み合わせの数だけ返してくれる関数(自作)
def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = inverse[r] * inverse[n-r] % mod
    return numerator * denominator % mod

h = H - A
w = B + 1
ans = 0
# マスを右上に1つずつずらして、必ず通る場所でパターンを足し合わせていく
while h > 0 and w <= W:
    ans = (ans + nCr(h+w-2, h-1) * nCr(H-h+W-w, H-h)) % mod
    h -= 1
    w += 1

print(ans)