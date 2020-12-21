# -*- coding: utf-8 -*-

H,W,A,B = map(int, input().split())

# 予め組み合わせ計算に必要な階乗のテーブルを作っておく
factorials = [0] * (H+W)
factorials[0] = 1
factorials[1] = 1
for i in range(2,H+W):
    factorials[i] = factorials[i-1] * i
    
# 組み合わせの数だけ返してくれる関数(自作)
def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorials[n]
    # 分母の計算
    denominator = factorials[r] * factorials[n-r]
    return numerator // denominator

h = H - A
w = B + 1
ans = 0
# マスを右上に1つずつずらして、必ず通る場所でパターンを足し合わせていく
while h > 0 and w <= W:
    ans = (ans + nCr(h+w-2, h-1) * nCr(H-h+W-w, H-h)) % (10 ** 9 + 7)
    h -= 1
    w += 1

print(ans)