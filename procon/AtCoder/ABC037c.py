# -*- coding: utf-8 -*-

N,K = map(int, input().split())
aN = list(map(int, input().split()))

# ひとつめの部分列を初期値とする
tmp = sum(aN[:K])
ans = tmp
# ふたつめ以降の部分列は-左端+右端しながら合計していく
for i in range(1, N-K+1):
    tmp = tmp - aN[i-1] + aN[i+K-1]
    ans += tmp
print(ans)