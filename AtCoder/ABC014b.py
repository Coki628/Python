# -*- coding: utf-8 -*-

N, X = map(int, input().split())
aN = list(map(int, input().split()))

# この後の2進表記に合わせて向きを反転
aN.reverse()
# 2進表記変換と0埋め
x_list = list(format(X, 'b').zfill(N))

ans = 0
for i in range(N):
    if x_list[i] == '1':
        ans += aN[i]
print(ans)
