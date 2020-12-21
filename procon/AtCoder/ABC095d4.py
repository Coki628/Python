# -*- coding: utf-8 -*-

"""
参考：http://blog.livedoor.jp/misteer/archives/9017497.html
　　　https://www.hamayanhamayan.com/entry/2018/04/21/225805
・累積max(累積和の応用)
・先に回る方向がiより手前にmaxがあったとしても、
　そのパターンは別の周回のiで考慮されているので、
　始点まで戻るマイナス分が多くなっていても問題ない。
　どのみち先に回る方向で最終的にansに採用されるのはiまで行った時のはず。
・他の人見ると、max(右回りのみ, 逆走もする, 今までの最大)が
　max(逆走もする, 今までの最大)だけでACしてるぽいけどなんでだろ。
　こっちは最初のサンプルからダメなんだが、どっか途中の処理が違うのかな。
"""

N, C = map(int, input().split())
x, v = [0], [0]
for i in range(N):
    _x, _v = map(int, input().split())
    x.append(_x)
    v.append(_v)
# 累積max
mx_r = [0] * (N+2)
cal = 0
for i in range(1, N+1):
    cal += v[i] - (x[i] - x[i-1])
    mx_r[i] = max(cal, mx_r[i-1])
mx_l = [0] * (N+2)
cal = v[N] - (C - x[N])
mx_l[N] = cal
for i in range(N-1, 0, -1):
    cal += v[i] - (x[i+1] - x[i])
    mx_l[i] = max(cal, mx_l[i+1])

ans = 0
for i in range(N+1):
    # 右回り→左回り(max(右回りのみ, 逆走もする, 今までの最大))
    ans = max(mx_r[i], mx_r[i] - x[i] + mx_l[i+1], ans)
    # 左回り→右回り
    if i != 0:
        ans = max(mx_l[i], mx_l[i] - (C - x[i]) + mx_r[i-1], ans)
print(ans)
