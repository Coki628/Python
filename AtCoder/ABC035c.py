# -*- coding: utf-8 -*-

N,Q = map(int, input().split())

ans = 0
for i in range(Q):
    l,r = map(int, input().split())
    # 入力に合わせた2進数表記の作成
    tmp = '0' * (l-1)
    tmp += '1' * (r-l+1)
    tmp += '0' * (N-r)
    # XORをとる(int()は第2引数で基数指定)
    ans ^= int(tmp, 2)
# bin()じゃなくてformat()使うと0bって付かない
ans = format(ans, 'b')
# 必要に応じて左に0追加
ans = '0' * (N - len(ans)) + ans

print(ans)