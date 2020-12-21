# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc105/editorial.pdf
　　　http://drken1215.hatenablog.com/entry/2018/08/13/125300
　　　https://www.hamayanhamayan.com/entry/2018/08/12/010816
・左端固定で右端動かすループの累積和だけ取れば、
　(右端の累積和)-(左端-1の累積和)で、どこでも分かるようになる。
・B-AがMの倍数 → BとAのmod Mが等しい(やってみれば分かる)
・各ループ内で、それまでに集計した累積和mod Mから左端として使える場所を数える。
"""

from collections import defaultdict

N, M = map(int, input().split())
aN = list(map(int, input().split()))

# 累積和を一時的に格納する変数
sm = 0
ans = 0
# 累積和のmod Mを格納するdict
d = defaultdict(int)
# 何もない累積和0の分を足しておく
d[0] = 1
for i in range(N):
    sm += + aN[i]
    # 今回のsm%Mを右端として、mod Mが同値になる左端候補の数を答えに加算
    ans += d[sm%M]
    # 今回のsm%Mはこの後左端候補として使うので+1
    d[sm%M] += 1
print(ans)
