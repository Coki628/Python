# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc105/editorial.pdf
　　　http://drken1215.hatenablog.com/entry/2018/08/13/125300
　　　https://www.hamayanhamayan.com/entry/2018/08/12/010816
・左端固定で右端動かすループの累積和だけ取れば、
　(右端の累積和)-(左端-1の累積和)で、どこでも分かるようになる。
・B-AがMの倍数 → BとAのmod Mが等しい(やってみれば分かる)
・余りの数を全部集計してから、各余りの数について2つ選ぶ組み合わせを数える。
"""

from collections import defaultdict

N, M = map(int, input().split())
aN = list(map(int, input().split()))

# 累積和を一時的に格納する変数
sm = 0
# 累積和のmod Mを格納するdict
d = defaultdict(int)
# 何もない累積和0の分を足しておく
d[0] = 1
for i in range(N):
    sm += + aN[i]
    d[sm%M] += 1

ans = 0
for k, v in d.items():
    # 各余りの数について、v個から2つ選ぶ組み合わせ(vC2)
    ans += v * (v-1) // 2
print(ans)
