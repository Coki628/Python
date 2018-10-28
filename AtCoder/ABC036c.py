# -*- coding: utf-8 -*-

N = int(input())
aN = [int(input()) for i in range(N)]

# 重複削除とソート
l = sorted(list(set(aN)))
# zip(key, value)で必要なdictを作る
d = dict(zip(l, range(len(l))))
# あとはマッピングに当てはめるだけ
for i in range(N):
    print(d[aN[i]])