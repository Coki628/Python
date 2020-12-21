# -*- coding: utf-8 -*-

# 入力の数値(m)
m = int(input())
# kmに変換
km = m / 1000
# 0.1未満はどの条件も当てはまらないので0のまま
vv = 0
# 各条件に合わせて計算
if km >= 0.1 and km <= 5:
	vv = km * 10
elif km >= 6 and km <= 30:
	vv = km + 50
elif km >= 35 and km <= 70:
	vv = (km - 30)/5 + 80
elif km >= 70:
	vv = 89
# 2桁未満はゼロ埋め
ans = '%02d' % vv

print(ans)