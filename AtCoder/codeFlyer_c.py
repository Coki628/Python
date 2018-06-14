 # -*- coding: utf-8 -*-

import itertools

# 入力を各変数に格納
city_cnt, D = map(int, input().split())
# 数字が文字列リストなので数値リストに変換
city_list = [int(i) for i in input().split()]
# 全都市の座標リストから3都市の全組み合わせリストを作成
dist_list = list(itertools.combinations(city_list, 3))

ans = 0
# 3都市の組み合わせの数ループ
for dist in dist_list:
	# 各都市の座標を変数に格納
	i = dist[0]
	j = dist[1]
	k = dist[2]
	# i < j < k 以外は何もしない
	if i < j and j < k:
		# 徒歩と電車の条件を満たすなら答えに+1
		if j - i <= D and k - j <= D and k - i > D:
			ans += 1
	else:
		continue

print(ans)