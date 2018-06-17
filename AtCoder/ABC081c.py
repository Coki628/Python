# -*- coding: utf-8 -*-

from collections import Counter 

N, K = map(int, input().split())
num_list = list(map(int, input().split()))

# 各数値ごとに数を集計
num_dict = Counter(num_list)
# 数の少ない順でソート
num_tpl = num_dict.most_common()[::-1]

ans = 0
for i in range(len(num_tpl)):
    # 種類数がK以下なら終了
    if len(num_tpl) - i <= K:
        print(ans)
        exit()
    # 数が少ない要素から、答えに追加していく
    ans += num_tpl[i][1]


## TLE : おそらくループ中に毎回ソートしてるのが良くなかったと思う

# ans = 0
# while True:
#     # 種類数がK以下なら終了
#     if len(num_dict) <= K:
#         print(ans)
#         exit()

#     # 一番数が少ない要素の個数分、答えに追加
#     ans += num_dict.most_common()[-1][1]
#     # 一番数が少ない要素を削除
#     del num_dict[num_dict.most_common()[-1][0]]

