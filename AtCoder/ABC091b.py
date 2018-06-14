# -*- coding: utf-8 -*-

# 青カードのリスト
blue_count = int(input())
blue_list = []
for i in range(blue_count):
	blue_list.append(input())
# 赤カードのリスト
red_count = int(input())
red_list = []
for i in range(red_count):
	red_list.append(input())

# 全単語リスト作成
all_words = []
# 赤青を連結
all_words.extend(blue_list)
all_words.extend(red_list)
# 重複を削除
all_words = list(set(all_words))

# 全単語でdictを作る
word_dict = {}
for word in all_words:
	word_dict[word] = 0
# 青カード分足す
for word in blue_list:
	word_dict[word] += 1
# 赤カード分引く
for word in red_list:
	word_dict[word] -= 1
# 結果値をリストに
result_list = []
for key, value in word_dict.items():
	result_list.append(value)
# 最大値を算出
result = max(result_list)
# 最大値が0以下なら0
if result <= 0:
	ans = 0
else:
	ans = result

print(ans)