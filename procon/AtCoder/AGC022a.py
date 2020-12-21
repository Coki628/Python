# -*- coding: utf-8 -*-

# 単語
keyword = input()
# 全アルファベットのリスト
word_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

ans_word = ""
ans_ltr = ""
# アルファベットを順番に検索
for ltr1 in word_list:
	# 一致する文字が見つからなかった所で候補の文字にしてループを抜ける
	if keyword.find(ltr1) == -1:
		ans_ltr = ltr1
		break
ans_word = keyword + ans_ltr

# 文字数が最大の26だった時の処理
if len(ans_word) == 26:
	# アルファベットを順番に検索
	for ltr1 in word_list:
		# 今度はzの手前までの部分から候補文字を決める
		if ans_word[0:ans_word.find("z")].find(ltr1) == -1:
			ans_ltr = ltr1
			break
	# zの手前の文字を候補文字に置換
	ans_word = ans_word[0:ans_word.find("z") - 1] + ans_ltr

# 最後の単語
if ans_word == keyword:
	ans_word = "-1"

print(ans_word)
