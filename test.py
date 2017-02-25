# coding: UTF-8

from subprocess import Popen, PIPE, STDOUT
import glob
import os
import sys

# カレントおよび下位ディレクトリのhtmlファイルを全部取得
p = Popen(["find", "-name", "*.html"],
	stdout=PIPE, stderr=STDOUT)
std_out, err = p.communicate()

print("対象ファイル取得")
print(std_out)

# 文字列をファイル毎にリストに格納
file_list = std_out.split('\n')
# 余分な改行が一つ入るので削除
del file_list[len(file_list) - 1]

# csvファイル出力用のリストを作成
okng_list1 = []
hint_list = []
okng_list2 = []
link_list = []

# ファイルの数だけチェックを処理する
print("バリデーションチェックを開始")
for file in file_list:
	print("対象ファイル：" + file)

	# HTML Tidyを実行して、結果を取得
	p = Popen(["tidy", "--markup", "false", "--input-encoding", "utf8", file],
		stdout=PIPE, stderr=STDOUT)
	std_out, err = p.communicate()

	# 検知のキーワードとNGになった位置
	keyword = "missing <"
	ng_index = std_out.find(keyword)

	# ヒント内容を格納する詳細リスト
	detail_list = []
	
	# タグの欠損があったらNGとファイル名、ヒントを出力
	if ng_index != -1:
		ng_index = 0
		# 取得結果から、NGになる箇所がある限りループ
		while keyword in std_out[ng_index:]:
			ng_index = std_out.find(keyword, ng_index + 1)
			# 該当箇所がなければ-1が返るのでここで抜ける
			if ng_index == -1:
				break
			# 該当箇所を出力して詳細リストに格納
			print(std_out[std_out.rfind("\n", 0, ng_index) + 1: \
				std_out.find("\n", ng_index)])
			detail_list.append(std_out[std_out.rfind("\n", 0, ng_index) + 1: \
				std_out.find("\n", ng_index)])
		print("NG")
		# ファイル出力用の内容をリストに格納
		okng_list1.append("NG")
		hint_list.append(detail_list)

	# 問題がなければOKを出力(正常にチェックが終了した時の文言をキーワードにする)
	elif "Info: Document content looks like" or "This document has errors that" in std_out:
		print("OK")
		# ファイル出力用の内容をリストに格納
		okng_list1.append("OK")
		hint_list.append("")
	else:
		print("チェックが正しく実施できませんでした。\nコマンド：tidy " + \
			file + " を実行して詳細を確認して下さい。")
		sys.exit()
# チェック完了
print("done")

# ファイルの数だけチェックを処理する
print("リンク切れチェックを開始")
for file in file_list:
	print("対象ファイル：" + file)

	# LinkCheckerを実行して、結果を取得
	p = Popen(["linkchecker", file], stdout=PIPE, stderr=STDOUT)
	std_out, err = p.communicate()

	# 検知のキーワードとNGになった位置
	keyword = "No such file or directory"
	ng_index = std_out.find(keyword)

	# リンク切れ箇所を格納するため詳細リストを空にする
	detail_list = []

	# リンク切れがあったらNGと該当箇所を出力
	if ng_index != -1:
		ng_index = 0
		# 取得結果から、NGになる箇所がある限りループ
		while keyword in std_out[ng_index:]:		
			ng_index = std_out.find(keyword, ng_index + 1)
			# 該当箇所がなければ-1が返るのでここで抜ける
			if ng_index == -1:
				break
			# 該当箇所を出力して詳細リストに格納
			print(std_out[std_out.rfind("Real URL", 0, ng_index) + 11: \
				std_out.find("\n", std_out.rfind("Real URL", 0, ng_index))])
			detail_list.append(std_out[std_out.rfind("Real URL", 0, ng_index) + 11: \
				std_out.find("\n", std_out.rfind("Real URL", 0, ng_index))])		
		print("NG")
		# ファイル出力用の内容をリストに格納
		okng_list2.append("NG")
		link_list.append(detail_list)
	
	# 問題がなければOKを出力
	elif "0 errors found" in std_out:
		print("OK")
		# ファイル出力用の内容をリストに格納
		okng_list2.append("OK")
		link_list.append("")
	else:
		print("チェックが正しく実施できませんでした。\nコマンド：linkchecker " + \
			file + " を実行して詳細を確認して下さい。")
		sys.exit()
# チェック完了
print("done")

# 結果ファイルが既に存在する場合
if os.path.isfile("result.tsv"):
	print("結果ファイルが存在します。上書きしますか？(y/n)")
	input_line = raw_input()
	if input_line == "y":
		f = open("result.tsv", "w")
		f.write("")
		f.close()
		print("既存データを削除して上書きします。")
	else:
		print("何もしません。")
		sys.exit()

# 出力用ファイルの作成
f = open("result.tsv", "a")
f.write("ファイル名\tバリデーション結果\tヒント\tリンク切れ有無\tリンク切れ箇所\n")

# 対象htmlファイルの数だけループ
i = 0
for file in file_list:	
	# 結果の書き込み
	f.write(file + "\t" + okng_list1[i] + "\t" + ", ".join(hint_list[i]) + \
		"\t" + okng_list2[i] + "\t" + ", ".join(link_list[i]) + "\n")
	i += 1
f.close()

print("ファイルを出力しました。")


