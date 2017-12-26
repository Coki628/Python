# coding utf-8

import requests
'''
駅情報取得スクリプト
　駅名情報をheartrailsのAPIから取得する
'''
# 路線名を格納するリスト
line_list = []
# 駅名を格納するリスト
station_list = []

# 地域から都道府県を取得
# APIにリクエストの送信
target_url = "http://express.heartrails.com/api/json?method=getPrefectures&area=関東"
# レスポンスのJSONを取得
r = requests.get(target_url)
# 辞書型(MAPみたいの)に変換
r_dict = r.json()
# ネストされたキーの中身は浅いほうから並べて取得
r_list = r_dict["response"]["prefecture"]

# 取得した都道府県から路線を取得
for pref in r_list:
	target_url = "http://express.heartrails.com/api/json?method=getLines&prefecture=" + pref
	r = requests.get(target_url)
	r_dict = r.json()
	r_list = r_dict["response"]["line"]
	# 取得した路線をリストにつめる
	for line in r_list:
		line_list.append(line)

# SETに変換して路線の重複を削除
print(len(line_list))
line_set = set(line_list)
print(len(line_set))

# 取得した路線から駅名を取得
for line in line_set:
	print(line + "の駅名を取得")
	target_url = "http://express.heartrails.com/api/json?method=getStations&line=" + line
	r = requests.get(target_url)
	r_dict = r.json()
	r_list = r_dict["response"]["station"]
	# 取得した駅名をリストにつめる
	for station in r_list:
		station_list.append(station["name"])
		print(station["name"])

# SETに変換して駅名の重複を削除
print(len(station_list))
station_set = set(station_list)
print(len(station_set))

# コンマを区切り字とした文字列を作成("○○","○○",～の形を作る)
result = "\"" + "\",\"".join(station_set) + "\""

# 結果出力用ファイルの作成
# with構文で確実にcloseする
with open('result.txt','w') as f:
	f.write(result)
print("結果出力ファイル作成完了")
	