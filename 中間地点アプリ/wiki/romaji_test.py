# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import MySQLdb
import re

st_name = "赤坂駅"
pref ="東京都"

# 駅検索のリクエストを投げて、正しい結果が返るかを確認
r = requests.get("https://ja.wikipedia.org/wiki/" + st_name)
soup = BeautifulSoup(r.content, "html.parser")
# <span style="font-size:120%;">のタグを取得
span1 = soup.find("span", style="font-size:120%;")

if span1 is not None:
    print(span1.contents[2].string[1:])
    # ちなみにこっちは日本語のカナ
    print(span1.contents[0].string)
else:
    # 曖昧さ回避ページに飛んだ時の処理
    # aタグから、属性でtitle="○○駅 (都道府県)"の形を調べる
    tag = soup.find("a", title=st_name + " (" + pref + ")")
    print(tag.attrs["title"])
    # 再検索URL
    print("https://ja.wikipedia.org" + tag.attrs["href"])
    # "○○駅 (都道府県)"で再検索
    r = requests.get("https://ja.wikipedia.org" + tag.attrs["href"])
    soup = BeautifulSoup(r.content, "html.parser")
    # <span style="font-size:120%;">のタグを取得
    span1 = soup.find("span", style="font-size:120%;")
    if span1 is not None:
        print(span1.contents[2].string[1:])
        # ちなみにこっちは日本語のカナ
        print(span1.contents[0].string)
    else:
        print("駅が見つかりません")


# 要素の詳細を確認してみる
# import pprint
# pprint.pprint(tag.__dict__)

# contentsの3つめの要素に"\nローマ字駅名"が格納されているので、それを取得
# (\nを外すために1文字目をトリム)


print("処理終了")

# # 取得した仮名を格納する変数
# kana = ""
# # 処理中の駅名を格納する変数
# st_name = ""
# # 処理中の都道府県名を格納する変数
# pref = ""

# # データベース接続とカーソル生成
# conn = MySQLdb.connect(
#     host='localhost', user='Coki', passwd='0628', db='app', charset='utf8')
# # 引数指定で結果をタプルではなく辞書型で取得
# cursor = conn.cursor(MySQLdb.cursors.DictCursor)

# # エラー処理（例外処理）
# try:
#     # 取得したレコードを全件ループ
#     cursor.execute("SELECT * FROM station WHERE pref_cd IN (47) ORDER BY id ASC")
#     for row in cursor:
#         if row["st_name"] != "" and row["st_name"] is not None:
#             print(row["st_name"], end = " ")
#             # 都道府県コードから処理中の都道府県名を取得
#             cursor.execute("SELECT * FROM pref WHERE pref_cd = %s", (row["pref_cd"],)) # 要素1個のタプルは後ろに,つけないとダメらしい
#             pref = cursor.fetchone()["pref_name"]
#             # 処理中の駅名を取得
#             st_name = row["st_name"] + "駅"
#             # 駅名を使ってリクエストを投げて結果を取得
#             parser.feed(requests.get("https://ja.wikipedia.org/wiki/" + st_name).text)
#             # 仮名が取得できたらDB更新
#             if kana != "":
#                 cursor.execute("UPDATE station SET st_kana = %s WHERE id = %s", (kana, row["id"]))
#                 # 保存を実行（忘れると保存されないので注意）
#                 conn.commit()
#                 # 取得した仮名を初期化
#                 kana = ""
#                 print("更新完了")
#             # 駅名が書き換わっていたら再度リクエストする
#             elif st_name != row["st_name"] + "駅":
#                 # 駅名を使ってリクエストを投げて結果を取得
#                 parser.feed(requests.get("https://ja.wikipedia.org/wiki/" + st_name).text)
#                 # 仮名が取得できたらDB更新
#                 if kana != "":
#                     cursor.execute("UPDATE station SET st_kana = %s WHERE id = %s", (kana, row["id"]))
#                     # 保存を実行（忘れると保存されないので注意）
#                     conn.commit()
#                     # 取得した仮名を初期化
#                     kana = ""
#                     print("更新完了")
#             else:
#                 print("仮名が見つかりません")

# except MySQLdb.Error as e:
#     print('MySQLdb.Error: ', e)

# # 接続を閉じる
# conn.close()

# print("処理終了")

