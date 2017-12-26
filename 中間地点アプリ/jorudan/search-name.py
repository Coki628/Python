# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import MySQLdb
'''
Jorudan駅名調査スクリプト
　各駅名でJorudanに接続して、正常な値が取得できたらDBのJorudan_nameをst_nameと同じものに更新。
　取得できなかったものは駅名とJorudan名が一致しないということなので手動対応する。
'''
# ひとつ前の駅名を保持
prev_st_name = ""

# データベース接続とカーソル生成
conn = MySQLdb.connect(
    host='localhost', user='Coki', passwd='0628', db='app', charset='utf8')
# 引数指定で結果をタプルではなく辞書型で取得
cursor = conn.cursor(MySQLdb.cursors.DictCursor)

# エラー処理（例外処理）
try:
    # 取得したレコードを全件ループ
    cursor.execute("SELECT DISTINCT st_name FROM station WHERE pref_cd IN (8,9,10,11,12,13,14) AND e_status <> 2 ORDER BY st_kana")
    for row in cursor:
        # ループ初回は前回の駅名がないので何もしない
        if prev_st_name != "":
            # 駅検索のリクエストを投げて、正しい結果が返るかを確認
            r = requests.get("http://www.jorudan.co.jp/norikae/cgi/nori.cgi?eki1=" + prev_st_name + "&eki2=" + row["st_name"] + "&S=検　索")
            soup = BeautifulSoup(r.content, "html.parser")
            # <div class="bk_list_body">があるか調べる
            div1 = soup.find_all("div", class_="bk_list_body")
            # リストが返却されるので、空でないかどうか確認
            if len(div1) != 0:
                print(prev_st_name + " - " + row["st_name"])
                print("OK")
                # 駅名がそのままジョルダン名として使えたので、DBを更新
                cursor.execute("UPDATE station SET jorudan_name = %s WHERE st_name = %s", (prev_st_name, prev_st_name))
                cursor.execute("UPDATE station SET jorudan_name = %s WHERE st_name = %s", (row["st_name"], row["st_name"]))
                # 保存を実行（忘れると保存されないので注意）
                conn.commit()
                print("Data Updated")
            else:
                # リストが空の時は、正常な結果が取得できていないのでNG
                print(prev_st_name + " - " + row["st_name"])
                print("NG")
        # 2つめの駅名を1つめの駅名に入れる
        prev_st_name = row["st_name"]

except MySQLdb.Error as e:
    print('MySQLdb.Error: ', e)

# 接続を閉じる
conn.close()

print("処理終了")
