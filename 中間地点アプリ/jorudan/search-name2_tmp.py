# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
'''
Jorudan駅名調査スクリプト2(エラー処理確認用)
　各駅名でJorudanに接続して、正常な値が取得できたらDBのJorudan_nameとURLから取得した駅名を比較。
　一致したものはJorudanに正しく接続できるものとみなす。
'''

prev_st_name = "新"
current_st_name = "泉岳寺"

# 駅検索のリクエストを投げて、正しい結果が返るかを確認
r = requests.get("http://www.jorudan.co.jp/norikae/cgi/nori.cgi?eki1=" + prev_st_name + "&eki2=" + current_st_name + "&Sok=決+定")
soup = BeautifulSoup(r.content, "html.parser")
# <div class="bk_list_body">があるか調べる
div1 = soup.find_all("div", class_="bk_list_body")
# リストが返却されるので、空でないかどうか確認
if len(div1) != 0:
    print(prev_st_name + " - " + current_st_name)
    # 検索結果で表示された駅名を取得
    h2 = soup.find("h2", class_="time")
    # ○○→○○の→より手前部分を取得(最初に改行コードが入っているので2文字目から)
    result_name_from = h2.string[1:h2.string.find("→")]
    # ○○→○○の→より後ろの部分を取得(最後に改行コードが入っているので最後から2番目の文字まで)
    result_name_to = h2.string[h2.string.find("→") + 1:-1]
    print(result_name_from + " - " + result_name_to)
    # 上記駅名の比較
    if (prev_st_name == result_name_from and current_st_name == result_name_to):
        print("OK")
    else:
        print("NG")
else:
    # リストが空の時は、正常な結果が取得できていないのでNG
    print(prev_st_name + " - " + current_st_name)
    print("NG")

print("処理終了")
