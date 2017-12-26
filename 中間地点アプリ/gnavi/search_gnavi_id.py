# coding utf-8

import requests
from html.parser import HTMLParser
"""
ぐるナビ駅情報調査スクリプト
　URLのID部分を順番に変えていき、各駅とIDの紐付けを確認する
""" 
# HTMLParserを継承したクラスを定義する
class MyParser(HTMLParser): 
    # 開始タグを扱うためのメソッド
    def handle_starttag(self, tag, attrs):
        # attrsリストの長さが3以上だったら処理する(こうしないとこの次で落ちる)
        if len(attrs) >= 3: 
            # inputタグで3番目の属性がname="fwp"の所を抽出
            if tag == "input" and attrs[2] == ("name", "fwp"): # attrsはリスト内にタプルでkey,valueの構造になってる
                # value属性の値を取得
                name = attrs[1][1]
                # 最後の文字が"駅"だったらそこは削る
                if name[-1:] == "駅":
                    name = name[:-1]
                # 改行なし出力
                print(name, end = "")

# パーサのインスタンスを生成してパースを実行
parser = MyParser()
#1～9999まで繰り返す
for num in range(1, 10000):
    # 4桁まで0埋め
    num = "{0:0>4}".format(num)
    # 改行の代わりにコンマで区切る
    print(str(num), end = ",")
    # 文字列型に変換してURLに組み込む
    parser.feed(requests.get("https://r.gnavi.co.jp/eki/000" + str(num) + "/rs/").text)
    # 改行のみ出力
    print("")
