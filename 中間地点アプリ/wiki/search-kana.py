# coding utf-8

import requests
from html.parser import HTMLParser
"""
駅名読み仮名取得スクリプト
　wikiから目的のタグを読み込み、仮名の情報を取得する
"""
# HTMLParserを継承したクラスを定義する
class MyParser(HTMLParser):

    # コンストラクタ
    def __init__(self):
        HTMLParser.__init__(self)
        self.isKana = False # 仮名の場合のフラグ
        self.isDone = False # 出力済フラグ

    # 開始タグを扱うためのメソッド
    def handle_starttag(self, tag, attrs):
        # まだ未出力の場合のみ対象とする
        if self.isDone == False:
            # attrsリストの長さが1以上だったら処理する(こうしないとこの次で落ちる)
            if len(attrs) >= 1: 
                # spanタグで1番目の属性がstyle="font-size:120%;"の所を抽出
                if tag == "span" and attrs[0] == ("style", "font-size:120%;"): # attrsはリスト内にタプルでkey,valueの構造になってる
                    self.isKana = True
                elif tag == "a" and len(attrs) >= 2:
                    global pref_list
                    for pref in pref_list:                   
                        if attrs[1] == ("title", "赤坂駅 (" + pref + ")"):
                            self.isKana = True

    # 要素内容を扱うためのメソッド
    def handle_data(self, data): 
        if self.isKana:
            print(data, end = "") # 読み仮名のデータを取得(改行なし)
            self.isKana = False
            self.isDone = True

    # 終了タグを扱うメソッド
    def handle_endtag(self, tag):
        # htmlタグの終了時に出力済フラグを戻す
        if tag == "html":
            self.isDone = False

# 駅名を格納するリスト
st_list = []
st_list.append("上大岡駅")
st_list.append("赤坂駅")
# st_list.append("西横浜駅")
pref_list = []
pref_list.append("東京都")
pref_list.append("神奈川県")

# パーサのインスタンスを生成してパースを実行
parser = MyParser()

# 駅の数だけ処理を実行
for st in st_list:
    # 改行の代わりにコンマで区切る
    print(st, end = ",")
    # 駅名を使ってリクエストを投げて結果を取得
    parser.feed(requests.get("https://ja.wikipedia.org/wiki/" + st).text)
    # 改行のみ出力
    print("")

print("処理終了")

