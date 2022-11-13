# 絵を光らせるソースコード

## 概要

Anime2Sketchを使ってイラストを線画化し、その線画を虹色に光らせ、元のイラストとくっつけた上で動画Gifにするという
ソースコードです。

ソースコードの多くは　[Anime2Sketch](https://github.com/Mukosame/Anime2Sketch) に拠っています。

## 実行環境

Python 3.9.12
ipynbを実行する場合は、Jupyter notebookが必要です。

以下のライブラリはOSやCPU/GPUなどに応じたそれぞれの環境に則したものが必要です。
https://pytorch.org/
- torch
- torchvision


### 使い方

1. 使用前に、Pretrained weight（netG.pth)をGoogleDriveから取得し、weightsディレクトリに配置する必要があります。

https://github.com/Mukosame/Anime2Sketch

2. multicolored.ipynbを開き、上から実行していきます。

3. 最後のセルのoriginal_file_pathに変換したい画像のファイルパスを設定します
