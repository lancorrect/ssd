# 判断直播中说话人个数

`1_download_and_random_select.py`：下载直播(flv)，并转成音频(wav)，然后随机挑选一定数量的直播

`2_segement_wav_with_vad.py`：按照vad的结果对wav进行切分

`3_ssd.py`：ssd可以识别出每个人的声纹，然后利用聚类算法得到聚类中心，计算它们的距离，大于一定阈值表明有多个说话人

# How to run

按着顺序运行即可

```bash
python filename.py
```



# 此repo的作用

帮助我复习多进程处理数据

# 联系

Email address: zhihaolancorrect@gmail.com

or

Please feel free to open an issue.