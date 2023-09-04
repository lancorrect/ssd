import urllib.request as request
import json
import os
import time
from tqdm import tqdm

url = "http://10.0.8.114/list/uat/live_video/"

date_min = None
date_max = None

anchors_html = request.urlopen(url).read()
anchors_info = json.loads(anchors_html)
for info in tqdm(anchors_info):
    date_cur_str = info['mtime']
    date_cur = time.strptime(date_cur_str, "%a, %d %b %Y %H:%M:%S GMT")
    date_min = date_cur if  date_min is None or date_min > date_cur else date_min
    date_max = date_cur if  date_max is None or date_max < date_cur else date_max

print(time.strftime("%a, %d %b %Y %H:%M:%S GMT", date_min))
print(time.strftime("%a, %d %b %Y %H:%M:%S GMT", date_max))