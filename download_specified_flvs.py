import urllib.request as request
import json
import os
from pathlib import Path
import time
import argparse
from tqdm import tqdm

url_head = "http://10.0.8.114/list/uat/live_video/"

def Download_Flvs(flvs_dir, anchor, live):
    url = url_head + '/' + anchor + '/' + live + '/'
    flvs_html = request.urlopen(url).read()
    flvs_info = json.loads(flvs_html)

    file_path = flvs_dir / anchor / live
    file_path.mkdir(parents=True, exist_ok=True)

    flvs_num = 0
    for flv in tqdm(flvs_info):
        flv_id = flv['name']
        url_flv = url + flv_id
        flv_dir = file_path/flv_id
        request.urlretrieve(url_flv, flv_dir)
        flvs_num += 1
    
    return flvs_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download flvs")
    parser.add_argument("--anchor_id", type=str, default='MS4wLjABAAAAaUtWLa5r-IyrpifnrbyCgbkObKs1hqRLVYHUxQQuHuY')
    parser.add_argument("--live_id", type=str, default='7252526501927684897')
    parser.add_argument("--flvs_dir", type=str, default='/data2/datasets/SSD/video', help='output directory')
    args = parser.parse_args()

    flvs_dir = Path(args.flvs_dir)
    flvs_dir.mkdir(parents=True, exist_ok=True)
    
    print("start downloading")
    flvs_num = Download_Flvs(flvs_dir, args.anchor_id, args.live_id)
    assert flvs_num == 698
    print("downloading completed")