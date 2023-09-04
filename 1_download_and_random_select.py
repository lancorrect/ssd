import urllib.request as request
import json
import os
from pathlib import Path
import time
from shutil import rmtree
from ffmpy3 import FFmpeg
from threading import Thread
from queue import Queue
import argparse
from tqdm import tqdm
import random

url_head = "http://10.0.8.114/list/uat/live_video/"

def random_select_flvs(flvs_num, date_set, date_before, json_file):
    if Path(json_file).is_file():
        with open(json_file, 'r', encoding='utf-8') as fin:
            selected_flvs = json.load(fin)
            fin.close()
        print(f"selcted flvs loaded.")
    else:
        qualified_flvs = []
        anchors_html = request.urlopen(url_head).read()
        anchors_info = json.loads(anchors_html)
        for anchor in tqdm(anchors_info):
            date_str = anchor['mtime']
            date = time.strptime(date_str, "%a, %d %b %Y %H:%M:%S GMT")

            if date<=date_set and date>=date_before:
                url_anchor = url_head + '/' + anchor['name']

                display_html = request.urlopen(url_anchor).read()
                display_info = json.loads(display_html)
                for display in display_info:
                    url_display = url_anchor + '/' + display['name']
                    flv_html = request.urlopen(url_display).read()
                    flv_info = json.loads(flv_html)
                    display_dict = {"anchor":anchor['name'], "display_id":display['name'], "flvs":[]}

                    for flv in flv_info:
                        display_dict["flvs"].append(flv['name'])
                    
                    qualified_flvs.append(display_dict)
    
        selected_flvs = random.choices(qualified_flvs, k=flvs_num)
        with open(json_file, 'w', encoding='utf-8') as fout:
            json.dump(selected_flvs, fout, indent=4, ensure_ascii=False)
            fout.close()
        print(f'selected flvs complete reading from url.')
    return selected_flvs

def Download_Flvs(queue_selected, wavs_dir):
    while True:
        if queue_selected.empty():
            break
        flv = queue_selected.get()
        anchor_id = flv['anchor']
        display_id = flv['display_id']
        flvs = flv['flvs']

        file_path = wavs_dir/ Path(anchor_id) / display_id
        file_path.mkdir(parents=True, exist_ok=True)

        for flv_id in flvs:
            url_cur = url_head + anchor_id + '/' + display_id + '/' + flv_id
            # print(url_cur)
            flv_dir = file_path/flv_id
            wav_dir = file_path/flv_dir.name.replace('flv', 'wav')
            request.urlretrieve(url_cur, flv_dir)
            ff = FFmpeg(inputs={str(flv_dir):None}, 
                        outputs={str(wav_dir):'-ar 16000 -ac 1 -ab 256k -v quiet -f wav -y'})
            ff.run()
            flv_dir.unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download flvs and contert them into wavs")
    parser.add_argument("--date_set_str", type=str, default='4_Aug_2023_12', help='the specified date')
    parser.add_argument("--threads", type=int, default=5, help='the number of threads')
    parser.add_argument("--json_file", type=str, default='./selected_flvs.json')
    parser.add_argument("--wavs_dir", type=str, default='/data2/datasets/SSD', help='output directory')
    args = parser.parse_args()
    
    date_min_str = 'Wed, 28 Jun 2023 12:29:09 GMT'
    date_max_str = 'Mon, 14 Aug 2023 09:48:37 GMT'
    date_min = time.strptime(date_min_str, "%a, %d %b %Y %H:%M:%S GMT")
    date_max = time.strptime(date_max_str, "%a, %d %b %Y %H:%M:%S GMT")
    
    date_set_str = args.date_set_str
    date_set = time.strptime(date_set_str, "%d_%b_%Y_%H")
    date_before_str = '1_Aug_2023_0'
    date_before = time.strptime(date_before_str, "%d_%b_%Y_%H")
    assert date_set>=date_min and date_set<=date_max, f"The specified date {date_set_str} is out of scope."

    wavs_dir = Path(args.wavs_dir) / date_set_str / "raw"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    displays_num = 10
    selected_flvs = random_select_flvs(displays_num, date_set, date_before, args.json_file)
    assert len(selected_flvs) == displays_num
    flvs_num = 0
    for item in selected_flvs:
        flvs_num += len(item['flvs'])
    print(f"total flvs num: {flvs_num}.")

    queue_selected = Queue()
    for flv in selected_flvs:
        queue_selected.put(flv)
    
    print("start download.")
    threads_num = args.threads
    l_tasks = []
    for i in range(threads_num):
        th = Thread(target=Download_Flvs, args=(queue_selected, wavs_dir))
        th.start()
        l_tasks.append(th)
        time.sleep(1)
    for th in l_tasks:
        th.join()
    print("download completed.")