import sys
sys.path.append('/home/wangsh/workspace/wzh/ssd/speech-nonstreaming-asr-wenet-cpu')
from pathlib import Path
import json
from ssd import ssd_processer as sp
from utils.audio.tools import read_wav
from tqdm import tqdm
import numpy as np
from queue import Queue
from threading import Thread, Lock
import time
from sklearn.metrics.pairwise import cosine_similarity

mutex = Lock()

# 循环
# def wavs_ssd(wavs, threshold):
#     all_labels = []
#     ssd_processer = sp.SSDProcesser()
#     for wav in tqdm(wavs):
#         stream = read_wav(str(wav))
#         embs = ssd_processer.embedding_loop(stream)
#         # print(f"embs' shape is {embs.shape}")
#         n_samples = embs.shape[0]
#         if n_samples <= 1:
#             # 如果n_samples为0，表示时间太短。如果n_samples为1，表示只有一类，要么多人，要么单人，无法判断。
#             labels = []
#         else:
#             labels = ssd_processer.clustering(embs)
#             labels = labels.tolist()
#         all_labels += labels

#     single_person_percentage = sum(all_labels) / len(all_labels)
#     flag = None
#     if single_person_percentage > threshold:
#         flag = 'multi'
#     else:
#         flag = 'single'
#     return flag, all_labels

# 多线程
def wavs_ssd(q_wav, threshold_ssd):
    global all_labels
    ssd_processer = sp.SSDProcesser()
    while True:
        if q_wav.empty():
            break
        wav = q_wav.get()
        stream = read_wav(str(wav))
        embs = ssd_processer.embedding_loop(stream)
        n_samples = embs.shape[0]
        label = 1  # 单人标签
        if n_samples > 1:
            centers = ssd_processer.clustering_center(embs)
            ssd_dis = cosine_similarity(centers[0].reshape(1, -1), centers[1].reshape(1, -1))[0][0]
            # print(ssd_dis)
            if ssd_dis >= threshold_ssd:
                label = 0  # 多人标签
        mutex.acquire()
        all_labels += [label]
        mutex.release()

    
if __name__=='__main__':
    data_dir = '/data2/datasets/SSD/4_Aug_2023_12'
    segment_flag = False
    segment_tag = 'segment_piece' if not segment_flag else 'segment_all'
    wavs_dir = Path(data_dir) / segment_tag

    ssd_result_list = []
    threshold_display = 0.7
    threshold_ssd = 0.75
    for anchor in wavs_dir.iterdir():
        for display in anchor.iterdir():
            wavs = list(display.rglob('*.wav'))

            # 循环
            # flag, all_labels = wavs_ssd(wavs, threshold)

            # 多线程
            q_wav = Queue()
            for wav in wavs:
                q_wav.put(wav)
            print("="*80)
            print(f"q_wav size is {q_wav.qsize()}")
            
            all_labels = []
            threads_num = 10
            l_tasks = []
            for i in range(threads_num):
                l = Thread(target=wavs_ssd, args=(q_wav, threshold_ssd, ))
                l_tasks.append(l)

            for l in l_tasks:
                l.start()
                time.sleep(1)
            for l in l_tasks:
                l.join()
                time.sleep(1)
            
            # print(len(all_labels))
            single_person_percentage = sum(all_labels) / len(all_labels)
            flag = None
            if single_person_percentage > threshold_display:
                flag = 'single'
            else:
                flag = 'multi'

            ssd_result_dict = {"anchor":anchor.name, "display":display.name, "flag":flag, "labels":all_labels}
            print(f"anchor:{anchor.name}, display:{display.name}, flag:{flag}")
            ssd_result_list.append(ssd_result_dict)

    print("start saving log.")
    ssd_result_file = './log/' + segment_tag + '_ssd_result.json'
    with open(ssd_result_file, 'w', encoding='utf-8') as fout:
        json.dump(ssd_result_list, fout, indent=4, ensure_ascii=False)
        fout.close()
    print("over.")
