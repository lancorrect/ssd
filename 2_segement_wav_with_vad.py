import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import argparse
import os
import sys
sys.path.append('/home/wangsh/workspace/wzh/fsmnvad/')
import json
from pathlib import Path
from tqdm import tqdm

import math
import time
import numpy as np
from copy import deepcopy
from multiprocessing import Process, Queue
# from queue import Queue
import multiprocessing as mp

import soundfile
import wave
from pydub import AudioSegment
from onnx_codes.utils import set_all_random_seed, to_device
from onnx_codes.vad import VADTask
from onnx_codes.wav_frontend import WavFrontend

global segments_result_list
with open('./log/selected_flvs.json', 'r') as fin:
    selected_flvs=json.load(fin)
    fin.close()

model_path = '/home/wangsh/workspace/wzh/fsmnvad/speech_fsmn_vad_zh-cn-16k-common-pytorch'
config_path = os.path.join(model_path, 'configuration.json')
model_cfg = json.loads(open(config_path).read())
model_dir = os.path.dirname(config_path)
vad_model_path = os.path.join(
    model_dir, model_cfg['model']['model_config']['vad_model_name'])
vad_model_config = os.path.join(
    model_dir, model_cfg['model']['model_config']['vad_model_config'])
vad_cmvn_file = os.path.join(
    model_dir, model_cfg['model']['model_config']['vad_mvn_file'])

batch_size = 1
if batch_size > 1:
    raise NotImplementedError("batch decoding is not implemented")

ngpu = 1
if ngpu >= 1 and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    batch_size = 1

# 1. Set random-seed
set_all_random_seed(0)

# Build encoder
vad_model, vad_infer_args = VADTask.build_model_from_file(
        vad_model_config, vad_model_path, device
    )

frontend = WavFrontend(cmvn_file=vad_cmvn_file, **vad_infer_args.frontend_conf)

def vad_segment(raw_inputs, vad_model):
    # Build data-iterator
    data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
    dtype = 'float32'

    loader = VADTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
    )

    for batch in loader:
        speech_lengths = torch.tensor([batch.shape[1]], dtype=torch.long)
        frontend.filter_length_max = math.inf
        fbanks, fbanks_len = frontend.forward_fbank(batch, speech_lengths)
        feats, feats_len = frontend.forward_lfr_cmvn(fbanks, fbanks_len)
        fbanks = to_device(fbanks, device=device)
        feats = to_device(feats, device=device)
        feats_len = feats_len.int()

        in_cache = {}
        with torch.no_grad():
            segments_result = vad_model(1, feats, feats_len, batch)
    
    return segments_result

# 流式处理
def segment_by_piece_or_all(wavs, wavs_segment_result, output_dir):
    for wav in tqdm(wavs):
        wav_dict = {wav.name:[]}
        stream, _ = soundfile.read(str(wav))
        segments_result = vad_segment(stream, deepcopy(vad_model))
        wav_dict[wav.name] += segments_result
        wavs_segment_result.append(wav_dict)

        sound = AudioSegment.from_wav(str(wav))
        for result_unit in segments_result[0]:
            start = result_unit[0]
            end = result_unit[1]
            speech_unit = sound[start:end]
            outname = str(wav.stem)+'_'+'0'*(8-len(str(start)))+str(start)+'_'+'0'*(8-len(str(end)))+str(end)+'.wav'
            outfile = Path(output_dir) / outname
            speech_unit.export(outfile, format="wav")

# 多进程处理
# def segment_by_piece_or_all(q_seg, wavs_segment_result, output_dir):
#     while True:
#         if q_seg.qsize() == 0:
#             break
#         print(q_seg.qsize())
#         wav = q_seg.get()
#         wav_dict = {wav.name:[]}
#         stream, _ = soundfile.read(str(wav))
#         segments_result = vad_segment(stream, deepcopy(vad_model))
#         wav_dict[wav.name] += segments_result
#         wavs_segment_result.put(wav_dict)
#         # wavs_segment_result.append(wav_dict)
#         # print(len(wavs_segment_result))

#         sound = AudioSegment.from_wav(str(wav))
#         for result_unit in segments_result[0]:
#             start = result_unit[0]
#             end = result_unit[1]
#             speech_unit = sound[start:end]
#             outname = str(wav.stem)+'_'+'0'*(8-len(str(start)))+str(start)+'_'+'0'*(8-len(str(end)))+str(end)+'.wav'
#             outfile = Path(output_dir) / outname
#             speech_unit.export(outfile, format="wav")
        
        # time.sleep(.1)
    
def get_wavs(display, segment_flag, merge_dir):
    wavs = list(display.rglob("*.wav"))  # 只要在generator前使用list就会改变generator，如果list没有用在原变量上，那么原变量直接变成空的
    wavs_num = len(wavs)
    for elem in selected_flvs:
        anchor = elem['anchor']
        display = elem['display_id']
        flvs_num = len(elem['flvs'])
        if anchor_id.name==anchor and display_id.name==display:
            assert wavs_num == flvs_num, f'{wavs_num} {flvs_num}'
            break 
    
    if segment_flag:
        with wave.open(str(wavs[0]), 'rb') as f:
            nchannels = f.getnchannels()
            sampwidth = f.getsampwidth()
            framerate = f.getframerate()
            f.close()
        merge = np.array([])
        for wav in tqdm(wavs):
            with wave.open(str(wav), "rb") as f:
                n_frames = f.getnframes()
                stream = f.readframes(n_frames)
                f.close()
            array = np.frombuffer(stream, dtype=np.int16)
            merge = np.append(merge, array)
        
        merge_file = merge_dir/'merge.wav'
        wfout = wave.open(str(merge_file), 'wb')
        wfout.setnchannels(nchannels)
        wfout.setsampwidth(sampwidth)
        wfout.setframerate(framerate)
        wfout.writeframes(merge.tobytes())
        wfout.close()
        print(merge_file)
        return [merge_file]
    else:
        return wavs

def dump_queue(q):
    q_list = []
    q.put("STOP")
    for i in iter(q.get, 'STOP'):
        q_list.append(i)
    time.sleep(.1)
    return q_list

if __name__ == '__main__':

    # 路径初始化
    data_dir = Path('/data2/datasets/SSD/')
    time_tag = '4_Aug_2023_12'
    segment_flag = True
    segment_tag = 'segment_piece' if not segment_flag else 'segment_all'

    wavs = data_dir / time_tag / 'raw'
    segments = data_dir/ time_tag / segment_tag
    segments.mkdir(parents=True, exist_ok=True)
    merge_dir = data_dir / time_tag / 'raw_merge'

    # 多进程设置
    segments_result_list = []
    processes_num = 5
    mp.set_start_method('spawn', force=True)

    for anchor_id in Path(wavs).iterdir():
        for display_id in anchor_id.iterdir():
            display_segment_dir = segments/anchor_id.name/display_id.name
            Path(display_segment_dir).mkdir(parents=True, exist_ok=True)
            merge_dir = merge_dir / anchor_id.name / display_id.name
            merge_dir.mkdir(parents=True, exist_ok=True)
            
            # wavs路径列表
            wavs_in_single_display = get_wavs(display_id, segment_flag, merge_dir)
            # segments_result_dict = {"anchor":anchor_id.name, "display":display_id.name, "wavs":[]}
            # wavs_segment_result = segments_result_dict["wavs"]
            
            # segment_by_piece_or_all(wavs_in_single_display, wavs_segment_result, display_segment_dir)

            # q_seg = Queue()
            # for wav in wavs_in_single_display:
            #     q_seg.put(wav)
            # print('='*60)
            # print(f"anchor: {anchor_id.name}, display:{display_id.name}, wavs: {q_seg.qsize()}")

            # #用多进程安全队列存储结果，否则可能不安全
            # wavs_segment_result = Queue()
            # # segments_result_dict = {"anchor":anchor_id.name, "display":display_id.name, "wavs":[]}
            # # wavs_segment_result = []

            # processes = []
            # for i in range(processes_num):
            #     p = Process(target=segment_by_piece_or_all, args=(q_seg, wavs_segment_result, display_segment_dir, ))
            #     processes.append(p)
            
            # for pro in processes:
            #     pro.start()
            #     time.sleep(1)
            
            # for pro in processes:
            #     pro.join()
            #     time.sleep(1)

            # print(f'left q_seg size: {q_seg.qsize()}')
            # # print('convert queue into list')
            # wavs_segment_result = dump_queue(wavs_segment_result)
            # segments_result_dict = {"anchor":anchor_id.name, "display":display_id.name, "wavs":wavs_segment_result}
            
            # print(segments_result_dict)
            # segments_result_list.append(segments_result_dict)
            
            # print(f"{len(wavs_segment_result)} wavs have been split.")
        #     break
        # break
    
    # print('save results.')
    # saved_file = './log/' + segment_tag + '.json'
    # with open(saved_file, 'w', encoding='utf-8') as fout:
    #     json.dump(segments_result_list, fout, indent=4, ensure_ascii=False)
    #     fout.close()
    # print('saving completed.')

