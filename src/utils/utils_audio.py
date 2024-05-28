import torch
import sys
import numpy as np
from tqdm import tqdm
import torchaudio
import os

import io
from pydub import AudioSegment
import cv2

def make_dataset(eval_video_list, test_audios_dir, data_all_video_info):

    video_names = eval_video_list

    dataset = []
    audiodata= {}
    pbar_audio = tqdm(total=len(video_names), desc='Loading audios', ncols=100)
    for i in range(len(video_names)):
        pbar_audio.update(1)

        # if i % 100 == 0:
        #     print('dataset loading [{}/{}]'.format(i, len(video_names)))

        # n_frames = len(os.listdir(join(gt_path, video_names[i], 'maps')))
        n_frames = data_all_video_info['all_video_info'][video_names[i]]['frame_num']
        video_fps = data_all_video_info['all_video_info'][video_names[i]]['fps']

        if n_frames <= 1:
            print("Less frames")
            continue
        
        audio_wav_path = test_audios_dir + video_names[i][:-4] + ".wav"

        if not os.path.exists(audio_wav_path):
            print("Not exists", audio_wav_path)
            continue
            
        # [audiowav,Fs] = torchaudio.load(audio_wav_path, normalization=False)
        [audiowav,Fs] = torchaudio.load(audio_wav_path, normalize=False)

        audiowav = audiowav[:1, :] * (2 ** -23) # normalize to [-1, 1]
        n_samples = Fs/float(video_fps) # number of audio samples per frame
        starts=np.zeros(n_frames+1, dtype=int)
        ends=np.zeros(n_frames+1, dtype=int)

        starts[0]=0
        ends[0]=0
        for videoframe in range(1,n_frames+1):
            startemp=max(0,((videoframe-1)*(1.0/float(video_fps))*Fs)-n_samples/2) # start inedex of audio for videoframe
            starts[videoframe] = int(startemp)

            endtemp=min(audiowav.shape[1],abs(((videoframe-1)*(1.0/float(video_fps))*Fs)+n_samples/2))
            ends[videoframe] = int(endtemp)

        audioinfo = {
            'audiopath': test_audios_dir,
            'video_id': video_names[i],
            'Fs' : Fs, # audio sampling rate
            'wav' : audiowav,
            'starts': starts,
            'ends' : ends
        }

        audiodata[video_names[i]] = audioinfo

    pbar_audio.update(1)

    return audiodata


def make_dataset_online(video_path, data_all_video_info):

    audiodata= {} 
    video_name = video_path.split("/")[-1]

    n_frames = data_all_video_info[video_name]['frame_num']
    video_fps = data_all_video_info[video_name]['fps']

    if n_frames <= 1:
        print("Less frames")

    online_audio = AudioSegment.from_file(video_path)

    # 将AudioSegment对象转换为字节流
    audio_byte_stream = io.BytesIO()
    online_audio.export(audio_byte_stream, format="wav")
    audio_byte_stream.seek(0)  # 重置流的位置到开始

    # 使用torchaudio从字节流中加载音频数据
    audiowav, Fs = torchaudio.load(audio_byte_stream, normalize=False) 

    audiowav = audiowav[:1, :] * (2 ** -23) # normalize to [-1, 1]
    n_samples = Fs/float(video_fps) # number of audio samples per frame
    starts=np.zeros(n_frames+1, dtype=int)
    ends=np.zeros(n_frames+1, dtype=int)

    starts[0]=0
    ends[0]=0
    for videoframe in range(1,n_frames+1):
        startemp=max(0,((videoframe-1)*(1.0/float(video_fps))*Fs)-n_samples/2) # start inedex of audio for videoframe
        starts[videoframe] = int(startemp)

        endtemp=min(audiowav.shape[1],abs(((videoframe-1)*(1.0/float(video_fps))*Fs)+n_samples/2))
        ends[videoframe] = int(endtemp)

    audioinfo = {
        'Fs' : Fs, # audio sampling rate
        'wav' : audiowav,
        'starts': starts,
        'ends' : ends
    }

    audiodata[video_name] = audioinfo

    return audiodata


def get_video_info(video_path):

    if not os.path.exists(video_path):
        print(">>> video path is not exist: ", video_path)

    video_name = video_path.split("/")[-1]

    video_info = {}
    video_info[video_name] = {}

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        # 获取帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 获取总帧数
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"FPS: {fps}")
        print(f"Total Frame Count: {frame_count}")

    # 释放视频捕获对象
    cap.release()

    video_info[video_name]["frame_num"] = frame_count
    video_info[video_name]["fps"] = fps

    return video_info



def get_audio_feature(audioind, audiodata, len_snippet, start_idx):
	"""
    audioind is video name 
    start_idx is frame number
    """

	# max_audio_Fs = 22050
	max_audio_Fs = 44100
	# min_video_fps = 10
	min_video_fps = 20

	max_audio_win = int(max_audio_Fs / min_video_fps * 32)

	audioexcer  = torch.zeros(1,max_audio_win)
	valid = {}
	valid['audio']=0

	if audioind in audiodata:

		excerptstart = audiodata[audioind]['starts'][start_idx+1]
		
		if start_idx+len_snippet >= len(audiodata[audioind]['ends']):
			print("Exceeds size", audioind)
			sys.stdout.flush()
			excerptend = audiodata[audioind]['ends'][-1]
		else:
			excerptend = audiodata[audioind]['ends'][start_idx+len_snippet]	
			
		try:
			valid['audio'] = audiodata[audioind]['wav'][:, excerptstart:excerptend+1].shape[1]
		except:
			pass
		
		audioexcer_tmp = audiodata[audioind]['wav'][:, excerptstart:excerptend+1]
		
		if (valid['audio']%2)==0:
			audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2))] = \
				torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
		else:
			audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2)+1)] = \
				torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
		
	audio_feature = audioexcer.view(1, 1,-1,1)
	return audio_feature