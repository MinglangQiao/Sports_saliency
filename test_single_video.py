import sys
import os
import numpy as np
import cv2
import torch
import argparse
import copy
from tqdm import tqdm
from os.path import join

from models.HyperSal import VideoSaliencyModel_ml
from models.HyperSal_Audio import AudioVideoSaliencyModel

from src.utils.utils_audio import get_audio_feature, make_dataset_online, get_video_info
from src.utils.utils_image import torch_transform_ml, str2bool, obtain_one_video_frames, blur, img_save

def validate(args):

    file_weight = args.file_weight
    len_temporal = args.clip_size
    test_video_path = args.test_video_path

    dname = test_video_path.split("/")[-1]
    # save result path
    os.makedirs(join(args.save_path, dname), exist_ok=True)

    print(">>>> file_weight: ", file_weight)
    print(">>>> save_path: ", args.save_path)

    if args.use_sound:
        data_all_video_info = get_video_info(test_video_path)

    if args.use_sound: # hyper wt sound 
        model = AudioVideoSaliencyModel(residual_fusion=args.residual_fusion, 
                                        fix_soundnet=True, # always be true during training
                                        hyper_type=args.hyper_type,
                                        test_mode=True, use_support_Mod=args.use_support_Mod)
        print(">>>> model: AudioVideoSaliencyModel")
    else: # hyper wo sound
        model = VideoSaliencyModel_ml()

    state_dict = torch.load(file_weight)

    # set strict=True
    model.load_state_dict(state_dict)
    print(">>>> load model: strict=True", file_weight)

    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()
    # model.backbone.eval()

    if args.use_sound:
        audiodata = make_dataset_online(test_video_path, data_all_video_info)

    one_video_frames = obtain_one_video_frames(test_video_path, downsample=True) # (386, 1080, 1920, 3)
    print("one_video_frames: ", test_video_path, np.shape(one_video_frames))

    # must place here
    frame_name_list = ['{}.png'.format(i) for i in range(len(one_video_frames))]
    print("frame_name_list: ", np.shape(frame_name_list))

    temp_frame_name_list = [frame_name_list[0] for _ in range(31)]
    temp_frame_name_list.extend(frame_name_list)
    frame_name_list = copy.deepcopy(temp_frame_name_list)
    # print("frame_name_list: ", np.shape(frame_name_list), frame_name_list[:3])

    # append 31 frames for prediction of frame 0
    temp_frame = [one_video_frames[0] for _ in range(31)]
    temp_frame.extend(one_video_frames)
    one_video_frames = copy.deepcopy(temp_frame)

    print("one_video_frames: ", np.shape(one_video_frames))
    
    snippet = []
    pbar1 = tqdm(total=len(one_video_frames), desc='Processing frames', ncols=100)
    for i in range(len(one_video_frames)):
        torch_img, img_size = torch_transform_ml(one_video_frames[i])

        snippet.append(torch_img)

        if i >= len_temporal - 1:
            
            audio_feature = None
            if args.use_sound:
                # audio_feature = get_audio_feature(dname, audiodata, args.clip_size, max(0, i-(2*len_temporal-2)))
                audio_feature = get_audio_feature(dname, audiodata, args.clip_size, max(0, i-(2*len_temporal-1)))

                if i < 2*len_temporal-1:
                    # print(">>> audio_feature0: ", np.shape(audio_feature)) # [1, 1, 70560, 1]
                    audio_feature = torch.flip(audio_feature, [2])
        
            clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
            clip = clip.permute((0, 2, 1, 3, 4))

            # print(">>>> i: ", i, dname, np.shape(clip), np.shape(audio_feature))
            process(model, clip, dname, frame_name_list[i], args, img_size, audio_feature=audio_feature,
                    frame_index=i)
            
            del snippet[0]
        
        pbar1.update(1)
    pbar1.close()
    

def process(model, clip, dname, frame_no, args, img_size, audio_feature=None, frame_index=None):
    with torch.no_grad():
        if audio_feature==None:
            smap, audio_fea = model(clip.cuda(), reurn_audio=True)
        else:
            smap = model(clip.cuda(), audio_feature.cuda())

    smap = smap.cpu().data[0].numpy()
    smap = cv2.resize(smap, (img_size[0], img_size[1]))
    smap = blur(smap)
    # 定性分析
    img_save(smap, join(args.save_path, dname, frame_no), normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_weight', default="./saved_models/best_VSTNet147.pth", type=str)
    parser.add_argument('--save_path', default='./results', type=str)
    parser.add_argument('--start_idx', default=-1, type=int)
    parser.add_argument('--num_parts', default=4, type=int)
    parser.add_argument('--clip_size', default=32, type=int)

    parser.add_argument('--hyper_type', default="multiply", type=str) # or add
    parser.add_argument('--use_sound',default=False, type=str2bool) 
    parser.add_argument('--residual_fusion', default=False, type=str2bool)
    parser.add_argument('--use_support_Mod', type=str2bool, default=True)

    parser.add_argument('--test_video_path', default="multiply", type=str) 

    # add gpu
    parser.add_argument('--gpu', default='0', type=str)

    args = parser.parse_args() 

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    validate(args)


"""

python test_single_video.py --gpu 6 \
	--use_sound True --residual_fusion True \
    --test_video_path "/xx/all_1000video/out_of_play_(2).mp4" \
    --save_path "/xx/model_output/" \
    --file_weight "/xx/VSTNet_pseudo_test.pth"

"""