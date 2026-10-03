"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
#
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
# CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=CVD2014 --frame_batch_size=32
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=LIVE-Qualcomm --frame_batch_size=8

import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from scipy.stats import entropy
import cv2
from PIL import Image
import os
import h5py
from models import build_model
import numpy as np
import random
from argparse import ArgumentParser
from config import get_config
import torch.nn.functional as F
from skimage.feature import greycomatrix, greycoprops




class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir,high_dir, low_dir, video_names,score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.no_names = video_names
        self.high_names = high_dir
        self.low_names = low_dir
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height
        self.entro = 0

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        no_name = self.no_names[idx]
        high_name = self.high_names[idx]
        low_name = self.low_names[idx]
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height, self.width, inputdict={'-pix_fmt':'yuvj420p'})
        else:
            #print(os.path.join(self.videos_dir, video_name))
            #video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
            video_data = skvideo.io.vread(no_name)
            high_data = skvideo.io.vread(high_name)
            low_data = skvideo.io.vread(low_name)
        video_score = self.score[idx]
        #print(video_score)

        transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        print(video_length)
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        #transformed_video_h = torch.zeros([video_length-1, video_channel, video_height, video_width])
        transformed_video_l = torch.zeros([video_length-1, video_channel, 224, 224])
        entro = []
        for frame_idx in range(video_length-1):
            frame = video_data[frame_idx+1]
            #frame_h =high_data[frame_idx]
            #frame_l =low_data[frame_idx]
            #entro.append(calculate_entropy(frame))
            entro.append(compute_fft2_complexity(frame))
            #entro.append(compute_glcm_complexity(frame))
            frame = Image.fromarray(frame)
            frame = transform(frame)
            #frame_h = Image.fromarray(frame_h)
            #frame_h = transform(frame_h)
            #frame_l = Image.fromarray(frame_l)
            #frame_l = transform(frame_l)
            transformed_video[frame_idx] = frame
            #transformed_video_h[frame_idx] = frame_h
            #transformed_video_l[frame_idx] = frame_l
        entro = (entro - min(entro))/(max(entro)-min(entro))
        entro = 100*(entro/sum(entro))

        sample = {'video': transformed_video,
                  #'video_h':transformed_video_h,
                  #'video_l':transformed_video_l,
                  'score': video_score,
                  'entropy': entro}

        return sample


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.tanh = nn.Tanh()
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            #print(x.shape)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1).squeeze(3)
                features_std = global_std_pool2d(x).squeeze(3)
                print(features_mean.shape)
                return features_mean, features_std

class SwinT(torch.nn.Module):
    def __init__(self):
        super(SwinT, self).__init__()
        self.features = build_model(config)
        checkpoint = torch.load(args.resume)
        new_pth = self.features.state_dict()  # 需要加载参数的模型
        # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in new_pth}
        pretrained_dict = {}  # 用于保存公共具有的参数
        for k, v in checkpoint['model'].items():
            for kk in new_pth.keys():
                if kk in k:
                    pretrained_dict[kk] = v
                    break
        new_pth.update(pretrained_dict)
        self.features.load_state_dict(new_pth)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self,x):
        with torch.no_grad():
          x = self.features(x)
          features_mean = nn.functional.adaptive_avg_pool1d(x, 1)
          features_std = global_std_pool1d(x)
        print(features_mean.shape)
        #print(features_std.shape)
        return features_mean,features_std

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)
def global_std_pool1d(x):
    """2D global standard variation pooling"""
    return torch.std(x,dim=2, keepdim=True)

def calculate_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray.shape)
    # 计算图像的信息熵
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist / float(np.sum(hist))
    entro = entropy(hist, base=2)
    #print(entro)
    return entro

def compute_glcm_complexity(image):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算灰度共生矩阵
    glcm = greycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # 计算GLCM属性（对比度、同质性等）
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    # 计算综合复杂度（示例：对比度和同质性的加权和）
    complexity = contrast + (1 - homogeneity)

    return complexity



def compute_fft2_complexity(image):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的傅里叶变换
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)

    # 计算频谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # 计算复杂度（高频成分的总和）
    complexity = np.sum(magnitude_spectrum)

    return complexity


def get_features(video,entro_data,device='cuda'):
    """feature extraction"""
    memory = 5
    max_frame = 32
    i = 0
    ii =0
    extractor1 = SwinT().to(device)
    extractor2 = ResNet50().to(device)
    video_length = video.shape[0]
    frame_start = 0
    frame_end = 0
    frame_batch = 32
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor1.eval()
    extractor2.eval()

    with torch.no_grad():
       while frame_end < video_length:
            temp_m = 0
            frame_batch = 0
            while temp_m < memory and ii < video_length and frame_batch <= max_frame:
                temp_m += entro_data[i]
                frame_batch += 1
                ii+=1
            frame_end = frame_start+frame_batch
            batch = video[frame_start:frame_end].to(device)
            #batch_h = video_high[frame_start:frame_end].to(device)
            #batch_l = video_low[frame_start:frame_end].to(device)
            #print(batch_h.shape)
            #print(batch_l.shape)
            features_mean1, features_std1 = extractor1(batch)# switch CNN or Swin-T
            #features_mean2, features_std2 = extractor2(batch)
            #output1 = torch.cat((output1, features_mean1), 0)
            #output2 = torch.cat((output2, features_std1), 0)
            #frame_end += frame_batch
            #frame_start += frame_batch
    #last_batch = video[frame_start:video_length].to(device)
    #features_mean, features_std = extractor1(last_batch)
    #output1 = torch.cat((output1, features_mean), 0)
    #output2 = torch.cat((output2, features_std), 0)
    #output = torch.cat((output1, output2), 1).squeeze()
            #output1 = torch.cat((output1, features_mean2), 1)
            #output2 = torch.cat((output2, features_mean2), 1)
            output1 = torch.cat((output1, features_std1), 0)
            output2 = torch.cat((output2, features_std1), 0)
            #out1 = torch.cat((out1,output1),0)
            #out2 = torch.cat((out2, output2), 0)
            frame_start += frame_batch
            #output1 = torch.cat((output1, features_mean2), 0)
            #output2 = torch.cat((output2, features_std2), 0)
            print(output1.shape)
            #frame_start += frame_batch
    output = torch.cat((output1, output2), 1).squeeze()
    #output = out1
    print(output.shape)
    #output = torch.cat((out1, out2), 1).squeeze()



    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='VSR', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=16,
                        help='frame batch size for feature extraction (default: 64)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--resume', help='resume from checkpoint',
                        default='C:/D/VQA/Swin-Transformer-main/checkpoint/ourswin_tiny_patch4_window7_224.pth')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file',
                        default='configs/swin/swin_tiny_patch4_window7_224.yaml')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # 这个是使用argparse模块时的必备行,将参数进行关联

    # args, unparsed = parser.parse_known_args()

    args = parser.parse_args()
    config = get_config(args)

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        videos_dir = 'C:/D/VQA/KoNViD_1k_videos/'  # videos dir
        features_dir = './CNN_features_KoNViD-1k/'  # features dir
        datainfo = './data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        videos_dir = '/media/ldq/Research/Data/CVD2014/'
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        videos_dir = '/media/ldq/Others/Data/12.LIVE-Qualcomm Mobile In-Capture Video Quality Database/'
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'
    if args.database == 'LIVE-VQC':
        videos_dir = 'C:/D/VQA/VSR/LIVE-VQC/'
        features_dir = 'CNN_adaorgswinonlylow_LIVE-VQC/'
        datainfo = 'data/LiveVQC_data.mat'
    if args.database == 'VSR':
        videos_dir = 'C:/D/VQA/VSR/sr/'  # videos dir
        features_dir = './CNN_adaourswinonly_VSR/'  # features dir
        datainfo = 'C:/D/VQA/VSFA-master/data/VSRinfo.mat'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
    high_names = [Info[Info['high_diffs'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['high_diffs'][0, :]))]
    low_names = [Info[Info['low_diffs'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['low_diffs'][0, :]))]
    no_names = [Info[Info['no_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['no_names'][0, :]))]
    high_opticals = [Info[Info['high_opticals'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['high_opticals'][0, :]))]
    low_opticals = [Info[Info['low_opticals'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['low_opticals'][0, :]))]
    #print(high_names)
    scores = Info['scores'][0, :]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])
    dataset = VideoDataset(videos_dir=videos_dir,high_dir=high_opticals,low_dir=low_opticals,video_names=no_names,score=scores, video_format=video_format, width=width, height=height)

    for i in range(len(dataset)):
        current_data = dataset[i]
        current_video = current_data['video']
        #current_video_h = current_data['video_h']
        #current_video_l = current_data['video_l']
        current_score = current_data['score']
        current_entropy = current_data['entropy']
        print('Video {}: length {}'.format(i, current_video.shape[0]))
        features = get_features(current_video, current_entropy, device)
        np.save(features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_score)
