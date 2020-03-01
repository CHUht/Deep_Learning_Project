"""
This script is to test transferability:
input : input images folder (the fooled images), model (.pth), num
output : accuracy
"""


import torchvision.transforms as transforms
import torch
from sparsefool import sparsefool
from utils import valid_bounds
from PIL import Image
import scipy.misc
import os

from Model import LeNet

def transferability_test(model_path, input_path):

    # Check for cuda devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # Load a pretrained model
    # net = torch_models.vgg16(pretrained=True)
    # net = LeNet()
    net = torch.load(model_path)
    print(net)
    net = net.to(device)
    net.eval()

    listing = os.listdir(input_path)
    cnt = 0
    cnt_correct = 0
    for image_name in listing:
        cnt += 1
        # Load Image and Resize
        im_orig_ori = Image.open(input_path + image_name).convert('L')
        im_sz = 28
        im = transforms.Compose([transforms.ToTensor()])(im_orig_ori)
        im_orig_ori.close()
        im = im[None, :, :, :].to(device)
        print(im.shape)
        # Visualize results
        predicted_val = net(im)
        predicted_val = predicted_val.data
        max_score, idx = torch.max(predicted_val, 1)
        #print(int(idx[0]))
        if int(idx[0]) == int(image_name[3]):
            cnt_correct += 1
        if cnt % 100 == 0:
            print(str(cnt_correct*100/cnt)+"%")

    print("Accuracy: "+ str(100*cnt_correct/cnt) + " %")

if __name__ == "__main__":
    transferability_test('./Model_trained/2layers.pth', "./output_3layers/")
