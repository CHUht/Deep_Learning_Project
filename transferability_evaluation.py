import torchvision.transforms as transforms
import torch
from PIL import Image
import os
import statistics
import time
statistic = {}
statistic_1 = {}


def transferability_test(input_path, model_path, test_resnet18_model=False):

    # Check for cuda devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    listing_models_all = os.listdir(model_path)
    listing_models = []
    if test_resnet18_model:
        for model_name in listing_models_all:
            if 'animals10' in model_name and 'resnet18_V' in model_name:
                listing_models.append(model_name)
    else:
        for model_name in listing_models_all:
            if 'animals10' in model_name and 'resnet' not in model_name:
                listing_models.append(model_name)
    print('List of models to be tested: ', listing_models)
    for model_name in listing_models:
        print('\nUsing model: ', model_name)
        # Load a pretrained model
        net = torch.load(model_path+model_name, map_location=torch.device('cpu'))
        net = net.to(device)
        net.eval()
        tm = time.time()
        file_list = os.listdir(input_path)
        cnt = 0
        cnt_fooled = 0
        for image_name in file_list:
            cnt += 1
            # Load Image and Resize
            im_orig_ori = Image.open(input_path+"/" + image_name)
            im = transforms.Compose([transforms.ToTensor()])(im_orig_ori)
            im_orig_ori.close()
            im = im[None, :, :, :].to(device)
            fool_label = torch.argmax(net.forward(torch.autograd.Variable(im, requires_grad=True)).data).item()
            if fool_label != int(image_name[0]):
                cnt_fooled += 1
            if cnt % 10 == 0:
                print('For class '+image_name[0]+': '+str(cnt_fooled*100/10)+"%")
        print('Time used: ', round(time.time()-tm))


if __name__ == "__main__":
    transferability_test("./data_output_animals10/delta=255/",model_path='./Model_trained/',
                         test_resnet18_model=True)