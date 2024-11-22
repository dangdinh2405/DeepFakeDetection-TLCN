import random
import yaml
from PIL import Image as pil_image

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms

from detectors import DETECTOR

import argparse

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='E:/TLCN/Main/training/config/detector/facexray.yaml',
                    help='path to detector YAML file')
parser.add_argument('--weights_path', type=str, 
                    default='E:/TLCN/Main/training/weights/facexray_best.pth')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

def to_tensor(img):
        return transforms.ToTensor()(img)


def create_data_dict(image_path, device):
    # Đọc ảnh
    image = pil_image.open(image_path).convert('RGB')

    # Tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh
        transforms.ToTensor(),         # Chuyển ảnh sang tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
    ])
    image_tensor = transform(image)

    # Chuyển ảnh thành batch (batch size = 1)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Đảm bảo tensor trên đúng thiết bị

    # Tạo data_dict
    data_dict = {
        'image': image_tensor,  # Tensor ảnh
    }

    return data_dict

def main():
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    weights_path = None
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    init_seed(config)

    if config['cudnn']:
        cudnn.benchmark = True

    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    prediction_lists = []
    img = create_data_dict('E:/TLCN/Main/training/imagetest/2_pre.png',device)
    model.eval()
    predictions = inference(model, img)
    logits = predictions['cls']
    prob = torch.sigmoid(logits)
    prediction_lists = list(prob[:, 1].cpu().detach().numpy())
    print(prediction_lists)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
