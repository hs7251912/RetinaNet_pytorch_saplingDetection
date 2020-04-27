from Build_Network import RetinaNet
from PIL import Image
import torch
import csv
from data_loader import CSV_Dataset, UnNormalizer, Normalizer, Resizer, AspectRatioBasedSampler, collater
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import skimage.color
import skimage
import time
import os
import cv2
import pandas as pd

class Img_Dataset(Dataset):
    def __init__(self,img_path_list,classes_file,transform=None):
        self.img_path_list = img_path_list
        self.classes_file = classes_file

        with open(self.classes_file,'r') as file:
            self.classes = self.load_classes(csv.reader(file, delimiter=','))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # image_names指的是图片的路径
        self.image_names = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = self.read_image(index)
        # annotations = self.read_annotations(index)
        # sample = {'image': image, 'annotations': annotations}
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_classes(self, csv_reader):
        result = {}

        for data in csv_reader:

            class_name, class_id = data
            class_id = int(class_id)

            result[class_name] = class_id

        return result

    def read_image(self, image_name):
        image = skimage.io.imread(self.image_names[image_name])

        if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)

        return image.astype(np.float32) / 255.0

    def num_classes(self):
        return max(self.classes.values()) + 1

    def label_to_name(self, label):
        return self.labels[label]



class Resizer(object):
    """
    reshape the image to an acceptabel size
    """

    def __call__(self, sample, min_side=3648, max_side=4864):
        image = sample['image']

        rows, cols, cnls = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        # annotations[:, :4] *= scale
        return {'image': torch.from_numpy(new_image), 'scale': scale}

class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image = sample['image']

        return {'image': ((image.astype(np.float32) - self.mean) / self.std)}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]

        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]

        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def read_image(image_path):
    image = skimage.io.imread(image_path)

    if len(image.shape) == 2:
        image = skimage.color.gray2rgb(image)

    return image.astype(np.float32) / 255.0

# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    # print('Box is: {}'.format(box))
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def main():
    class_list = './data/SaplingDetection/StepCroped_anno/classes.csv'
    # 模型路径
    model_path = './checkpoint/sapling_best.pt'
    # 图片路径
    img_dir = '/home/xair/Larry/retinanet-based-on-pytorch-1.1.0/data/Images'
    img_path_list = []
    # img_path = '/home/xair/Larry/retinanet-based-on-pytorch-1.1.0/data/Images/1497.JPG'
    img_list = os.listdir(img_dir)
    for img in img_list:
        img_path = os.path.join(img_dir,img)
        img_path_list.append(img_path)

    # ------------------------ 加载数据 --------------------------- #
    # Data augmentation and normalization for training
    # Just normalization for validation
    # 定义预训练变换
    dataset_img = Img_Dataset(img_path_list,class_list,transforms.Compose([Normalizer(), Resizer()]))
    dataloader_img = DataLoader(dataset_img)

    # 初始化模型
    model = RetinaNet(dataset_img.num_classes())
    # 将参数转换为cuda型
    if torch.cuda.is_available():
        model = model.cuda()
    # 加载模型
    model.load_state_dict(torch.load(model_path))

    scores_threshold = 0.05
    max_boxes_per_image = 100
    iou_threshold = 0.5
    model.training = False
    model.eval()
    unnormalize = UnNormalizer()

    # Draws a caption above the box in an image
    # def draw_caption(image, box, caption):
    #     # print('Box is: {}'.format(box))
    #     b = np.array(box).astype(int)
    #     cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    #     cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx,data in enumerate(dataloader_img):
        # with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
        data['image'] = data['image'].permute(0,3,1,2)
        with torch.no_grad():
            st = time.time()
            # print((data['image'].permute(2, 0, 1).cuda().float()).shape)
            scores, classification, transformed_anchors = model(data['image'].cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            # tensor([1.0000, 1.0000, 1.0000, 0.8962, 0.0568, 0.0527], device='cuda:0')
            # print(scores)
            # (array([0, 1, 2, 3, 4]),)
            # print(idxs)
            # [0 1 2 3 4]
            # print(idxs[0])
            # 得到数组的长度：5
            # print(len(idxs[0]))
            
            # 将 transform后的图片还原回去
            print(data['image'].shape)
            # img shape: 3680,4896,3
            img = np.array(255 * unnormalize(data['image'][0, :, :, :])).copy()
            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            # cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            # img = np.transpose(img,(2,0,1))
            print(img.shape)
            for j in range(idxs[0].shape[0]):
                # tensor([[279.6362, 65.1737, 417.0521, 224.3418],
                #         [466.5759, 28.8749, 563.4033, 159.8910],
                #         [696.0101, 7.5895, 764.2466, 72.8110]], device='cuda:0')
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_img.labels[int(classification[idxs[0][j]])]
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name,score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(img, (x1, y1, x2, y2), caption)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            print('Saving Result')
            cv2.imwrite('./result/{}.jpeg'.format(idx),img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

if __name__ == '__main__':
    main()