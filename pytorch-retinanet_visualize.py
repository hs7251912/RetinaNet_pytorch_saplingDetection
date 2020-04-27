import numpy as np
import torchvision
import os
import copy
# Test file
import time
import sys
import cv2
from skimage.io import imsave
from data_loader import CSV_Dataset, UnNormalizer, Normalizer, Resizer, AspectRatioBasedSampler, collater
from Build_Network import RetinaNet
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from others import compute_overlap, _compute_ap
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def main():
    val_file = './data/SaplingDetection/StepCroped_anno/val.csv'
    class_list = './data/SaplingDetection/StepCroped_anno/classes.csv'
    # 读取
    # CSV_Dataset:将annotation_file和classes_file中的内容加入CSV_Dataset的属性中，例如：self.classes,self.image_data,self.image_names
    dataset_val = CSV_Dataset(val_file, class_list, transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    # 该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，
    # 后续只需要再包装成Variable即可作为模型的输入
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    if not os.path.exists('result'):
        os.mkdir('result')
    # 加载模型
    model_path = './checkpoint/sapling_best.pt'
    # 初始化模型
    model = RetinaNet(dataset_val.num_classes())
    # 将参数转换为cuda型
    if torch.cuda.is_available():
        model = model.cuda()
    # 加载模型
    model.load_state_dict(torch.load(model_path))

    print('initializing paremeters')
    scores_threshold = 0.05
    max_boxes_per_image = 100
    iou_threshold = 0.5

    model.training = False
    model.eval()
    unnormalize = UnNormalizer()

    # Draws a caption above the box in an image
    def draw_caption(image, box, caption):
        # print('Box is: {}'.format(box))
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx,data in enumerate(dataloader_val):
        # with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
        with torch.no_grad():
            # st = time.time()
            print((data['image'].cuda().float()).shape)
            scores, classification, transformed_anchors = model(data['image'].cuda().float())
            # print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            # tensor([1.0000, 1.0000, 1.0000, 0.8962, 0.0568, 0.0527], device='cuda:0')
            # print(scores)
            # (array([0, 1, 2, 3, 4]),)
            # print(idxs)
            # [0 1 2 3 4]
            # print(idxs[0])
            # 得到数组的长度：5
            # print(len(idxs[0]))
            img = np.array(255 * unnormalize(data['image'][0, :, :, :])).copy()
            img[img < 0] = 0
            img[img > 255] = 255
            print(img.shape)

            img = np.transpose(img, (1, 2, 0))
            print(img.shape)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                # tensor([[279.6362, 65.1737, 417.0521, 224.3418],
                #         [466.5759, 28.8749, 563.4033, 159.8910],
                #         [696.0101, 7.5895, 764.2466, 72.8110]], device='cuda:0')
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
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