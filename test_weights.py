import numpy as np
import os
import cv2
from data_loader import CSV_Dataset, UnNormalizer, Normalizer, Resizer, AspectRatioBasedSampler, collater
from Build_Network import RetinaNet
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from others import compute_overlap, _compute_ap
os.environ["CUDA_VISIBLE_DEVICES"]='1'


val_file = './data/SaplingDetection/StepCroped_anno/val.csv'
class_list = './data/SaplingDetection/StepCroped_anno/classes.csv'
# 读取
# CSV_Dataset:将annotation_file和classes_file中的内容加入CSV_Dataset的属性中，例如：self.classes,self.image_data,self.image_names
dataset_val = CSV_Dataset(val_file, class_list, transform=transforms.Compose([Normalizer(), Resizer()]))
sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
# 该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，
# 后续只需要再包装成Variable即可作为模型的输入
dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

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

all_detections = [[None for i in range(dataset_val.num_classes())] for j in range(len(dataset_val))]
all_annotations = [[None for i in range(dataset_val.num_classes())] for j in range(len(dataset_val))]

for image_id, data in enumerate(dataset_val):
    """
    part 1:
        get detections and adjust them
    """
    # 获得检测的scores,classes和boxes
    scores, classes, boxes = model(data['image'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
    # 获得调整图片大小的scale
    scale = data['scale']
    # 检测物体的分类得分
    scores = scores.cpu().detach().numpy()
    # 被检测物体的分类
    classes = classes.cpu().detach().numpy()
    # 检测框的坐标
    boxes = boxes.cpu().detach().numpy()
    # 调整检测框的大小
    boxes = boxes / scale
    # 有效的检测框
    valid_scores_indices = np.where(scores > scores_threshold)[0]
    if valid_scores_indices.shape[0] > 0:
        scores = scores[valid_scores_indices]

        scores_sort = np.argsort(-scores)[:max_boxes_per_image]

        # select detections
        image_boxes = boxes[valid_scores_indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_labels = classes[valid_scores_indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # copy detections to all_detections
        for label in range(dataset_val.num_classes()):
            all_detections[image_id][label] = image_detections[image_detections[:, -1] == label, :-1]
    else:
        # copy detections to all_detections
        for label in range(dataset_val.num_classes()):
            all_detections[image_id][label] = np.zeros((0, 5))

    """
    part 2:
        get annotations
    """
    annotation = dataset_val.read_annotations(image_id)
    for label in range(dataset_val.num_classes()):
        all_annotations[image_id][label] = annotation[annotation[:, 4] == label, :4].copy()

"""
part 3:
calculate mAP
"""
average_precisions = {}

for label in range(dataset_val.num_classes()):
    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    scores = np.zeros((0,))
    num_annotations = 0.0

    for i in range(len(dataset_val)):
        detections = all_detections[i][label]
        annotations = all_annotations[i][label]
        num_annotations += annotations.shape[0]
        detected_annotations = []

        for d in detections:
            scores = np.append(scores, d[4])

            if annotations.shape[0] == 0:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)
                continue

            overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
            else:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)

    # no annotations -> AP for this class is 0
    if num_annotations == 0:
        average_precisions[label] = 0, 0
        continue

    # sort by score
    indices = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = _compute_ap(recall, precision)
    average_precisions[label] = average_precision, num_annotations
print('\nmAP:')
for label in range(dataset_val.num_classes()):
    label_name = dataset_val.label_to_name(label)
    print('{}: {}'.format(label_name, average_precisions[label][0]))