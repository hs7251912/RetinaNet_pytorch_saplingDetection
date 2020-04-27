from data_loader import CSV_Dataset, Normalizer, Resizer, AspectRatioBasedSampler, collater
from torchvision import transforms
from Build_Network import RetinaNet
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from others import compute_overlap, _compute_ap
import os
from datetime import datetime
# python3
datetime.now().timestamp()

os.environ["CUDA_VISIBLE_DEVICES"]='1'

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

def run_the_net(train_file, class_list, test_file=None):
    print('loading data')
    # 将输入的图片经过正则化Normalizer()和缩放Resizer()
    dataset_train = CSV_Dataset(train_file, class_list, transform=transforms.Compose([Normalizer(),  Resizer()]))
    # AspectRatioBasedSampler:创建一个批次大小相同的图片
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=4, drop_last=False)
    # 迭代器
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    # 加载test dataset
    if test_file is None:
        print('test file not given')
        return
    else:
        print('loading test data')
        dataset_test = CSV_Dataset(test_file, class_list, transform=transforms.Compose([Normalizer(), Resizer()]))

    print('initializing retinanet')
    # Create the model
    model = RetinaNet(dataset_train.num_classes())
    if torch.cuda.is_available():
            model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # 网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    model.training = True
    best_loss = float('inf')

    print('Num training images: {}'.format(len(dataset_train)))
    print('training')
    # 定义迭代次数为100
    for epoch_num in range(100):
        model.train()
        average_loss = 0.0

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            # RetinaNet在training模式下返回losses.Calcu_Loss
            classification_loss, regression_loss = model([data['image'].cuda().float(), data['annotations']])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            average_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss.append(float(loss))

        print(
            'Epoch: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, float(classification_loss), float(regression_loss), average_loss / dataset_train.__len__().__float__()))

        # print('the average loss in no.%d training:' % (epoch_num+1), average_loss / dataset_train.__len__().__float__())
        # if average_loss<best_loss:
        #     print('Saving..')
        scheduler.step(np.mean(epoch_loss))
        torch.save(model.state_dict(), './checkpoint/BlockDectection_retinanet_{}_loss_{}'.format(epoch_num,average_loss / dataset_train.__len__().__float__()))


            # torch.save(model.state_dict(), './checkpoint/'+'/epoch'+str(epoch_num+1)+'-'+str("%.4f" % average_loss)+'.pth')
            # best_loss = average_loss

    print('initializing paremeters')
    scores_threshold = 0.05
    max_boxes_per_image = 100
    iou_threshold = 0.5

    print('Evaluating dataset')
    # 转换为测试模式
    model.training = False
    model.eval()

    # 使用生成器生成retinanet的检测结果
    # 检测结果List的size是：
    # all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]


    all_detections = [[None for i in range(dataset_test.num_classes())] for j in range(len(dataset_test))]
    all_annotations = [[None for i in range(dataset_test.num_classes())] for j in range(len(dataset_test))]

    for image_id, data in enumerate(dataset_test):
        """
        part 1:
            get detections and adjust them
        """
        # 获得检测的scores,classes和boxes
        scores, classes, boxes = model(data['image'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
        # 获得调整图片大小的scale
        scale = data['scale']
        #检测物体的分类得分
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
            image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # copy detections to all_detections
            for label in range(dataset_test.num_classes()):
                all_detections[image_id][label] = image_detections[image_detections[:, -1] == label, :-1]
        else:
        # copy detections to all_detections
            for label in range(dataset_test.num_classes()):
                all_detections[image_id][label] = np.zeros((0, 5))


        """
        part 2:
            get annotations
        """
        annotation = dataset_test.read_annotations(image_id)
        for label in range(dataset_test.num_classes()):
            all_annotations[image_id][label] = annotation[annotation[:, 4] == label, :4].copy()


    """
    part 3:
    calculate mAP
    """
    average_precisions = {}

    for label in range(dataset_test.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(dataset_test)):
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
    for label in range(dataset_test.num_classes()):
        label_name = dataset_test.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))

    torch.save(model.state_dict(), './checkpoint/model_final.pt')


if __name__ == '__main__':
    run_the_net('./data/SaplingDetection/StepCroped_anno/train.csv', './data/SaplingDetection/StepCroped_anno/classes.csv', './data/SaplingDetection/StepCroped_anno/test.csv')
