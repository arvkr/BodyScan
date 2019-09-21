import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, UnNormalizer, Normalizer
import model

print('CUDA available: {}'.format(torch.cuda.is_available()))


def bbox_extraction(file_list='./data/images2.csv'):
    weights_path = './models/csv_retinanet_25.pt'
    csv_classes = './classes.csv'

    dataset_val = CSVDataset(train_file=file_list, class_list= csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    # dataset_val = CSVDataset(train_file=file_list, class_list= csv_classes, transform=transforms.Compose([Normalizer()]))
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=False)
    retinanet.load_state_dict(torch.load(weights_path))

    use_gpu = True
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if use_gpu:
        retinanet = retinanet.to(device)

    retinanet.eval()

    unnormalize = UnNormalizer()

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            scores, classification, transformed_anchors = retinanet(data['img'].to(device).float())

            def get_bbox(classification, transformed_anchors, label=0):
                bbox = {}
                idx = np.where(classification == label)[0][0]
                co_ord = transformed_anchors[idx, :]
                bbox['x1'] = int(co_ord[0])
                bbox['y1'] = int(co_ord[1])
                bbox['x2'] = int(co_ord[2])
                bbox['y2'] = int(co_ord[3])

                return bbox

            scores = scores.cpu().numpy()
            classification = classification.cpu().numpy()
            transformed_anchors = transformed_anchors.cpu().numpy()
            # print('scores:',scores)
            # print('classification:', classification)
            # print('transformed_anchors', transformed_anchors)
            bbox = {}
            bbox['neck'] = get_bbox(classification, transformed_anchors, label=0)
            bbox['stomach'] = get_bbox(classification, transformed_anchors, label = 1)

            # print('neck',bbox['neck'] )
            # print('stomach',bbox['stomach'] )


            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            cv2.rectangle(img,(bbox['neck']['x1'], bbox['neck']['y1']), (bbox['neck']['x2'], bbox['neck']['y2']), color=(0, 0, 255), thickness=2)
            cv2.rectangle(img,(bbox['stomach']['x1'], bbox['stomach']['y1']),(bbox['stomach']['x2'], bbox['stomach']['y2']), color=(0, 0, 255), thickness=2)

            # cv2.imshow('img', img)
            # cv2.imwrite('./sample_11.jpg',img)
            # cv2.waitKey(0)

            return bbox

# bbox_extraction()

# if __name__ == '__main__':
#  main()