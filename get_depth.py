import torch
from options.train_options import TrainOptions
from loaders import aligned_data_loader
from models import pix2pix_model

BATCH_SIZE = 1

def depth_predict(image_pathfile = 'data/image.txt'):
    opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    video_data_loader = aligned_data_loader.DAVISDataLoader(image_pathfile, BATCH_SIZE)
    video_dataset = video_data_loader.load_data()
    model = pix2pix_model.Pix2PixModel(opt)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    model.switch_to_eval()
    save_path = 'data/outputs/'
    for i, data in enumerate(video_dataset):
        print(i)
        stacked_img = data[0]
        targets = data[1]
        pred_d_m = model.run_and_save_DAVIS(stacked_img, targets, save_path)

    return pred_d_m

depth = depth_predict()