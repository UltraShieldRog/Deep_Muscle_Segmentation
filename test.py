DEBUG=True
def log(s):
    if DEBUG:
        print(s)


import torch
# import visdom
import argparse
import yaml
import cv2
import nibabel as nib

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore, averageMeter
from scipy.ndimage.interpolation import zoom
from torch.utils import data
import timeit
import time

from ptsemseg.utils import *


def test(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']
    patch_size = [256,256]

    log('n_classes is: {}'.format(cfg['training']['n_classes']))
    model = get_model(cfg['model'], cfg['training']['n_classes'])
    state = convert_state_dict(torch.load(cfg['test']['cp_path'])["model_state"])
    model.load_state_dict(state)

    model.eval()
    model.to(device)

    running_metrics_val = runningScore(2)
    # test_prefix = data_path + '/test/img/'
    # gt_prefix = data_path + '/test/lbl/'
    # running_metrics_val = runningScore(2, len(os.listdir(test_prefix)))
    create_folder(data_path + '/test/', cfg['model']['arch'])
    print('model: ', cfg['model']['arch'])

    if cfg['test']['nifti']:
        start = time.process_time()
        test_path = cfg['test']['nifti']['nifti_in']
        out_path = cfg['test']['nifti']['nifti_out']
        for t in os.listdir(test_path):
            roi_img = nib.load(os.path.join(test_path, t))
            roi_dat = roi_img.get_data()
            pred_roi = np.zeros(roi_dat.shape)
            roi_aff = roi_img.affine
            roi_hdr = roi_img.header

            slice_no = roi_dat.shape[2]
            for s in range(slice_no):
                roi_slice = roi_dat[:, :, s]
                if len(np.unique(roi_slice)) <= 1:
                    continue
                padded_slice, pad_info = pad_slice(roi_slice)
                padded_slice = padded_slice.astype(np.float64)
                padded_slice /= 255.0
                padded_slice = np.expand_dims(padded_slice, 0)
                # padded_slice = padded_slice.transpose(2, 0, 1)
                temp = padded_slice.copy()
                padded_slice = np.concatenate((padded_slice, temp), axis=0)
                padded_slice = np.concatenate((padded_slice, temp), axis=0)
                padded_slice = np.expand_dims(padded_slice, 0)
                padded_slice = torch.from_numpy(padded_slice).float()
                padded_slice = padded_slice.to(device)

                outputs = model(padded_slice)
                pred_slice = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

                result_slice = unpad_slice(pred_slice, pad_info)
                pred_roi[:, :, s] = result_slice

            pred_nii = nib.Nifti1Image(pred_roi, roi_aff, header=roi_hdr)
            nib.save(pred_nii, os.path.join(out_path, t.rstrip('.nii.gz') + '_' + cfg['model']['arch'] + '.nii.gz'))
        end = time.process_time()
        print(end-start)
    else:
        test_prefix = data_path + '/test/img/'
        gt_prefix = data_path + '/test/lbl/'

        for t in os.listdir(test_prefix):
            img = cv2.imread(test_prefix + t)
            lbl = cv2.imread(gt_prefix + t.rstrip('.png') + '_lbl.png', 0)
            lbl = np.array(lbl, dtype=np.uint8) // 255
            img = img.astype(np.float64)
            img /= 255.0
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()
            img = img.to(device)
            outputs = model(img)
            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

            running_metrics_val.update(lbl, pred)
            # print(np.unique(pred))

            # print(data_path + '/test/' + cfg['model']['arch'] + '/' + t)
            if cfg['test']['save']:
                cv2.imwrite(data_path + '/test/' + cfg['model']['arch'] + '/' + t, pred*255)
            img = cv2.imread(test_prefix + t)
            lbl = cv2.imread(gt_prefix + t.rstrip('.png') + '_lbl.png', 0)
            lbl = np.array(lbl, dtype=np.uint8) // 255


    score = running_metrics_val.get_scores()
    print(score)


def create_folder(path, model_name):
    if os.path.exists(path+model_name):
        return
    else:
        os.mkdir(path+model_name)


def norm_slice(img):
    img = img.astype(np.float)
    img *= 1/img.max()*255.0
    return img


def pad_bg_x(img, size=256):
    x_left = 0
    x_right = 0
    x, y = img.shape
    if x < size:
        padded_x = size - x
        x_left = int(padded_x / 2)
        x_right = padded_x - x_left
        return [x_left, x_right]

    if x > size:
        cut_x = x - size
        x_left = int(cut_x / 2)
        x_right = cut_x - x_left
        return [-x_left, -x_right]

    return [x_left, x_right]


def pad_bg_y(img, size=256):
    y_up = 0
    y_down = 0
    x, y = img.shape

    if y < size:
        padded_y = size - y
        y_up = int(padded_y / 2)
        y_down = padded_y - y_up
        return [y_up, y_down]

    if y > size:
        cut_y = y - size
        y_up = int(cut_y / 2)
        y_down = cut_y - y_up
        return [-y_up, -y_down]

    return [y_up, y_down]


def pad_slice(img_slice):
    img_slice = norm_slice(img_slice)
    # print('original: ', img_slice.shape)
    img_slice = np.rot90(img_slice)


    # ---- pad to 256*256 ----
    padded_img = img_slice.copy()
    pad_left, pad_right = pad_bg_x(img_slice)
    # print('pad left/right: ', pad_left, pad_right)
    if pad_left >= 0 and pad_right >= 0:
        padded_img = np.pad(padded_img, ((pad_left, pad_right), (0, 0)), mode='constant')
    elif pad_left <= 0 and pad_right <= 0:
        cut_left = -pad_left
        cut_right = -pad_right
        x_len = padded_img.shape[0]
        padded_img = padded_img[cut_left:x_len - cut_right, ...]
        cut_img_x_left = padded_img[:cut_left, ...]
        cut_img_x_right = padded_img[x_len - cut_right:, ...]
    #         padded_img = padded_img[x-cut_x:,...]
    # print('after pad x: ', padded_img.shape)

    pad_up, pad_down = pad_bg_y(img_slice)
    # print('pad up/down: ', pad_up, pad_down)
    if pad_up >= 0 and pad_down >= 0:
        padded_img = np.pad(padded_img, ((0, 0), (pad_up, pad_down)), mode='constant')
    elif pad_up <= 0 and pad_down <= 0:
        cut_up = -pad_up
        cut_down = -pad_down
        y_len = padded_img.shape[1]
        padded_img = padded_img[..., cut_up:y_len - cut_down]
        cut_img_y_up = padded_img[..., :cut_up]
        cut_img_y_down = padded_img[..., y_len - cut_down:]
    # print('after pad y: ', padded_img.shape)
    return padded_img, [pad_left, pad_right, pad_up, pad_down]


def unpad_slice(pred_slice, pad_info):

        pad_left, pad_right, pad_up, pad_down = pad_info

        unpadded_slice = pred_slice.copy()
        size_x, size_y = pred_slice.shape
        if pad_up >= 0 and pad_down >= 0:
            unpadded_slice = unpadded_slice[..., pad_up:size_y - pad_down]
        elif pad_up <= 0 and pad_down <= 0:
            cut_up = -pad_up
            cut_down = -pad_down
            unpadded_slice = np.pad(unpadded_slice, ((0, 0), (cut_up, cut_down)), mode='constant')

        # print('after unpadded y: ', unpadded_slice.shape)

        if pad_left >= 0 and pad_right >= 0:
            unpadded_slice = unpadded_slice[pad_left:size_x - pad_right, ...]
        elif pad_left <= 0 and pad_right <= 0:
            cut_left = -pad_left
            cut_right = -pad_right
            unpadded_slice = np.pad(unpadded_slice, ((cut_left, cut_right), (0, 0)), mode='constant')

        # print('after unpadded y: ', unpadded_slice.shape)
        unpadded_slice = np.rot90(unpadded_slice, 3)
        # print('original: ', unpadded_slice.shape)

        return unpadded_slice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_test.yml",
        help="Configuration file to use",
    )

    # parser.add_argument(
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    test(cfg)
