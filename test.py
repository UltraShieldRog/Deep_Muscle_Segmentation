DEBUG=True
def log(s):
    if DEBUG:
        print(s)


import torch
# import visdom
import argparse
import yaml
import cv2
import nibabel

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore, averageMeter
from scipy.ndimage.interpolation import zoom
from torch.utils import data

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

    test_prefix = data_path + '/test/img/'
    gt_prefix = data_path + '/test/lbl/'
    running_metrics_val = runningScore(2, len(os.listdir(test_prefix)))
    create_folder(data_path + '/test/', cfg['model']['arch'])
    print('model: ', cfg['model']['arch'])

    # if cfg['test']['nifti']:
    #     test_path = cfg['test']['nifti_in']

    # else:
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
        if cfg['training']['n_classes'] == 2:
            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        elif cfg['training']['n_classes'] == 1:
            pred = np.squeeze(outputs.data.cpu().numpy(), axis=0)
            pred = np.squeeze(pred, axis=0)
            # print(pred.shape, np.unique(pred))
            pred /= np.max(pred)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            # print(np.unique(pred))
            pred = np.array(pred, dtype=np.int64)
            print(type(pred), pred.shape, lbl.shape)
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
