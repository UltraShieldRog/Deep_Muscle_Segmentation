DEBUG=True
def debug(s):
    if DEBUG:
        print(s)

import time
from ptsemseg.utils import *
import argparse
import fnmatch
import matplotlib.pyplot as plt
from os.path import join as pjoin
import csv


# folderpath = '/home/heng/Desktop/Research/isbi/fly-dataset/flyJanelia/'
# content = 'File\tPrecision\tRecall\tF1'#content format of the compareswc
# count = 0
# with open(folderpath+'datainfo/datainfo.txt') as f:
#     lines = f.readlines()#read every line
#     for item in lines:
#         if item.__contains__('.'):#escape the first line and recognize the path
#             filename=item.split()[0]
#             print(str(count) + ': ' + filename + ' is on processing')
#             count += 1
#             gt = folderpath + 'labels_v1/' + filename
#             pred = folderpath + 'pred_unet3dreg_regression/' + filename
#             gt = loadtiff3d(gt)
#             print('gt max: {}  gt min: {}'.format(gt.max(), gt.min))
#             pred = loadtiff3d(pred)
#             print('pred max: {}  pred min: {}'.format(pred.max(), pred.min()))
#             gt = (gt > 60).astype('int')
#             pred = (pred > 60).astype('int')
#             print(gt.shape)
#             print(pred.shape)
#             TP = np.sum(pred[gt == 1])
#             FP = np.sum(pred[gt == 0])
#             FN = np.sum(gt[pred == 0])
#             print('TP: {}; FP: {}; FN: {}'.format(TP, FP, FN))
#             precision = TP / (TP + FP)
#             recall = TP / (TP + FN)
#             f1 = (2 * (recall * precision)) / (recall + precision)
#             content = content + '\n' + filename + '\t%.2f\t%.2f\t%.2f' % (precision, recall, f1)
#
# lines = content.split('\n')
#
# with open(folderpath + '40compare_unet3dreg_trainV3.csv', "w") as csv_file:
#     writer = csv.writer(csv_file)
#     for line in lines:
#         writer.writerow([line])

def compute_scores(gt, pred, threshold=40):
    gt = (gt > 0).astype('int')
    pred = (pred > threshold).astype('int')
    # print(gt.shape)
    # print(pred.shape)
    TP = np.sum(pred[gt == 1])
    FP = np.sum(pred[gt == 0])
    FN = np.sum(gt[pred == 0])

    print('TP: {}; FP: {}; FN: {}'.format(TP, FP, FN))

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    # f1 = (2 * (recall * precision)) / (recall + precision)
    if np.isnan(precision):
        precision = 0
    if np.isnan(recall):
        recall = 0
    print('precision: {}, recall: {}'.format(precision, recall))
    return precision, recall,TP, FN, FP

def compare_with_gt(folderpath, runID, threshold=40):
    import csv

    content = 'File\tPrecision\tRecall\tF1'  # content format of the compareswc
    count = 0
    meta_file = pjoin(folderpath, 'meta.txt')
    with open(meta_file) as f:
        lines = f.read().splitlines()  # read every line
        for item in lines:
            if item.__contains__('.'):  # escape the first line and recognize the path
                filename = item.split('/')[-1]
                gt = item.replace('images', 'ground_truth_original')
                pred = item.replace('images', 'pred_'+runID)
                gt = loadtiff3d(gt)
                # print('gt max: {}  gt min: {}'.format(gt.max(), gt.min))
                pred = loadtiff3d(pred)
                # print('pred max: {}  pred min: {}'.format(pred.max(), pred.min()))
                scores = compute_scores(gt, pred, threshold)
                content = content + '\n' + filename + '\t%.2f\t%.2f\t%.2f' % (scores['precision'], scores['recall'], scores['f1'])

    lines = content.split('\n')

    with open(pjoin(folderpath, runID+'_compare.csv'), "w") as csv_file:
        writer = csv.writer(csv_file)
        for line in lines:
            writer.writerow([line])

def single_img_prediction(gt, pred):
    precision_list = list()
    recall_list = list()
    flag = 0
    for t in range(-10, 275, 5):
        print('threshold: {}'.format(t))
        if flag:
            precision = np.nan
            recall = np.nan
        else:
            precision, recall,TP, FN, FP = compute_scores(gt, pred, t)
        precision_list.append(precision)
        recall_list.append(recall)
        if TP <5 :
            flag = 1
            # break
    return precision_list, recall_list

def group_img_prediction(dataset_folder, runID):
    # debug('group_img_prediction: dataset {} runID {}'.format(dataset_folder, runID))
    from os.path import join as pjoin
    import numpy
    meta_file = pjoin(dataset_folder, 'meta_test.txt')
    precision_list = None
    recall_list = None
    flag = 0
    with open(meta_file) as f:
        lines = f.read().splitlines()  # read every line
        for item in lines:
            if item.__contains__('.'):  # escape the first line and recognize the path
                filename = item.split('/')[-1]
                gt = item.replace('test', 'ground_truth_original')
                pred = item.replace('test', 'pred_' + runID)
                gt = loadtiff3d(gt)
                # print('gt max: {}  gt min: {}'.format(gt.max(), gt.min))
                pred = loadtiff3d(pred)
                if precision_list is None:
                    precision_list, recall_list = single_img_prediction(gt, pred)
                    # debug('precision_list first: {}'.format(precision_list))
                else:
                    cur_presion, cur_recall = single_img_prediction(gt, pred)
                    precision_list = numpy.vstack((precision_list, cur_presion))
                    # debug('precision_list later: {}'.format(precision_list))
                    recall_list = numpy.vstack((recall_list, cur_recall))
                    flag = 1
    # print('precision_list shape {} '.format(precision_list.shape))
    if flag == 0:
        precision_mean = precision_list
        recall_mean = recall_list
    else:
        precision_mean = np.nanmean(precision_list, axis=0)
        recall_mean = np.nanmean(recall_list, axis=0)

    # debug('group_img_prediction: precision_mean {} recall_mean  {}'.format(precision_mean, recall_mean))
    return {'precision_mean': precision_mean, 'recall_mean': recall_mean}


def drawplot(xpoints=None, ypoints=None, label = 'line 1', draw=False):

    # plotting the line 1 points
    if not draw:
        plt.plot(xpoints, ypoints, label=label)
        return

    # naming the x axis
    plt.xlabel('Recall - axis')
    # naming the y axis
    plt.ylabel('Precision - axis')
    # giving a title to my graph
    plt.title('Precision-recall for different methods')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="params_compare")
    parser.add_argument(
        "--log_folder",
        nargs="?",
        type=str,
        default="/home/heng/Research/isbi/final_experiment",
        help="Log folder where all methods test model path stored"
    )
    parser.add_argument(
        "--dataset_folder",
        nargs="?",
        type=str,
        default="/home/heng/Research/isbi/fly-dataset/flyJanelia",
        help="Dataset folder used to compare different methods"
    )
    args = parser.parse_args()
    with open('/home/heng/Research/isbi/compare.txt', 'a') as f:
        f.write('\n\nground truth is original swc radius plus 3  ignore NAN\n')

    for f in os.listdir(args.log_folder):
        if fnmatch.fnmatch(f,'log_*.txt'):
            # method = f.split('log_')[-1].split('.txt')[0]
            log_file_path = pjoin(args.log_folder, f)
            # debug('Process method: {}'.format(method))
            with open(log_file_path) as log:
                lines = log.read().splitlines()
                for item in lines:
                    if item.__contains__(':'):
                        # if int(item.split(':')[0]) == 1:
                        # debug('We only check {} yml here'.format(item.split(':')[0]))
                        model_path = item.split(':')[-1]
                        yml_name = item.split('/')[-3]
                        yml_id = yml_name.split('_')[-1]
                        if yml_id == '1':
                            method = 'TR'
                        elif yml_id == '2':
                            method = 'SR'
                        elif yml_id == '3':
                            method = 'DSRI'
                        # elif yml_id == '4':
                        #     method = 'Student learning from teacher output only' # to change!!!!!!!
                        elif yml_id == '5':
                            method = 'UR'
                        # elif yml_id == '6':
                        #     method = 'UC'
                        # elif yml_id == '8':
                        #     method = 'student resiudal path only'
                        # elif yml_id == '9':
                        #     method = 'student resiudal path only with teacher output'
                        # elif yml_id == '10':
                        #     method = 'student resiudal path only with both outputs'
                        # model_number = model_path.split('_')[-1].split('.pkl')[0]
                        #
                        # if int(model_number) != 4:  # to change!!!!!
                        #     continue
                        # debug('We only check {} th model here'.format(model_number))
                        runID = model_path.split('/')[-2]
                        single_method_points = group_img_prediction(args.dataset_folder, runID)
                        # drawplot(single_method_points['precision_mean'], single_method_points['recall_mean'], label=method)
                        recall_mean = single_method_points['recall_mean']
                        precision_mean = single_method_points['precision_mean']
                        # debug(recall_mean)
                        # s1mask = np.isfinite(single_method_points['recall_mean'])
                        # recall_mean[s1mask] = 0
                        # s2mask = np.isfinite(single_method_points['precision_mean'])
                        # precision_mean[s2mask] = 0
                        # smask = np.logical_or(s1mask, s2mask)
                        debug('method: {} \n recall_mean: {}\nprecision_mean: {}'.format(method, recall_mean, precision_mean))
                        plt.plot(recall_mean, precision_mean , label=method)
                        with open('/home/heng/Research/isbi/compare.txt', 'a') as file:
                            file.write('{}: highest precision is {}, reached when threshold is {}; highest recall is {}, reached when threshold is {} \n'.format(
                                method, np.max(precision_mean), 5*np.argmax(precision_mean),
                                np.max(recall_mean),
                                5 * np.argmax(recall_mean)))

    # drawplot(draw=True)
    # naming the x axis
    plt.xlabel('Recall')
    # naming the y axis
    plt.ylabel('Precision')
    # giving a title to my graph
    plt.title('Precision-recall for different methods')

    # show a legend on the plot
    plt.legend()
    plt.rcParams.update({'font.size': 40})
    # function to show the plot
    plt.show()

