import argparse
import torch
import re
import torch.nn as nn
import glob
import numpy as np
import torch
import os


DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_disp(args):
    test_path = args.data_path
    dirs = args.data_dirs
    print(dirs)
    # dirs = ['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742']
    disp_list = {}
    for dir in dirs:
        npy_path = test_path + r"/" + dir + r"/" + "metric3d/npy/data"
        npy_file = glob.glob(npy_path + r"/*.npy")
        # print(len(npy_file))
        disp_dir = {}
        for file in npy_file:
            key = file.split('/')[-1].replace("RGBResize_metric3d.npy", "Keypoint.txt")
            disp = np.load(file)
            disp_dir[key] = 1 / disp
        disp_list[test_path + r"/" + dir] = disp_dir
    print(len(disp_list[test_path + r"/" + dir]))

    return disp_list

def find_scale(disp_list):
    ans = [[],[],[],[],[],[],[],[]]
    disp_pred = []
    disp_gt = []
    with torch.no_grad():
        for file, disp in disp_list.items():
            for key, value in disp.items():

                disp_value = value

                f = open(file + '/Keypoint/' + key, 'r')
                gts = f.readlines()
                f.close()
                for gt in gts:
                    x, y, d, c = gt.split()
                    x = round(float(x) * 582) - 1
                    x = int(max(0,min(582, x)))
                    y = round(float(y) * 429) - 1
                    y = int(max(0, min(429, y)))
                    d = float(d) * 582
                    p = max(0, disp_value[y, x])
                    disp_pred.append(p)
                    disp_gt.append(d)
    disp_pred = np.array(disp_pred)
    disp_gt = np.array(disp_gt)
    scale = np.sum(disp_pred * disp_gt) / np.sum(disp_pred * disp_pred)
    mae = np.abs(scale * disp_pred - disp_gt).mean()
    rmse = pow(np.mean((scale * disp_pred - disp_gt)**2), 0.5)
    return scale, mae, rmse

def find_scale_offset(disp_list):
    disp_pred = []
    disp_gt = []
    with torch.no_grad():
        for file, disp in disp_list.items():
            for key, value in disp.items():

                disp_value = value

                f = open(file + '/Keypoint/' + key, 'r')
                gts = f.readlines()
                f.close()
                for gt in gts:
                    x, y, d, c = gt.split()
                    x = round(float(x) * 582) - 1
                    x = int(max(0,min(582, x)))
                    y = round(float(y) * 429) - 1
                    y = int(max(0, min(429, y)))
                    d = float(d) * 582
                    p = max(0, disp_value[y, x])
                    disp_pred.append(p)
                    disp_gt.append(d)
    disp_pred = np.array(disp_pred)
    disp_gt = np.array(disp_gt)
    X = np.vstack([disp_pred, np.ones(len(disp_pred))]).T
    Y = disp_gt.reshape(-1, 1)
    P = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    scale = P[0][0]
    offset = P[1][0]
    mae = np.abs(scale * disp_pred + offset - disp_gt).mean()
    rmse = pow(np.mean((scale * disp_pred + offset - disp_gt)**2), 0.5)
    return scale, offset, mae, rmse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', help="directory to load test datasets", default="/media/jiayu/My Passport1/academic/depth_estimation/datasets/RGB-NIR_stereo/tzhi/RGBNIRStereoRelease/rgbnir_stereo/data")
    parser.add_argument('--data_dirs', nargs='+', type=str, help="disp scale for midas", default=['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742'])
    parser.add_argument('--scale', action='store_true', help="if set estimates scale factor only")

    args = parser.parse_args()

    disp_list = load_disp(args)
    if not args.scale:
        scale, offset, mae, rmse = find_scale_offset(disp_list)
        print("offset: ", offset)
    else:
        scale, mae, rmse = find_scale(disp_list)
    print("scale: ", scale)
    print("mae: ", mae)
    print("rmse: ", rmse)