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

def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale

def load_disp(args):
    output_path = args.pfm_directory
    test_path = args.test_directory
    dirs = args.test_dirs
    # dirs = ['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742']
    disp_list = {}
    for dir in dirs:
        pfm_path = test_path + r"/" + dir + r"/" + "MonoDisp"
        pfm_file = glob.glob(pfm_path + r"/*.pfm")
        # print(len(npy_file))
        disp_dir = {}
        for file in pfm_file:
            key = file.split('/')[-1].replace("RGBResize-dpt_beit_large_512.pfm", "Keypoint.txt")
            disp, _ = read_pfm(file)
            # disp_dir[key] = disp * 0.0006898412
            disp_dir[key] = disp * args.disp_scale + args.disp_offset
        disp_list[test_path + r"/" + dir] = disp_dir
    print(len(disp_list[test_path + r"/" + dir]))

    return disp_list

def evaluate(args, disp_list):
    count = 0
    count_all = 0
    ans = [[],[],[],[],[],[],[],[]]
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
                    c = int(c)
                    p = max(0, disp_value[y, x])
                    count_all += 1
                    # 剔除异常点
                    if (p-d)*(p-d) > args.exclude:
                        continue
                    ans[c].append((p-d)*(p-d))
                    count += 1

        rmse = []
        for c in range(8):
            rmse.append(pow(sum(ans[c]) / len(ans[c]), 0.5))
        print('Common    Light     Glass     Glossy  Vegetation   Skin    Clothing    Bag       Mean')
        print(round(rmse[0], 4), '  ', round(rmse[1], 4), '  ', round(rmse[2], 4), '  ', round(rmse[3], 4), '  ', round(rmse[4], 4), '  ', round(rmse[5], 4), '  ', round(rmse[6], 4), '  ', round(rmse[7], 4), '  ', round(sum(rmse) / 8.0, 4))
        print("总点数：", count_all)
        print("剔除点数：", count_all - count)
        print("剩余点数：", count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_directory', help="directory to load test datasets", default="./datasets/test")
    parser.add_argument('--pfm_directory', help="directory to load pfm files", default="./output")
    parser.add_argument('--disp_scale', type = float, help="disp scale for midas", default=0.0006504932339649878)
    parser.add_argument('--disp_offset', type = float, help="disp offset for midas", default=0.7388445603651852)
    parser.add_argument('--exclude', type = float, help="disp scale for midas", default=2)
    parser.add_argument('--test_dirs', nargs='+', type=str, help="disp scale for midas", default=['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742']
)

    args = parser.parse_args()

    disp_list = load_disp(args)
    evaluate(args, disp_list)