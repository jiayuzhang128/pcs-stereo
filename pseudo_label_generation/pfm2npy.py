import re
import os
import cv2
import glob
import argparse
import numpy as np

def disp2im(disp, max=16.0):
    im = np.clip(np.abs(disp) / max, 0, 1)
    return im

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
        data = np.float32(np.flipud(data))

        return data, scale
    
def pfm2npy(args):
    data_path = args.data_directory
    dirs = args.data_dirs
    # dirs = ['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742']
    disp_list = {}
    for dir in dirs:
        pfm_path = data_path + r"/" + dir + r"/" + "MonoDisp"
        os.makedirs(pfm_path + r"/npy", exist_ok=True)
        os.makedirs(pfm_path + r"/png", exist_ok=True)
        pfm_file = sorted(glob.glob(pfm_path + r"/*.pfm"))
        disp_dir = {}
        for pfm in pfm_file:
            # print(pfm, npy)
            npy_file = pfm.split('/')[-1].replace(".pfm", ".npy")
            png_file = pfm.split('/')[-1].replace(".pfm", ".png")
            # print(png_file)
            midas_disp, _ = read_pfm(pfm)
            midas_disp = np.float32(midas_disp * args.disp_scale + args.disp_offset)
            np.save(pfm_path + r"/npy/" + npy_file, midas_disp)
            midas_disp_im = disp2im(midas_disp)
            midas_disp_im_color = cv2.applyColorMap(np.uint8(midas_disp_im * 255), cv2.COLORMAP_INFERNO)
            cv2.imwrite(pfm_path + r"/png/" + png_file, midas_disp_im_color)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', help="directory to load test datasets", default="./datasets/test")
    parser.add_argument('--disp_scale', type = float, help="disp scale for midas", default=0.0006504932339649878)
    parser.add_argument('--disp_offset', type = float, help="disp offset for midas", default=0.7388445603651852)
    parser.add_argument('--data_dirs', nargs='+', type=str, help="choose dir to process", default=['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742'])

    args = parser.parse_args()

    pfm2npy(args)


