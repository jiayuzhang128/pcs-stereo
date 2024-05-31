import re
import os
import cv2
import glob
import argparse
import numpy as np

def disp2im(disp, max=16.0):
    im = np.clip(np.abs(disp) / disp.max(), 0, 1)
    return im
    
def npy2im_midas(args):
    data_path = args.data_directory
    dirs = args.data_dirs
    # dirs = ['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742']
    disp_list = {}
    for dir in dirs:
        pfm_path = data_path + r"/" + dir + r"/" + "metric3d/npy/data"
        os.makedirs(pfm_path.replace('/npy/data', '') + r"/png", exist_ok=True)
        pfm_file = sorted(glob.glob(pfm_path + r"/*.npy"))
        disp_dir = {}
        for pfm in pfm_file:
            png_file = pfm.split('/')[-1].replace(".npy", ".png")
            midas_disp = 1 / np.load(pfm)
            midas_disp = np.float32(midas_disp * args.disp_scale + args.disp_offset)
            midas_disp_im = disp2im(midas_disp)
            midas_disp_im_color = cv2.applyColorMap(np.uint8(midas_disp_im * 255), cv2.COLORMAP_INFERNO)
            cv2.imwrite(pfm_path.replace('/npy/data', '') + r"/png/" + png_file, midas_disp_im_color)
            
def npy2im(input, output):

    os.makedirs(output,exist_ok=True)
    npy_file = sorted(glob.glob(input + r"/*.npy"))
    print(len(npy_file))
    for npy in npy_file:
        png_file = npy.split('/')[-1].replace(".npy", ".png")
        disp = np.load(npy)
        disp = np.float32(disp)
        disp_im = disp2im(disp)
        disp_im_color = cv2.applyColorMap(np.uint8(disp_im * 255), cv2.COLORMAP_INFERNO)
        cv2.imwrite(output + r"/" + png_file, disp_im_color)

# def npy2im(input, output):

#     os.makedirs(output,exist_ok=True)
#     npy_file = sorted(glob.glob(input + r"/*.npy"))
#     print(len(npy_file))
#     for npy in npy_file:
#         png_file = npy.split('/')[-1].replace(".npy", ".png")
#         disp = np.load(npy)
#         disp = 46.63464318292732 / np.float32(disp) + 0.03591201880716065
#         disp_im = disp2im(disp)
#         disp_im_color = cv2.applyColorMap(np.uint8(disp_im * 255), cv2.COLORMAP_INFERNO)
#         cv2.imwrite(output + r"/" + png_file, disp_im_color)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', help="directory to load test datasets", default="/media/jiayu/share/win10data/projects/rgbms/tmp/igev_eval/eval/npy")
    parser.add_argument('--output_directory', help="directory to load test datasets", default="/media/jiayu/My Passport1/academic/FMCS-Stereo/MiDaS/input")
    parser.add_argument('--disp_scale', type = float, help="disp scale for metric3d", default=46.63464318292732)
    parser.add_argument('--disp_offset', type = float, help="disp offset for metric3d", default=0.03591201880716065)
    parser.add_argument('--data_dirs', nargs='+', type=str, help="choose dir to process", default=['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742'])

    args = parser.parse_args()

    # npy2im_midas(args)
    npy2im(args.data_directory, args.output_directory)


