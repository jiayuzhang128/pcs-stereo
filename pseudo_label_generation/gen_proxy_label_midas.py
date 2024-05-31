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

def distance(x, d_min=0.01, d_max=30, delta_d_min=0.1, delta_d_max=0.5):
    """计算距离置信度
    Args:
        x : 输入
        d_min (_type_): 可信深度区间左边界
        d_max (_type_): 可信深度区间右边界
        delta_d_min (_type_): 系数
        delta_d_max (_type_): 系数
    """

    if x < d_min:
        result = np.exp(-np.abs(x - d_min)**2/(2*delta_d_min**2))
    elif x > d_max:
        result = np.exp(-np.abs(x - d_max)**2/(2*delta_d_max**2))
    else:
        result = 1
    return result

def get_distance_confidence(mono2stereo_disp, d_min=0.01, d_max=30):
    """获取距离置信度，及其掩码

    Args:
        mono2stereo_disp (_type_): 单目转双目视差图
    """
    w_d = np.vectorize(distance)(mono2stereo_disp, d_min, d_max)
    mask = np.where(w_d < 0.5, 0, 1)
    return w_d, mask

def gradient(x, delta_g=0.5):
    """计算梯度置信度"""
    return np.exp(-x**2/(2*delta_g**2))
    
def get_gradient_confidence(mono2stereo_disp, mono2stereo_disp_reverse):
    """获取梯度置信度，及其掩码

    Args:
        mono2stereo_disp (_type_): 单目转双目视差图
        mono2stereo_disp_reverse (_type_): 单目转双目逆视差图
    """
    gradient_magnitudes = []
    gradient_magnitudes_reverse = []
    sobel_x = cv2.Sobel(mono2stereo_disp, cv2.CV_64F, 1, 0, ksize=3)  # 在x方向上计算梯度
    sobel_y = cv2.Sobel(mono2stereo_disp, cv2.CV_64F, 0, 1, ksize=3)  # 在y方向上计算梯度
    sobel_x_reverse = cv2.Sobel(mono2stereo_disp_reverse, cv2.CV_64F, 1, 0, ksize=3)  # 在x方向上计算梯度
    sobel_y_reverse = cv2.Sobel(mono2stereo_disp_reverse, cv2.CV_64F, 0, 1, ksize=3)  # 在y方向上计算梯度
    # 计算梯度的幅值和方向
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    gradient_magnitude_reverse = np.sqrt(np.square(sobel_x_reverse) + np.square(sobel_y_reverse))

    w_g_forward = np.vectorize(gradient)(gradient_magnitude)
    w_g_reverse = np.vectorize(gradient)(gradient_magnitude_reverse)
    w_g = w_g_forward * w_g_reverse
    mask = np.where(w_g < 0.5, 0, 1)

    return w_g, mask

def generate_confidence_map(disp, d_min=0.01, d_max=30):
    disp_reverse = 1 / disp
    w_d, mask_d = get_distance_confidence(disp, d_min, d_max)
    w_g, mask_g = get_gradient_confidence(disp, disp_reverse)
    confidence_map = w_d * w_g
    
    return confidence_map, np.float32(mask_d), np.float32(mask_g)
        
def generate_consistency_mask(mono2stereo_disp, stereo_disp, t1=1.0, t2=1.0):
    """Generate a mask of where the stereo disp is wrong, based on the difference between the stereo network and the teacher, monocular network"""

    mono2stereo_disp = np.abs(mono2stereo_disp)
    stereo_disp = np.abs(stereo_disp)
        
    # mask where they differ by a large amount
    mask = ((stereo_disp - mono2stereo_disp) / mono2stereo_disp) < t1
    mask *= ((mono2stereo_disp - stereo_disp) / stereo_disp) < t2

    return np.float32(mask), 1 - np.float32(mask)

def generate_consistency_mask_2(mono2stereo_disp, stereo_disp, t1=6.0, t2=2.0):
    """Generate a mask of where the stereo disp is wrong, based on the difference between the stereo network and the teacher, monocular network"""

    mono2stereo_disp = np.abs(mono2stereo_disp)
    stereo_disp = np.abs(stereo_disp)
        
    # mask where they differ by a large amount
    mask = (stereo_disp - mono2stereo_disp) < t1
    mask *= (mono2stereo_disp - stereo_disp) < t2

    return np.float32(mask), 1 - np.float32(mask)

def find_scale_offset(disp_mono, disp_stereo):
    disp_pred = disp_mono.flatten()
    disp_gt = disp_stereo.flatten()
    X = np.vstack([disp_pred, np.ones(len(disp_pred))]).T
    Y = disp_gt.reshape(-1, 1)
    try:
        P = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        scale = P[0][0]
        offset = P[1][0]
        mae = np.abs(scale * disp_pred + offset - disp_gt).mean()
    except np.linalg.LinAlgError:
        scale = 1
        offset = 0
        mae = 0

    return scale, offset, mae


def load_disp(args):
    data_path = args.data_directory
    dirs = args.data_dirs
    # '20170221_1357', '20170222_0715', '20170222_1207', '20170222_1638', '20170223_0920', '20170223_1217', '20170223_1445', '20170224_1022'
    # dirs = ['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742']
    disp_list = {}
    for dir in dirs:
        pfm_path = data_path + r"/" + dir + r"/" + "MonoDisp"
        pfm_file = sorted(glob.glob(pfm_path + r"/*.pfm"))
        npy_path = data_path + r"/" + dir + r"/" + "ProxyCRE" + r"/npy"
        npy_file = sorted(glob.glob(npy_path + r"/*.npy"))
        # print(len(npy_file))
        disp_dir = {}
        for pfm, npy in zip(pfm_file, npy_file):
            # print(pfm, npy)
            key = npy.split('/')[-1].replace("_cre.npy", "")
            midas_disp, _ = read_pfm(pfm)
            midas_disp = midas_disp * args.disp_scale + args.disp_offset
            cre_disp = np.float32(np.load(npy))
            disp_dir[key] = (midas_disp, cre_disp)
        disp_list[data_path + r"/" + dir] = disp_dir
    print(len(disp_list[data_path + r"/" + dir]))

    return disp_list

def mean_scale_offset(scales, offsets):
    """计算有效伪标签的scale和offset均值"""
    scales = np.array(scales)
    offsets = np.array(offsets)

    low, high = np.percentile(scales, [10, 90])
    mask = (scales > low) & (scales < high)
    filtered_scales = scales[mask]
    filtered_offsets = offsets[mask]
    mean_scale = np.mean(filtered_scales)
    mean_offset = np.mean(filtered_offsets)

    return mean_scale, mean_offset

def gen_semi_proxy_label(args, disp_list):
    """Generate semi dense proxy labels, by filtering out large gradient \ small or large distance \ consistency with mono disp

    Args:
        args (argparse): options to contral
        disp_list (dict): mono/stereo disp by pretrained model
    """
    for dir, disp_dir in disp_list.items():
        assert args.output_directory == "SemiProxy", "make sure output dir is 'SemiProxy'"
        output_path = dir + r"/" + args.output_directory
        os.makedirs(output_path + r"/npy", exist_ok=True)
        if args.save_png:
            os.makedirs(output_path + r"/png", exist_ok=True)
            os.makedirs(output_path + r"/confidence", exist_ok=True)
            os.makedirs(output_path + r"/consistency", exist_ok=True)

        for file, disps in disp_dir.items():
            midas_disp, cre_disp = disps
            if args.consistency_mask:
                consistency_mask, reverse_consistency_mask = generate_consistency_mask(midas_disp, cre_disp, 0.5, 0.5)
            else:
                consistency_mask = np.ones_like(cre_disp, np.float32)
                reverse_consistency_mask = np.ones_like(cre_disp, np.float32)
                
            _, mask_d_mono, mask_g_mono = generate_confidence_map(midas_disp, 0.5)
            _, mask_d, mask_g = generate_confidence_map(cre_disp)
            if not args.distance_mask:
                mask_d = np.ones_like(cre_disp, np.float32)
            if not args.gradient_mask:
                mask_g = np.ones_like(cre_disp, np.float32)
            
            reliable_mask = consistency_mask * mask_g_mono * mask_d_mono * mask_g * mask_d
            # 当stereo完全失效时,使用midas直接作为proxy_label
            if np.count_nonzero(reliable_mask) < 0.25 * reliable_mask.size:
                print("scale: ", file)
                semi_proxy = np.zeros_like(cre_disp)
            else:
                reliable_cre_disp = cre_disp * reliable_mask
                reliable_midas_disp = midas_disp * reliable_mask
                scale, offset, mae = find_scale_offset(reliable_midas_disp, reliable_cre_disp)
                absoluate_midas_disp = midas_disp * scale + offset
                absoluate_consistency_mask, reverse_absoluate_consistency_mask = generate_consistency_mask_2(absoluate_midas_disp, cre_disp, 8.0, 1.0)

                semi_proxy = cre_disp * mask_d * mask_g * absoluate_consistency_mask 

            np.save(output_path + r"/npy/" + file + "_semi_proxy.npy", semi_proxy)
            if args.save_png:
                proxy_im = disp2im(semi_proxy)
                proxy_im_color = cv2.applyColorMap(np.uint8(proxy_im * 255.0), cv2.COLORMAP_INFERNO)
                png_path = output_path + r"/png/" + file + "_semi_proxy.png"
                cv2.imwrite(png_path, proxy_im_color)

                if args.distance_mask or args.gradient_mask:
                    confidence_path = output_path + r"/confidence/" + file + "_semi_proxy.png"
                    cv2.imwrite(confidence_path, mask_d * mask_g * 255.0)

                if args.consistency_mask:
                    consistency_path = output_path + r"/consistency/" + file + "_semi_proxy.png"
                    cv2.imwrite(consistency_path, consistency_mask * 255.0)
                    
def gen_dense_proxy_label(args, disp_list):
    for dir, disp_dir in disp_list.items():
        assert args.output_directory == "DenseProxy", "make sure output dir is 'DenseProxy'"
        output_path = dir + r"/" + args.output_directory
        os.makedirs(output_path + r"/npy", exist_ok=True)
        if args.save_png:
            os.makedirs(output_path + r"/png", exist_ok=True)
            os.makedirs(output_path + r"/confidence", exist_ok=True)
            os.makedirs(output_path + r"/consistency", exist_ok=True)
        
        fail_labels = []
        mono_masks = []
        scales = []
        offsets = []

        for file, disps in disp_dir.items():
            midas_disp, cre_disp = disps
            if args.consistency_mask:
                consistency_mask, reverse_consistency_mask = generate_consistency_mask(midas_disp, cre_disp, 0.5, 0.5)
            else:
                consistency_mask = np.ones_like(cre_disp, np.float32)
                reverse_consistency_mask = np.ones_like(cre_disp, np.float32)
                
            _, mask_d_mono, mask_g_mono = generate_confidence_map(midas_disp, 0.5)
            _, mask_d, mask_g = generate_confidence_map(cre_disp)
            if not args.distance_mask:
                mask_d = np.ones_like(cre_disp, np.float32)
            if not args.gradient_mask:
                mask_g = np.ones_like(cre_disp, np.float32)
            
            reliable_mask = consistency_mask * mask_g_mono * mask_d_mono * mask_g * mask_d
            if np.count_nonzero(reliable_mask) < 0.25 * reliable_mask.size:
                fail_labels.append(file)
                mono_masks.append(mask_g_mono * mask_d_mono)

            else:
                reliable_cre_disp = cre_disp * reliable_mask
                reliable_midas_disp = midas_disp * reliable_mask
                scale, offset, mae = find_scale_offset(reliable_midas_disp, reliable_cre_disp)
                absoluate_midas_disp = midas_disp * scale + offset
                # print(scale, offset, mae)
                absoluate_consistency_mask, reverse_absoluate_consistency_mask = generate_consistency_mask_2(absoluate_midas_disp, cre_disp, 8.0, 1.0)

                dense_proxy = cre_disp * mask_d * mask_g * absoluate_consistency_mask + absoluate_midas_disp * mask_d_mono * mask_g_mono * reverse_absoluate_consistency_mask

                scales.append(scale)
                offsets.append(offset)

                np.save(output_path + r"/npy/" + file + "_dense_proxy.npy", dense_proxy)
                if args.save_png:
                    proxy_im = disp2im(dense_proxy)
                    proxy_im_color = cv2.applyColorMap(np.uint8(proxy_im * 255.0), cv2.COLORMAP_INFERNO)
                    png_path = output_path + r"/png/" + file + "_dense_proxy.png"
                    cv2.imwrite(png_path, proxy_im_color)

                    if args.distance_mask or args.gradient_mask:
                        confidence_path = output_path + r"/confidence/" + file + "_dense_proxy.png"
                        cv2.imwrite(confidence_path, mask_d * mask_g * 255.0)

                    if args.consistency_mask:
                        consistency_path = output_path + r"/consistency/" + file + "_dense_proxy.png"
                        cv2.imwrite(consistency_path, consistency_mask * 255.0)
                    
                    # absoluate_midas_disp_im = disp2im(absoluate_midas_disp)
                    # absoluate_midas_disp_im_color = cv2.applyColorMap(np.uint8(absoluate_midas_disp_im * 255.0), cv2.COLORMAP_INFERNO)
                    # cv2.imwrite(dir + r"/MonoDisp" + r"/png/" + file + "-dpt_beit_large_512.png", absoluate_midas_disp_im_color)

        mean_scale, mean_offset = mean_scale_offset(scales, offsets)
        print(mean_scale, mean_offset)
        
        # 当stereo完全失效时,使用midas直接作为proxy_label
        for file, mono_mask in zip(fail_labels, mono_masks):
            midas_disp = disp_dir[file][0]
            absoluate_midas_disp = midas_disp * mean_scale + mean_offset
            print("scale:", file)
            dense_proxy = absoluate_midas_disp * mono_mask

            np.save(output_path + r"/npy/" + file + "_dense_proxy.npy", dense_proxy)
            if args.save_png:
                proxy_im = disp2im(dense_proxy)
                proxy_im_color = cv2.applyColorMap(np.uint8(proxy_im * 255.0), cv2.COLORMAP_INFERNO)
                png_path = output_path + r"/png/" + file + "_dense_proxy.png"
                cv2.imwrite(png_path, proxy_im_color)
                    
                # absoluate_midas_disp_im = disp2im(absoluate_midas_disp)
                # absoluate_midas_disp_im_color = cv2.applyColorMap(np.uint8(absoluate_midas_disp_im * 255.0), cv2.COLORMAP_INFERNO)
                # cv2.imwrite(dir + r"/MonoDisp" + r"/png/" + file + "-dpt_beit_large_512.png", absoluate_midas_disp_im_color)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', help="directory to load test datasets", default="./datasets/test")
    parser.add_argument('--output_directory', help="directory to save proxy label files", default="SemiProxy", choices=["SemiProxy", "DenseProxy"])
    parser.add_argument('--disp_scale', type = float, help="disp scale for midas", default=0.0006504932339649878)
    parser.add_argument('--disp_offset', type = float, help="disp offset for midas", default=0.7388445603651852)
    parser.add_argument('--data_dirs', nargs='+', type=str, help="choose dir to process", default=['20170221_1357', '20170222_0715', '20170222_1207', '20170222_1638', '20170223_0920', '20170223_1217', '20170223_1445', '20170224_1022'])
    parser.add_argument('--save_png', action="store_true", help="save image of proxy label")
    # ablation
    parser.add_argument('--distance_mask', action="store_true", help="use distance mask")
    parser.add_argument('--gradient_mask', action="store_true", help="use gradient mask")
    parser.add_argument('--consistency_mask', action="store_true", help="use consistency mask")

    args = parser.parse_args()

    disp_list = load_disp(args)

    if args.output_directory == "SemiProxy":
        gen_semi_proxy_label(args, disp_list)
    elif args.output_directory == "DenseProxy":
        gen_dense_proxy_label(args, disp_list)

