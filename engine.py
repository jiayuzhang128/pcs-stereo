from __future__ import absolute_import, division, print_function

import time
import glob
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from collections import OrderedDict
from PIL import ImageOps
import numpy as np

import json

from utils.utils import *
from losses.normalized_gradient_loss import GradientLoss

from datasets.pittsburgh_stereo_dateste import PittburghStereoDataset
from networks.igev.core.igev_stereo import IGEVStereo
from networks.psmnet.models import stackhourglass
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

torch.cuda.manual_seed(666)
torch.manual_seed(666)
np.random.seed(666)

try:
    from torch.cuda.amp import GradScaler
except:
    # 导入失败则创建一个GradScaler类的占为符版本，其中都是空方法
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# torch.backends.cudnn.benchmark = True

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class TrainEngine():
    
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.models = {}
        self.parameters_to_train = []

        self.device = 'cuda'
        torch.cuda.set_device('cuda:{}'.format(self.opt.gpus[0]))
        self.profile = self.opt.profile
        self.unfreeze_feat = False # flag to contral unfreeze feature net
        
        # GradScaler用于在训练过程中缩放模型的梯度，防止出现数值问题
        self.scaler = GradScaler(enabled=self.opt.mixed_precision)

        # ==================构建模型=======================

        if self.opt.train_stereo:
            if self.opt.stereo_network == 'IGEVStereo':
                self.models['stereo_network'] = torch.nn.DataParallel(IGEVStereo(self.opt).cuda(), device_ids=self.opt.gpus, output_device=self.opt.gpus[0])

            elif self.opt.stereo_network == 'PSMNet':
                self.models['stereo_network'] = torch.nn.DataParallel(stackhourglass(self.opt.max_disp).cuda(), device_ids=self.opt.gpus, output_device=self.opt.gpus[0])

            self.parameters_to_train += list(self.models['stereo_network'].parameters())


        # ==================设置训练参数=======================
        if self.opt.train_stereo:
            self.model_optimizer = optim.AdamW(
                self.parameters_to_train,
                self.opt.lr[0],
                weight_decay=self.opt.weight_decay)

            self.model_lr_scheduler = ChainedScheduler(
                self.model_optimizer,
                T_0=int(self.opt.lr[2]),
                T_mul=1,
                eta_min=self.opt.lr[1],
                last_epoch=-1,
                max_lr=self.opt.lr[0],
                warmup_steps=0,
                gamma=0.9)


        # ==================加载模型权重=======================
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print('Training model named:\n  ', self.opt.model_name)
        print('Models and tensorboard events files are saved to:\n  ', self.opt.log_dir)
        print('Training is using:\n  ', self.device)

        # ==================加载数据集=======================
        datasets_dict = {'pittsburgh': PittburghStereoDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        
        self.train_dataset = self.dataset(
            self.opt.data_path,
            self.opt.list_path,
            self.opt.train_splits,
            self.opt.height,
            self.opt.width,
            is_train=True,
            ftype=self.opt.ftype,
            fill=self.opt.fill,
            pad_mode=self.opt.pad_mode,
            no_norm=True,
            is_flip=self.opt.is_flip,
            flip_direction=self.opt.flip_direction)
        
        self.num_total_steps = self.train_dataset.__len__() // self.opt.batch_size * self.opt.num_epochs

        self.padded_width = self.train_dataset.padded_width
        self.padded_height = self.train_dataset.padded_height

        assert self.padded_height % 32 == 0, 'Input height must divided by 32'
        assert self.padded_width % 32 == 0, 'Input width must divided by 32'

        # 后续unpad数据，计算loss
        self.padder = self.train_dataset.padder

        self.train_loader = DataLoader(
            self.train_dataset,
            self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        val_dataset = self.dataset(
            self.opt.data_path,
            self.opt.list_path,
            self.opt.test_splits,
            self.opt.height,
            self.opt.width,
            is_train=False,
            ftype=self.opt.ftype,
            fill=self.opt.fill,
            pad_mode=self.opt.pad_mode,
            no_norm=True,
            flip_direction=self.opt.flip_direction)

        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        self.val_iter = iter(self.val_loader)

        print('There are {:d} training items and {:d} validation items\n'.format(
            len(self.train_dataset), len(val_dataset)))

        # ==================设置tensorboard=======================
        self.writers = {}
        for mode in ['train', 'val']:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.save_opts()

    def load_model(self):
        """Load model(s) from disk
        """

        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            'Cannot find folder {}'.format(self.opt.load_weights_folder)
        print('loading model from folder {}'.format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print('Loading {} weights...'.format(n))
            path = os.path.join(self.opt.load_weights_folder, '{}'.format(n))


            if self.opt.train_stereo:
                self.models['stereo_network'].load_state_dict(torch.load(path), strict=False)

        # loading adam state
        if self.opt.train_stereo:
            optimizer_load_path = os.path.join(self.opt.load_weights_folder, 'adam.pth')
            if os.path.isfile(optimizer_load_path):
                print('Loading model Adam weights')
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            else:
                print('Cannot find model Adam weights so Adam is randomly initialized')

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def set_train(self):
        """Convert all models to training mode
        """

        if self.opt.train_stereo:
            self.models['stereo_network'].train()
            if self.opt.stereo_network == 'IGEVStereo':
                self.models['stereo_network'].module.freeze_bn()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        if self.opt.train_stereo:
            for n in self.models.values():
                n.eval()

    def train(self):
        """Run the entire training pipeline
        """

        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print('Training')
        self.set_train()

        if self.opt.train_stereo:
            self.model_lr_scheduler.step()

        for batch_idx, inputs in enumerate(self.train_loader):
            if batch_idx >5:
              break

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)


            if self.opt.train_stereo:
                self.model_optimizer.zero_grad()
                
            self.scaler.scale(losses['loss']).backward()

            if self.opt.train_stereo:
                self.scaler.unscale_(self.model_optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, 1.0)
                self.scaler.step(self.model_optimizer)
                self.scaler.update()

            duration = time.time() - before_op_time
            
            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses['loss'].cpu().data)

                self.log('train', inputs, outputs, losses)

            self.step += 1

        if not self.opt.no_eval:
            self.val()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, value in inputs.items():
            inputs[key] = value.cuda()

        outputs = {}
        # stereo disp
        if self.opt.train_stereo:
            # 创建文件夹存放训练期间视差文件
            output_directory = self.opt.tmp_img_path
            if os.path.exists(output_directory):
                pass
            else:
                os.makedirs(output_directory)

            if self.opt.stereo_network == 'IGEVStereo':
                self.igev_padder = IGEVInputPadder(self.padder.unpad(inputs['rgb']).shape, divis_by=32)
        
                stereo_rgb, stereo_nir= self.igev_padder.pad(self.padder.unpad(inputs['rgb']), self.padder.unpad(inputs['nir']))
                outputs['stereo_disp_init'], outputs['stereo_disps'] = self.models['stereo_network'](stereo_rgb * 255.0, stereo_nir * 255.0, iters=self.opt.train_iters)
        
                disp = self.igev_padder.unpad(outputs['stereo_disps'][-1]).detach().cpu().numpy()[0][0]

            elif self.opt.stereo_network == 'PSMNet':
                outputs['stereo_disps'] = self.models['stereo_network'](inputs['rgb'], inputs['nir'])
                disp = self.padder.unpad(outputs['stereo_disps'][-1]).detach().cpu().numpy()[0][0]

            img_rgb = self.padder.unpad(inputs['rgb']).permute(0,2,3,1).detach().cpu().numpy()[0]
            img_rgb = cv2.cvtColor(np.uint8(img_rgb*255), cv2.COLOR_RGB2BGR)
            vmax = np.percentile(disp, 99)
            disp_color = cv2.applyColorMap(np.uint8(np.clip(disp/vmax, 0,1)*255), cv2.COLORMAP_JET)
            disp_side = np.concatenate((img_rgb, disp_color), axis=1)
            file_stem = str(self.step)
            if self.step % 250 == 0:
                cv2.imwrite(output_directory + '/' + file_stem + '.png', disp_side)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def compute_gradient_loss(self, disp1, disp2):
        """计算视差平滑损失"""
        self.gradient_loss = GradientLoss()
        disp1 = torch.clamp(disp1, 0.01, 192)
        disp2 = torch.clamp(disp2, 0.01, 192)
        log_disp_diff = torch.log(disp1) - torch.log(disp2)
#         log_disp_diff = disp1 - disp2
        return self.gradient_loss(log_disp_diff)

    def compute_consistency_loss(self, proxy_label, stereo_disp):
        """计算代理监督损失"""
        proxy_mask = (proxy_label == 0)
        abs_diff = torch.abs(proxy_label - stereo_disp) * (1 - proxy_mask.to(torch.float32))

        return abs_diff.mean(1, True)

    def compute_losses(self, inputs, outputs):
        """
        Compute the reprojection and smoothness losses for a minibatch
        inputs need unpad
        outputs need unpad only for [disp predictive_mask]
        """

        losses = {}
        loss = 0

        left_image = self.padder.unpad(inputs['rgb'])

        # disp
        if self.opt.train_stereo:

            if self.opt.stereo_network == 'IGEVStereo':
                stereo_disp_init = self.igev_padder.unpad(outputs['stereo_disp_init'])
                stereo_disps = [self.igev_padder.unpad(disp) for disp in outputs['stereo_disps']]


                # igev loss weight gamma = 0.9
                igev_loss_gamma = self.opt.igev_loss_gamma
                # num of igev iters
                n_iters = len(stereo_disps)

            elif self.opt.stereo_network in ['PSMNet']:
                stereo_disps = [self.padder.unpad(disp) for disp in outputs['stereo_disps']]

                # psmnet loss weight gamma = [0.5, 0.7, 0.9]
                psmnet_loss_gamma = self.opt.psmnet_loss_gamma
                # num of psmnet iters
                n_iters = len(stereo_disps)

        # ========consistency loss=====
        consistency_loss = 0
        if self.opt.train_stereo & self.opt.use_consistency_loss:
            
            if self.opt.use_dense_label:
                if self.opt.stereo_network == 'IGEVStereo':
                # igev loss L_init + \sum^N_{i=1}\gamma^{N-i}L_i gamma=0.9
                    consistency_loss += self.compute_consistency_loss(inputs['dense_proxy_label'], stereo_disp_init)
                    for idx, stereo_disp in enumerate(stereo_disps):
                        consistency_loss += igev_loss_gamma ** (n_iters-idx-1) * self.compute_consistency_loss(inputs['dense_proxy_label'], stereo_disp) 
                    consistency_loss = mean_nonzero(consistency_loss)

                elif self.opt.stereo_network in ['PSMNet']:
                    for idx, stereo_disp in enumerate(stereo_disps):
                        consistency_loss += psmnet_loss_gamma[idx] * self.compute_consistency_loss(inputs['dense_proxy_label'], stereo_disp)
                    consistency_loss = mean_nonzero(consistency_loss)
            else:
                if self.opt.stereo_network == 'IGEVStereo':
                    tmp_loss = self.compute_consistency_loss(inputs['semi_proxy_label'], stereo_disp_init)
                    zero_count = 0
                    nonzero_loss = []
                    for i in range(tmp_loss.shape[0]):
                        if torch.all(tmp_loss[i]==0):
                            zero_count += 1
                        else:
                            nonzero_loss.append(tmp_loss[i].unsqueeze(0))
                    if zero_count == tmp_loss.shape[0]:
                        consistency_loss = 0
                    else:
                        tmp_consistency_loss = mean_nonzero(torch.cat(nonzero_loss, dim=0))
                    consistency_loss += tmp_consistency_loss

                    if not (zero_count == tmp_loss.shape[0]):
                        for idx, stereo_disp in enumerate(stereo_disps):
                            nonzero_loss = []
                            for i in range(tmp_loss.shape[0]):
                                if torch.all(tmp_loss[i]==0):
                                    zero_count += 1
                                else:
                                    nonzero_loss.append(tmp_loss[i].unsqueeze(0))
                            temp_consistency_loss = mean_nonzero(torch.cat(nonzero_loss, dim=0))
                            consistency_loss += igev_loss_gamma ** (n_iters-idx-1) * temp_consistency_loss

                elif self.opt.stereo_network in ['PSMNet']:
                    for idx, stereo_disp in enumerate(stereo_disps):
                        # 忽略无smei_proxy_label项的一致性损失
                        tmp_loss = self.compute_consistency_loss(inputs['semi_proxy_label'], stereo_disp)
                        zero_count = 0
                        nonzero_loss = []
                        for i in range(tmp_loss.shape[0]):
                            if torch.all(tmp_loss[i]==0):
                                zero_count += 1
                            else:
                                nonzero_loss.append(tmp_loss[i].unsqueeze(0))
                        # print(zero_count)
                        if zero_count == tmp_loss.shape[0]:
                            consistency_loss = 0
                            break
                        else:
                            tmp_consistency_loss = mean_nonzero(torch.cat(nonzero_loss, dim=0))
                        consistency_loss += psmnet_loss_gamma[idx] * tmp_consistency_loss

                        

        # =========gradient loss========
        gradient_loss = 0
        if self.opt.train_stereo & self.opt.use_gradient_loss:

            if self.opt.stereo_network == 'IGEVStereo':
                stereo_disp = stereo_disps[-1]
                gradient_loss = self.compute_gradient_loss(stereo_disp, inputs['mono_proxy_label'])
                gradient_loss += self.compute_gradient_loss(stereo_disp_init, inputs['mono_proxy_label'])

            elif self.opt.stereo_network in ['MobileStereoNet3D', 'MobileStereoNet3D']:
                stereo_disp = stereo_disps[-1]
                gradient_loss = self.compute_gradient_loss(stereo_disp, inputs['mono_proxy_label'])

            elif self.opt.stereo_network in ['PSMNet']:
                stereo_disp = stereo_disps[-1]
                gradient_loss = self.compute_gradient_loss(stereo_disp, inputs['mono_proxy_label'])

        if self.opt.train_stereo:
            loss +=  self.opt.gradient_weight * gradient_loss + self.opt.consistency_weight * consistency_loss

        losses['loss'] = loss
        return losses

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, 'models', 'weights_{}'.format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if self.opt.train_stereo:
            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, '{}.pth'.format(model_name))
                to_save = model.state_dict()
                torch.save(to_save, save_path)
                
            save_path = os.path.join(save_folder, '{}.pth'.format('adam'))
            torch.save(self.model_optimizer.state_dict(), save_path)

    def get_eavl_disp(self, args, model):
        data_path = args.data_path
        output_directory = self.log_path + r'/' + args.eval_out_dir

        disp_list = OrderedDict()
        test_folder = [data_path + r'/' + split for split in args.test_splits.split(',')]
        for i, file in enumerate(test_folder):
            disp_dir = OrderedDict()
            left_imgs = file + '/RGBResize/*.png'
            right_imgs = file + '/NIRResize/*.png'
            # print(file)

            with torch.no_grad():
                left_images = sorted(glob.glob(left_imgs, recursive=True))
                right_images = sorted(glob.glob(right_imgs, recursive=True))
                
                if args.save_eval_results:
                    print(f'Found {len(left_images)} images. Saving files to {output_directory}/')

                for (rgb_file, nir_file) in list(zip(left_images, right_images)):
                    if self.opt.stereo_network == 'IGEVStereo':
                        # print(rgb_file)
                        rgb = load_image(rgb_file)
                        nir = load_image(nir_file)
                        rgb, nir = self.igev_padder.pad(rgb, nir)
                        disp = model(rgb, nir, iters=args.valid_iters, test_mode=True)
                        disp = self.igev_padder.unpad(disp)
                        # print(disp[0][0].shape)

                    elif self.opt.stereo_network in ['PSMNet']: 
                        rgb = self.train_dataset.transform(self.train_dataset.loader(rgb_file).convert('RGB')).unsqueeze(0)
                        nir = self.train_dataset.transform(ImageOps.colorize(self.train_dataset.loader(nir_file).convert('L'), black='black', white='white')).unsqueeze(0)
                        rgb = self.padder.pad(rgb)
                        nir = self.padder.pad(nir)
                        disp = model(rgb, nir)
                        disp = self.padder.unpad(disp)

                    disp = nn.functional.interpolate(disp,(429,582))[0][0].cpu().numpy()
                    file_stem = rgb_file.split('/')[-1].split('.')[-2]
                    key = rgb_file.split('/')[-1].replace('RGBResize.png','Keypoint.txt')
                    disp_dir[key] = disp

                    if args.save_eval_results:
                        npy_dir = output_directory + '/' + file.split('/')[-1] + r'/npy'
                        png_dir = output_directory + '/' + file.split('/')[-1] + r'/png'
                        os.makedirs(png_dir, exist_ok=True)
                        os.makedirs(npy_dir, exist_ok=True)

                        np.save(npy_dir +f'/{file_stem}.npy', disp)
                        disp = np.round(disp * 256).astype(np.uint16)
                        plt.imsave(png_dir + f'/{file_stem}.png', disp, cmap='jet')

                disp_list[file] = disp_dir
        return disp_list

    def evaluate_stereo(self, args, disp_list):
        """evaluate stereo network while training

        Args:
            disp_list (list): predicted disparity value
        """

        ans = [[],[],[],[],[],[],[],[]]
        with torch.no_grad():
            for file, disp in disp_list.items():
                for key, value in disp.items():

                    disp_value = nn.functional.interpolate(torch.from_numpy(value).unsqueeze(0).unsqueeze(0),(429,582),mode='bilinear')[0][0].cpu().numpy()
                    # print(key)

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
                        ans[c].append((p-d)*(p-d))

            rmse = []
            for c in range(8):
                rmse.append(pow(sum(ans[c]) / (len(ans[c])+1e-3), 0.5))
            print('Common    Light     Glass     Glossy  Vegetation   Skin    Clothing    Bag       Mean')
            print(round(rmse[0], 4), '  ', round(rmse[1], 4), '  ', round(rmse[2], 4), '  ', round(rmse[3], 4), '  ', round(rmse[4], 4), '  ', round(rmse[5], 4), '  ', round(rmse[6], 4), '  ', round(rmse[7], 4), '  ', round(sum(rmse) / 8.0, 4))
            print()

    def val(self):
        """Validate the model on a single minibatch
        """
        if self.opt.stereo_network == 'IGEVStereo':
            self.igev_padder = IGEVInputPadder((429,582), divis_by=32)
        self.set_eval()

        if self.opt.train_stereo:
            eval_disp_list = self.get_eavl_disp(self.opt, self.models['stereo_network'])
            self.evaluate_stereo(self.opt, eval_disp_list)

        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        
        if self.opt.train_stereo:
            print_string = 'epoch {:>3} |stereo_lr {:.9f} | batch {:>6} | examples/s: {:5.1f}' + \
                ' | loss: {:.5f} | time elapsed: {} | time left: {}'
            print(print_string.format(
                self.epoch, 
                self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                batch_idx, samples_per_sec, loss,
                sec_to_hm_str(time_sofar), 
                sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar('{}'.format(l), v, self.step)

        for j in range(min(1, self.opt.batch_size)):  
            # write a maxmimum of one images
            writer.add_image(
                'rgb_{}'.format(j),
                self.padder.unpad(inputs['rgb'])[j].data, self.step)

            writer.add_image(
                'nir_{}'.format(j),
                self.padder.unpad(inputs['nir'])[j].data, self.step)

            writer.add_image(
                'dense_proxy_color_{}'.format(j),
                colorize_image(inputs[('dense_proxy_label')][j]), self.step)

            # writer.add_image(
            #     'semi_proxy_{}'.format(j),
            #     colorize_image(inputs[('semi_proxy_label')][j]), self.step)

            if self.opt.train_stereo:
                if self.opt.stereo_network == 'IGEVStereo':
                    writer.add_image(
                        'stereo_disp_color_{}'.format(j),
                        colorize_image(self.igev_padder.unpad(outputs['stereo_disps'][-1])[j]), self.step)
                    writer.add_image(
                        'stereo_disp_{}'.format(j),
                        normalize_image(self.igev_padder.unpad(outputs['stereo_disps'][-1])[j]), self.step)

                if self.opt.stereo_network == 'PSMNet':
                    writer.add_image(
                        'stereo_disp_color_{}'.format(j),
                        colorize_image(self.padder.unpad(outputs['stereo_disps'][-1])[j]), self.step)




                