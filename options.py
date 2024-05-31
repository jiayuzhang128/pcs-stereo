from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class FMCSOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Lite-Mono options")

        # PATHS
        self.parser.add_argument("--data_path",
                                type=str,
                                help="path to the training data",
                                required=True)

        self.parser.add_argument("--list_path",
                                type=str,
                                default="lists")

        self.parser.add_argument("--train_splits", 
                                type=str,
                                default="20170221_1357,20170222_0715,20170222_1207,20170222_1638,20170223_0920,20170223_1217,20170223_1445,20170224_1022")


        self.parser.add_argument("--log_dir",
                                type=str,
                                help="log directory",
                                default="./tmp")

        # DATASET options
        self.parser.add_argument("--is_flip",
                                type=bool,
                                help="filp data or not",
                                default=False)

        self.parser.add_argument("--fill",
                                type=int,
                                help="constant value to pad image",
                                default=0)

        self.parser.add_argument("--pad_mode",
                                type=str,
                                help="pad mod for loading images",
                                choices=["symmetric", "replicate", "edge", "constant"],
                                default="constant")
        
        self.parser.add_argument("--dataset",
                                type=str,
                                help="dataset to train on",
                                default="pittsburgh",
                                choices=["pittsburgb", "rgb_ms"])
        
        self.parser.add_argument("--ftype",
                                type=str,
                                help="if set, trains from raw png files (instead of jpgs)",
                                default="png",
                                choices=["png","jpg","jpeg"])
        
        self.parser.add_argument("--height",
                                type=int,
                                help="input image height",
                                default=429)
        
        self.parser.add_argument("--width",
                                type=int,
                                help="input image width",
                                default=582)

        self.parser.add_argument("--no_norm",
                                help="input image width",
                                action="store_false")

        self.parser.add_argument("--flip_direction",
                                help="direction of flipped image",
                                default="h",
                                choices=["h", "w"])
        
        # TRAINING options
        self.parser.add_argument("--model_name",
                                type=str,
                                help="the name of the folder to save the model in",
                                default="pittsburgh")
        
        self.parser.add_argument("--train_stereo",
                                help="if set, start train the final stereo network",
                                action="store_true")

        self.parser.add_argument("--train_feat",
                                help="if set, start train the feature network",
                                action="store_true")

        self.parser.add_argument("--epoch_to_unfreeze_feat",
                                type=int,
                                help="epoch to unfreeze feature network",
                                default=15)

        self.parser.add_argument("--model",
                                type=str,
                                help="which model to load",
                                choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"],
                                default="lite-mono")
        
        self.parser.add_argument("--weight_decay",
                                type=float,
                                help="weight decay in AdamW",
                                default=1e-2)
        
        self.parser.add_argument("--drop_path",
                                type=float,
                                help="drop path rate",
                                default=0.2)
        
        self.parser.add_argument("--num_layers",
                                type=int,
                                help="number of resnet layers",
                                default=18,
                                choices=[18, 34, 50, 101, 152])
        
        self.parser.add_argument("--disparity_smoothness",
                                type=float,
                                help="disparity smoothness weight",
                                default=1e-3)
        
        self.parser.add_argument("--min_depth",
                                type=float,
                                help="minimum depth",
                                default=0.1)
        
        self.parser.add_argument("--max_depth",
                                type=float,
                                help="maximum depth",
                                default=100.0)
        
        self.parser.add_argument("--profile",
                                type=bool,
                                help="profile once at the beginning of the training",
                                default=True)

        self.parser.add_argument("--smoothness_weight",
                                type=float,
                                help="weight of the smoothness loss",
                                default=0.5)

        self.parser.add_argument("--reprojection_weight",
                                type=float,
                                help="weight of the reprojection loss",
                                default=0.5)
        
        self.parser.add_argument("--gradient_weight",
                                type=float,
                                help="weight of the smoothness loss",
                                default=0.5)

        self.parser.add_argument("--consistency_weight",
                                type=float,
                                help="weight of the consistency loss",
                                default=1.0)
        
        self.parser.add_argument("--tmp_img_path",
                                type=str,
                                help="save stereo disp at training for visualization",
                                default="./tmp_disp_img")

        # OPTIMIZATION options
        self.parser.add_argument("--feature_type",
                                type=str,
                                help="way of feat_net extracting feature",
                                default="shared",
                                choices=["shared", "separate"])

        self.parser.add_argument("--batch_size",
                                type=int,
                                help="batch size",
                                default=16)

        self.parser.add_argument("--mutual_batch_size",
                                type=int,
                                help="batch size of mutually optimization",
                                default=1)
        
        self.parser.add_argument("--lr",
                                nargs="+",
                                type=float,
                                help="learning rates of DepthNet and PoseNet. "
                                    "Initial learning rate, "
                                    "minimum learning rate, "
                                    "First cycle step size.",
                                default=[0.0001, 5e-6, 31])
        
        self.parser.add_argument("--num_epochs",
                                type=int,
                                help="number of epochs",
                                default=50)
        
        self.parser.add_argument("--scheduler_step_size",
                                type=int,
                                help="step size of the scheduler",
                                default=15)

        # FeatNet options
        self.parser.add_argument("--n_dims",
                                help="dimensions of output feature",
                                type=int,
                                default=10)

        self.parser.add_argument("--spp_branches",
                                help="branches of spp layer",
                                type=list,
                                default=None)

        self.parser.add_argument("--activation",
                                help="activation of layer",
                                type=str,
                                default="relu")

        self.parser.add_argument("--im_pad",
                                help="pad of input images",
                                type=int,
                                default=None)

        self.parser.add_argument("--norm",
                                help="do L2 normalization or not",
                                action="store_true")

        self.parser.add_argument("--disp_mono2stereo_scale",
                                help="transform mono disp to stereo disp",
                                type=int,
                                default=25.830988)

        # IGEV options
        self.parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
        self.parser.add_argument("--image_size", type=int, nargs="+", default=[320, 736], help="size of the random image crops used during training.")
        # Training parameters
        self.parser.add_argument("--train_iters", type=int, default=6, help="number of updates to the disparity field in each forward pass.")
        self.parser.add_argument("--igev_loss_gamma", type=float, default=0.9, help="loss weight for igev loss.")
        # Validation parameters
        self.parser.add_argument("--valid_iters", type=int, default=32, help="number of flow-field updates during validation forward pass")
        # Architecure choices
        self.parser.add_argument("--corr_implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        self.parser.add_argument("--shared_backbone", action="store_true", help="use a single backbone for the context and feature encoders")
        self.parser.add_argument("--corr_levels", type=int, default=2, help="number of levels in the correlation pyramid")
        self.parser.add_argument("--corr_radius", type=int, default=4, help="width of the correlation pyramid")
        self.parser.add_argument("--n_downsample", type=int, default=2, help="resolution of the disparity field (1/2^K)")
        self.parser.add_argument("--slow_fast_gru", action="store_true", help="iterate the low-res GRUs more frequently")
        self.parser.add_argument("--n_gru_layers", type=int, default=3, help="number of hidden GRU levels")
        self.parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128]*3, help="hidden state and context dimensions")
        self.parser.add_argument("--max_disp", type=int, default=192, help="max disp of geometry encoding volume")
        
        # MobileStereoNet options
        self.parser.add_argument("--msnet_loss_gamma", type=list, default=[0.5, 0.5, 0.7, 1.0], help="loss weight for msnet loss.")

        # PSMNet options
        self.parser.add_argument("--psmnet_loss_gamma", type=list, default=[0.5, 0.7, 1.0], help="loss weight for msnet loss.")
        # ABLATION options

        self.parser.add_argument("--stereo_network",
                                type=str,
                                help="stereo network as backbone",
                                default="IGEVStereo",
                                choices=["IGEVStereo", "MobileStereoNet3D", "MobileStereoNet2D", "PSMNet", "RaftStereo"])

        self.parser.add_argument("--use_feature_metric",
                                help="if set, use feature metric to train stereo",
                                action="store_true")

        self.parser.add_argument("--use_smoothness_loss",
                                help="if set, use smoothness loss for training",
                                action="store_true")

        self.parser.add_argument("--use_gradient_loss",
                                help="if set, use gradient loss for training",
                                action="store_true")\
        
        self.parser.add_argument("--use_consistency_loss",
                                help="if set, use consistency loss for training",
                                action="store_true")
        
        self.parser.add_argument("--use_dense_label",
                                help="if set, use dense proxy label consistency loss for training",
                                action="store_true")
        
        self.parser.add_argument("--use_reprojection_loss",
                                help="if set, use reprojection loss for training",
                                action="store_true")

        self.parser.add_argument("--use_hca",
                                help="if set, use Hierarchical Context Aggregation loss",
                                action="store_true")

        self.parser.add_argument("--use_contrast",
                                help="if set, use contrastive loss",
                                action="store_true")

        self.parser.add_argument("--avg_reprojection",
                                help="if set, use average reprojection loss",
                                action="store_true")
        
        self.parser.add_argument("--disable_automasking",
                                help="if set, does not do auto-masking",
                                action="store_true")
        
        self.parser.add_argument("--predictive_mask",
                                help="if set, use a predictive masking scheme as in Zhou et al",
                                action="store_true")
        
        self.parser.add_argument("--no_ssim",
                                help="if set, disable ssim in the loss",
                                action="store_true")
        
        self.parser.add_argument("--mypretrain",
                                type=str,
                                help="if set, use my pretrained encoder")
        
        self.parser.add_argument("--weights_init",
                                type=str,
                                help="pretrained or scratch",
                                default="pretrained",
                                choices=["pretrained", "scratch"])
        
        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                help="if set disables CUDA",
                                action="store_true")
        
        self.parser.add_argument("--gpus",
                                help="visiable gpus",
                                nargs="+",
                                type=int,
                                default=[0])
        
        self.parser.add_argument("--num_workers",
                                type=int,
                                help="number of dataloader workers",
                                default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                type=str,
                                help="name of model to load")
        
        self.parser.add_argument("--models_to_load",
                                nargs="+",
                                type=str,
                                help="models to load",
                                default=["encoder", "depth", "pose_encoder", "pose"])

        self.parser.add_argument("--log_frequency",
                                type=int,
                                help="number of batches between each tensorboard log",
                                default=100)
        
        self.parser.add_argument("--save_frequency",
                                type=int,
                                help="number of epochs between each save",
                                default=1)

        # EVALUATION options
        self.parser.add_argument("--test_splits", 
                                type=str,
                                default="20170222_0951,20170222_1423,20170223_1639,20170224_0742",
                                help="which split to run eval on")
        
        self.parser.add_argument("--save_eval_results",
                                help="if set saves predicted disparities in png and npy format",
                                action="store_true")
        
        self.parser.add_argument("--no_eval",
                                help="if set disables evaluation",
                                action="store_true")
        
        self.parser.add_argument("--remove_outlaiers",
                                action="store_true")
        
        self.parser.add_argument("--eval_out_dir",
                                type=str,
                                help="if set will output the disparities to this folder",
                                default="eval_disp")
        

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
