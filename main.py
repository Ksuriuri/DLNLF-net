import os
from DLNLF.model import *
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='resnet')

# init parameters
parser.add_argument('--save_dir', type=str, default=r'/home/public/Documents/hhy/medical_physis/63DMQ/temp.npz')
parser.add_argument('--num_res_blocks', type=int, default=4, help='number of residual blocks')

# train parameters
parser.add_argument('--input_dir', type=str, default=r'/home/public/Documents/hhy/data/63DMQ/axial2_27', help='dir of input images')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=160)
parser.add_argument('--one_sample_num', type=int, default=27)
parser.add_argument('--num_train', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=7e-6)  # b_cnn_cnn b_nl_cnn: 7e-6; b_dn_cnn: 6e-6
parser.add_argument('--beta1', type=float, default=0.9)

args = parser.parse_args()

dlnlf = DLNLF(
    save_dir=args.save_dir,
    num_res_blocks=args.num_res_blocks,
    one_sample_num=args.one_sample_num
)
dlnlf.train(
    input_dir=args.input_dir,
    batch_size=args.batch_size,
    num_epochs=args.num_epochs,
    learning_rate=args.learning_rate,
    beta1=args.beta1,
    num_train=args.num_train
)
