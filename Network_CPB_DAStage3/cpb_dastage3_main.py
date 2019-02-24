import sys

sys.path.append('../Data_Initialization/')
import os
from cpb_dastage3_model import cpb_dastage3_model
import Initialization as init
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', required=True, help='[the name of the model]')
parser.add_argument('-gpu', required=True, help='[set particular gpu for calculation]')
parser.add_argument('-data_domain', required=True, help='[choose the data domain between source and target]')

parser.add_argument('-epoch', default=200, type=int)
parser.add_argument('-num_class', default=6, type=int)
parser.add_argument('-learning_rate', default=2e-4, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-img_height', default=32, type=int)
parser.add_argument('-img_width', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.data_domain == 'Source':
    reload_path = '../checkpoint/baseline_s2t/baseline_s2t-???'
elif args.data_domain == 'Target':
    reload_path = '../checkpoint/baseline_t2s/baseline_t2s-???'
else:
    reload_path = ''

src_data, tar_data = init.loadData(data_domain=args.data_domain)
src_training, src_validation, src_test = src_data
tar_training, tar_test = tar_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    res_model = cpb_dastage3_model(model_name=args.model_name,
                                   sess=sess,
                                   train_data=[src_training, tar_training],
                                   val_data=src_validation,
                                   tst_data=[src_test, tar_test],
                                   reload_path=reload_path,
                                   epoch=args.epoch,
                                   num_class=args.num_class,
                                   learning_rate=args.learning_rate,
                                   batch_size=args.batch_size,
                                   img_height=args.img_height,
                                   img_width=args.img_width)

    res_model.train()
