import sys

sys.path.append('../Data_Initialization/')
import os
from cpb_dastage2_model import cpb_dastage2_model
import Initialization as init
import argparse
import tensorflow as tf
import cpb_dastage2_utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', required=True, help='[the name of the model]')
parser.add_argument('-gpu', required=True, help='[set particular gpu for calculation]')
parser.add_argument('-data_domain', required=True, help='[choose the data domain between source and target]')

parser.add_argument('-epoch', default=200, type=int)
parser.add_argument('-num_class', default=6, type=int)
parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-img_height', default=32, type=int)
parser.add_argument('-img_width', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.data_domain == 'Source':
    src_name = 'source'
    tar_name = 'target'
    reload_path = '../checkpoint/baseline_s2t/baseline_s2t-300'
elif args.data_domain == 'Target':
    src_name = 'target'
    tar_name = 'source'
    reload_path = '../checkpoint/baseline_t2s/baseline_t2s-14'
else:
    src_name = ''
    tar_name = ''
    reload_path = ''

src_training = init.loadPickle(utils.experimentalPath, src_name + '_training.pkl')
src_validation = init.loadPickle(utils.experimentalPath, src_name + '_validation.pkl')
src_test = init.loadPickle(utils.experimentalPath, src_name + '_test.pkl')

tar_training = init.loadPickle(utils.experimentalPath, tar_name + '_' + src_name + '.pkl')
tar_test = init.loadPickle(utils.experimentalPath, tar_name + '_test.pkl')

src_training = utils.normalizeInput(src_training, mode='Paired')
src_validation = utils.normalizeInput(src_validation, mode='Paired')
src_test = utils.normalizeInput(src_test, mode='Paired')

tar_training = utils.normalizeInput(tar_training, mode='Unpaired')
tar_test = utils.normalizeInput(tar_test, mode='Paired')

print('source training image shape', str(src_training[0].shape))
print('source training label shape', src_training[1].shape)
print('source training image mean/std', str(src_training[0].mean()), str(src_training[0].std()))

print('source validation image shape', str(src_validation[0].shape))
print('source validation label shape', src_validation[1].shape)
print('source validation image mean/std', str(src_validation[0].mean()), str(src_validation[0].std()))

print('source test image shape', src_test[0].shape)
print('source test label shape', src_test[1].shape)
print('source test image mean/std', str(src_test[0].mean()), str(src_test[0].std()))

print('target training image shape', str(tar_training.shape))
print('target training image mean/std', str(tar_training.mean()), str(tar_training.std()))

print('target test image shape', tar_test[0].shape)
print('target test label shape', tar_test[1].shape)
print('target test image mean/std', str(tar_test[0].mean()), str(tar_test[0].std()))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    res_model = cpb_dastage2_model(model_name=args.model_name,
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
