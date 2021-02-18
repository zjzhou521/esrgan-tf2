from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf

from modules.models import RRDB_Model
from modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim)

flags.DEFINE_string('cfg_path', './configs/esrgan.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
# flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_integer('is_validate',0, 'to validate or merely infer')

def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RRDB_Model(None, cfg['ch_size'], cfg['network_G'])

    # load checkpoint
    checkpoint_dir = './pretrained_downloaded/psnr_pretrain_inference'
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()
    # process all images in the folder
    print("[*] Processing on inputs folder")
    inputs_path = "./inputs"
    outputs_path = "./outputs_downloaded"
    if(FLAGS.is_validate==1): # generate low res and compare
        print("   image_name                   PSNR/SSIM        PSNR/SSIM (higher,better)")
        for img_name in os.listdir(inputs_path):
            if(img_name[0]=='.'):
                continue
            raw_img = cv2.imread(os.path.join(inputs_path, img_name))
            # Generate low resolution image with original images
            lr_img, hr_img = create_lr_hr_pair(raw_img, cfg['scale']) # scale=4
            # lr_img = raw_img
            sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
            bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)
            str_format = "  [{}] Bic={:.2f}db/{:.2f}, SR={:.2f}db/{:.2f}"
            print(str_format.format(
                img_name + ' ' * max(0, 20 - len(img_name)),
                calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)),
                calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))))
            output_img_path = os.path.join(outputs_path, 'ESRGAN_4x_' + img_name)
            # outputs_img = np.concatenate((bic_img, sr_img, hr_img), 1)
            outputs_img = sr_img
            cv2.imwrite(output_img_path, outputs_img)
            output_lr_img_path = os.path.join(outputs_path, 'ESRGAN_1x_' + img_name)
            outputs_lr_img = lr_img
            cv2.imwrite(output_lr_img_path, outputs_lr_img)
        print("[*] write the visual results in {}".format(outputs_path))
    else: # only infer high scale
        print("inferring, scale = 4")
        for img_name in os.listdir(inputs_path):
            print("processing image:",img_name)
            raw_img = cv2.imread(os.path.join(inputs_path, img_name))
            lr_img = raw_img
            sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
            output_img_path = os.path.join(outputs_path, 'ESRGAN_4x_' + img_name)
            outputs_img = sr_img
            cv2.imwrite(output_img_path, outputs_img)
        print("[*] write the visual results in {}".format(outputs_path))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
