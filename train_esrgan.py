from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from modules.models import RRDB_Model, DiscriminatorVGG128
from modules.lr_scheduler import MultiStepLR
from modules.losses import (PixelLoss, ContentLoss, DiscriminatorLoss,
                            GeneratorLoss)
from modules.utils import (load_yaml, load_dataset, ProgressBar,
                           set_memory_growth)
import time
flags.DEFINE_string('cfg_path', './configs/esrgan.yaml', 'config file path')
flags.DEFINE_string('gpu', '1', 'which gpu to use')

def main(_):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network (Generator and Discriminator)
    generator = RRDB_Model(cfg['input_size'], cfg['ch_size'], cfg['network_G'])
    discriminator = DiscriminatorVGG128(cfg['gt_size'], cfg['ch_size'])
    generator.summary(line_length=80)
    discriminator.summary(line_length=80)

    # load dataset without shuffle
    train_dataset = load_dataset(cfg, 'train_dataset', shuffle=False)

    # define Adam optimizer
    learning_rate_G = MultiStepLR(cfg['lr_G'], cfg['lr_steps'], cfg['lr_rate'])
    learning_rate_D = MultiStepLR(cfg['lr_D'], cfg['lr_steps'], cfg['lr_rate'])
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=learning_rate_G,
                                           beta_1=cfg['adam_beta1_G'],
                                           beta_2=cfg['adam_beta2_G'])
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=learning_rate_D,
                                           beta_1=cfg['adam_beta1_D'],
                                           beta_2=cfg['adam_beta2_D'])

    # define losses function
    pixel_loss_fn = PixelLoss(criterion=cfg['pixel_criterion'])
    fea_loss_fn = ContentLoss(criterion=cfg['feature_criterion'])
    gen_loss_fn = GeneratorLoss(gan_type=cfg['gan_type'])
    dis_loss_fn = DiscriminatorLoss(gan_type=cfg['gan_type'])

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer_G=optimizer_G,
                                     optimizer_D=optimizer_D,
                                     model=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=20)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        if cfg['pretrain_name'] is not None:
            pretrain_dir = './checkpoints/' + cfg['pretrain_name']
            if tf.train.latest_checkpoint(pretrain_dir):
                checkpoint.restore(tf.train.latest_checkpoint(pretrain_dir))
                checkpoint.step.assign(0)
                print("[*] training from pretrain model {}.".format(pretrain_dir))
            else:
                print("[*] cannot find pretrain model {}.".format(pretrain_dir))
        else:
            print("[*] training from scratch.")

    # define training step function
    @tf.function
    def train_step(lr, hr):
        # get output and loss
        with tf.GradientTape(persistent=True) as tape:
            sr = generator(lr, training=True)
            hr_output = discriminator(hr, training=True)
            sr_output = discriminator(sr, training=True)
            losses_G = {}
            losses_G['reg'] = tf.reduce_sum(generator.losses)
            losses_G['pixel'] = cfg['w_pixel'] * pixel_loss_fn(hr, sr) # pixel loss
            losses_G['feature'] = cfg['w_feature'] * fea_loss_fn(hr, sr) # feature loss
            losses_G['gan'] = cfg['w_gan'] * gen_loss_fn(hr_output, sr_output) # generator loss
            total_loss_G = tf.add_n([l for l in losses_G.values()])
            losses_D = {}
            losses_D['reg'] = tf.reduce_sum(discriminator.losses)
            losses_D['gan'] = dis_loss_fn(hr_output, sr_output) # discriminator loss
            total_loss_D = tf.add_n([l for l in losses_D.values()])
        grads_G = tape.gradient(total_loss_G, generator.trainable_variables)
        grads_D = tape.gradient(total_loss_D, discriminator.trainable_variables)
        optimizer_G.apply_gradients(zip(grads_G, generator.trainable_variables))
        optimizer_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))
        return total_loss_G, total_loss_D, losses_G, losses_D

    # training loop
    summary_writer = tf.summary.create_file_writer(
        './logs/' + cfg['sub_name'])
    # prog_bar = ProgressBar(cfg['niter'], checkpoint.step.numpy())
    remain_steps = max(cfg['niter'] - checkpoint.step.numpy(), 0)
    cnter = remain_steps
    # start training
    for lr, hr in train_dataset.take(remain_steps):
        cnter -= 1
        t_start = time.time()
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()
        total_loss_G, total_loss_D, losses_G, losses_D = train_step(lr, hr)
        # visualize
        # prog_bar.update("loss_G={:.4f}, loss_D={:.4f}, lr_G={:.1e}, lr_D={:.1e}".format(total_loss_G.numpy(), total_loss_D.numpy(),optimizer_G.lr(steps).numpy(), optimizer_D.lr(steps).numpy()))
        stps_epoch = int(cfg['train_dataset']['num_samples']/cfg['batch_size'])
        t_end = time.time()
        current_epoch = int(steps/stps_epoch)
        print("epoch=%3d step=%4d/%d G_loss=%3.4f D_loss=%3.4f G_lr=%.5f D_lr=%.5f stp_time=%.3f cnter=%6d"%(current_epoch,int(steps%stps_epoch),stps_epoch,total_loss_G.numpy(),total_loss_D.numpy(),optimizer_G.lr(steps).numpy(),optimizer_D.lr(steps).numpy(),t_end-t_start,cnter))
        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('loss_G/total_loss', total_loss_G, step=steps)
                tf.summary.scalar('loss_D/total_loss', total_loss_D, step=steps)
                for k, l in losses_G.items():
                    tf.summary.scalar('loss_G/{}'.format(k), l, step=steps)
                for k, l in losses_D.items():
                    tf.summary.scalar('loss_D/{}'.format(k), l, step=steps)
                tf.summary.scalar('learning_rate_G', optimizer_G.lr(steps), step=steps)
                tf.summary.scalar('learning_rate_D', optimizer_D.lr(steps), step=steps)
        # save checkpoint
        if (steps % stps_epoch == 0):
            manager.save()
            print("\n[*] save ckpt file at {}".format(manager.latest_checkpoint))
    print("\n [*] training done!")

if __name__ == '__main__':
    app.run(main)
