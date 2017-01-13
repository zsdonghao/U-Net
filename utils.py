#! /usr/bin/python
# -*- coding: utf8 -*-



"""
License
=======

Copyright (c) 2016 Data Science Institute, Imperial College London.  All rights reserved.

                                    Oct 2016

Contact
=======
Questions? Please contact hao.dong11@imperial.ac.uk

"""

import tensorflow as tf
import tensorlayer as tl
import nibabel as nib
import numpy as np
import os
# from tensorlayer.activation import pixel_wise_softmax
# import sys  # this line added
# sys.setrecursionlimit(1000000)  # this line added

import skimage
# from skimage.transform import swirl


## loss and matrix
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.

  References
  -----------
  - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)
    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)
    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def cross_entropy_weight(y_, output_map, weight=1, name="cross_entropy_weight"):
    """Compute cost entropy with two images, note that it do not compute softmax internally.

    Fangde : haven't used

    tf.clip_by_value to avoid NaN

    Parameters
    -----------
    y_ : 4D tensor [batch_size, height, weight, channel]
        target outputs
    output_map : 4D tensor [batch_size, height, weight, channel]
        predict outputs

    Examples
    ---------
    >>> outputs = pixel_wise_softmax(network.outputs)
    >>> wc = cross_entropy_weight(y_, outputs, weight=20)
    """
    return -tf.reduce_mean(weight * y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name=name)


## Load file
def read_Nifti1Image(file_dir, name):
    """
    http://nipy.org/nibabel/gettingstarted.html
    """
    img_dir = os.path.join(file_dir, name)
    img = nib.load(img_dir)
    print("  *Name: %s Shape: %s " % (name, img.shape))
    # print(type(img))            # <class 'nibabel.nifti1.Nifti1Image'>
    # print(img.get_data_dtype() == np.dtype(np.int16))   # True
    # return np.array(img, dtype=np.float32)
    return img

def prepare_data(file_dir, file_list, shape=(), threshold=None):
    print(" * Preparing %s" %file_list)
    data = np.empty(shape=(0,shape[1],shape[2],1))
    for f in file_list:
        img = read_Nifti1Image(file_dir, f)
        X = img.get_data()
        X = np.transpose(X, (1,0,2))
        X = X[:,:,:,np.newaxis]
        #
        if threshold:
            X = (X > threshold).astype(int)
        else:
            X = X/np.max(X)
        if X.shape == shape:
            data = np.vstack((data, X))
        else:
            print("    *shape don't match")
    return data

def prepare_data2(file_dir, file_list, label_list, shape=(), dim_order=(1,0,2)):
    print(" * Preparing %s %s" % (file_list, label_list))
    # data = np.empty(shape=(0,shape[1],shape[2],1))    ######## Akara : slower than list append
    # data2 = np.empty(shape=(0,shape[1],shape[2],1))
    data = []
    data2 = []
    # j = 0
    for f, f2 in zip(file_list, label_list):
        print("%s - %s" % (f, f2))
        ## read original image
        img = read_Nifti1Image(file_dir, f)
        X = img.get_data()
        X = np.transpose(X, dim_order)
        X = X[:,:,:,np.newaxis]
        ## read label image
        img = read_Nifti1Image(file_dir, f2)
        Y = img.get_data()
        Y = np.transpose(Y, dim_order)
        Y = Y[:,:,:,np.newaxis]
        # print(X.shape, shape)
        ## if shape correct
        if X.shape == shape:
            for i in range(Y.shape[0]):
                # print(i, 'Y', np.mean(Y[i]), np.max(Y[i]))
                # print(i, 'X', np.mean(X[i]), np.max(X[i]))
                ## if label exists
                if np.max(Y[i]) > 0:
                    ## display data values
                    # print('%d Y max:%.3f mean:%.3f' % (i, np.max(Y[i]), np.mean(Y[i])))#, np.median(Y[i]))
                    print('%d X max:%.3f min:%.3f' % (i, np.max(X[i]), np.min(X[i])))#, np.median(X[i]))
                    ## make image [0,1]
                    # X[i] = (X[i] - np.min(X[i])) / (np.max(X[i]) - np.min(X[i]))
                    ## make label binary
                    Y[i] = (Y[i] > 0.5).astype(int)
                    mask = (Y[i] != 2).astype(int)
                    Y[i] = Y[i] * mask
                    # Y[i] = (Y[i] == 4).astype(int)
                    # print(j)
                    # j+= 1
                    # print(i, np.mean(Y[i]), np.max(Y[i]))
                    # print(data2.shape,Y[i].shape)
                    ## stack data
                    # data = np.vstack((data, [X[i]]))  ###### Akara : slower than list append
                    # data2 = np.vstack((data2, [Y[i]]))
                    # print(X[i].dtype)  # float 32
                    data.append(X[i].astype(np.float32))
                    data2.append(Y[i].astype(np.float32))
        else:
            print("    *shape doesn't match")
        ## plot an example
        # for i in range(0, data.shape[0], 1):
        #     # tl.visualize.frame(X[i,:,:,0], second=0.01, saveable=False, name='slice x:'+str(i),cmap='gray')
        #     tl.visualize.images2d(images=np.asarray([data[i,:,:,:], data2[i,:,:,:]]), second=0.01, saveable=False, name='slice x:'+str(i), dtype=None)
        # exit()
    return np.asarray(data, dtype=np.float32), np.asarray(data2, dtype=np.float32)

def train(total_loss, global_step, init_lr, decay_factor, num_batches_per_epoch, num_epoch_per_decay, mode='sgd'):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  MOVING_AVERAGE_DECAY = 0.9999
  def _add_loss_summaries(total_loss):
      """Add summaries for losses in CIFAR-10 model.
      Generates moving average for all losses and associated summaries for
      visualizing the performance of the network.
      Args:
        total_loss: Total loss from loss().
      Returns:
        loss_averages_op: op for generating moving averages of losses.

        https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10.py
      """
      # Compute the moving average of all individual losses and the total loss.
      loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
      losses = tf.get_collection('losses')
      loss_averages_op = loss_averages.apply(losses + [total_loss])

      # Attach a scalar summary to all individual losses and the total loss; do the
      # same for the averaged version of the losses.
      for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

      return loss_averages_op
  # Variables that affect learning rate.
  # num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * num_epoch_per_decay)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(init_lr,
                                  global_step,
                                  decay_steps,
                                  decay_factor,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    if mode == 'sgd':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif mode == 'adam':
        opt = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False)
    else:
        raise Exception("%s not support" % mode)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


## Model
def u_net_2d_64_1024_deconv_pro(x, n_out=2):
    """ 2-D U-Net for Image Segmentation.

    Parameters
    -----------
    x : tensor or placeholder of input with shape of [batch_size, row, col, channel]
    batch_size : int, batch size
    n_out : int, number of output channel, default is 2 for foreground and background (binary segmentation)

    Returns
    --------
    network : TensorLayer layer class with identity output
    outputs : tensor, the output with pixel-wise softmax

    Notes
    -----
    - Recommend to use Adam with learning rate of 1e-5
    """
    batch_size = int(x._shape[0])
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
    ## define initializer
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    ## u-net model
    # convolution
    # with tf.device('\gpu:0'):
    net_in = tl.layers.InputLayer(x, name='input')
    conv1 = tl.layers.Conv2dLayer(net_in, act=tf.nn.relu,
                shape=[3,3,nz,64], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv1')
    conv2 = tl.layers.Conv2dLayer(conv1, act=tf.nn.relu,
                shape=[3,3,64,64], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv2')
    pool1 = tl.layers.PoolLayer(conv2, ksize=[1,2,2,1],
                strides=[1,2,2,1], padding='SAME',
                pool=tf.nn.max_pool, name='pool1')
    conv3 = tl.layers.Conv2dLayer(pool1, act=tf.nn.relu,
                shape=[3,3,64,128], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv3')
    conv4 = tl.layers.Conv2dLayer(conv3, act=tf.nn.relu,
                shape=[3,3,128,128], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv4')
    pool2 = tl.layers.PoolLayer(conv4, ksize=[1,2,2,1],
                strides=[1,2,2,1], padding='SAME',
                pool=tf.nn.max_pool, name='pool2')
    conv5 = tl.layers.Conv2dLayer(pool2, act=tf.nn.relu,
                shape=[3,3,128,256], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv5')
    conv6 = tl.layers.Conv2dLayer(conv5, act=tf.nn.relu,
                shape=[3,3,256,256], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv6')
    pool3 = tl.layers.PoolLayer(conv6, ksize=[1,2,2,1],
                strides=[1,2,2,1], padding='SAME',
                pool=tf.nn.max_pool, name='pool3')
    conv7 = tl.layers.Conv2dLayer(pool3, act=tf.nn.relu,
                shape=[3,3,256,512], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv7')
    conv8 = tl.layers.Conv2dLayer(conv7, act=tf.nn.relu,
                shape=[3,3,512,512], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv8')
    # print(conv8.outputs)    # (10, 30, 30, 512)
    pool4 = tl.layers.PoolLayer(conv8, ksize=[1,2,2,1],
                strides=[1,2,2,1], padding='SAME',
                pool=tf.nn.max_pool,name='pool4')
    conv9 = tl.layers.Conv2dLayer(pool4, act=tf.nn.relu,
                shape=[3,3,512,1024], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv9')
    conv10 = tl.layers.Conv2dLayer(conv9, act=tf.nn.relu,
                shape=[3,3,1024,1024], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv10')
    print(" * After conv: %s" % conv10.outputs)   # (batch_size, 32, 32, 1024)
    # deconvoluation
    deconv1 = tl.layers.DeConv2dLayer(conv10, act=tf.identity, #act=tf.nn.relu,
                shape=[3,3,512,1024], strides=[1,2,2,1], output_shape=[batch_size,nx/8,ny/8,512],
                padding='SAME', W_init=w_init, b_init=b_init, name='devcon1_1')
    # print(deconv1.outputs)  #(10, 30, 30, 512)
    deconv1_2 = tl.layers.ConcatLayer([conv8, deconv1], concat_dim=3, name='concat1_2')
    deconv1_3 = tl.layers.Conv2dLayer(deconv1_2, act=tf.nn.relu,
                shape=[3,3,1024,512], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv1_3')
    deconv1_4 = tl.layers.Conv2dLayer(deconv1_3, act=tf.nn.relu,
                shape=[3,3,512,512], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv1_4')
    deconv2 = tl.layers.DeConv2dLayer(deconv1_4, act=tf.identity, #act=tf.nn.relu,
                shape=[3,3,256,512], strides=[1,2,2,1], output_shape=[batch_size,nx/4,ny/4,256],
                padding='SAME', W_init=w_init, b_init=b_init, name='devcon2_1')
    deconv2_2 = tl.layers.ConcatLayer([conv6, deconv2], concat_dim=3, name='concat2_2')
    deconv2_3 = tl.layers.Conv2dLayer(deconv2_2, act=tf.nn.relu,
                shape=[3,3,512,256], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv2_3')
    deconv2_4 = tl.layers.Conv2dLayer(deconv2_3, act=tf.nn.relu,
                shape=[3,3,256,256], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv2_4')
    deconv3 = tl.layers.DeConv2dLayer(deconv2_4, act=tf.identity, #act=tf.nn.relu,
                shape=[3,3,128,256], strides=[1,2,2,1], output_shape=[batch_size,nx/2,ny/2,128],
                padding='SAME', W_init=w_init, b_init=b_init, name='devcon3_1')
    deconv3_2 = tl.layers.ConcatLayer([conv4, deconv3], concat_dim=3, name='concat3_2')
    deconv3_3 = tl.layers.Conv2dLayer(deconv3_2, act=tf.identity, #act=tf.nn.relu,
                shape=[3,3,256,128], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv3_3')
    deconv3_4 = tl.layers.Conv2dLayer(deconv3_3, act=tf.nn.relu,
                shape=[3,3,128,128], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv3_4')
    deconv4 = tl.layers.DeConv2dLayer(deconv3_4, act=tf.identity, #act=tf.nn.relu,
                shape=[3,3,64,128], strides=[1,2,2,1], output_shape=[batch_size,nx,ny,64],
                padding='SAME', W_init=w_init, b_init=b_init, name='devconv4_1')
    deconv4_2 = tl.layers.ConcatLayer([conv2, deconv4], concat_dim=3, name='concat4_2')
    deconv4_3 = tl.layers.Conv2dLayer(deconv4_2, act=tf.nn.relu,
                shape=[3,3,128,64], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv4_3')
    deconv4_4 = tl.layers.Conv2dLayer(deconv4_3, act=tf.nn.relu,
                shape=[3,3,64,64], strides=[1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv4_4')
    network = tl.layers.Conv2dLayer(deconv4_4,
                act=tf.identity,
                shape=[1,1,64,n_out],       # [0]:foreground prob; [1]:background prob
                strides=[1,1,1,1],
                padding='SAME',
                W_init=w_init, b_init=b_init, name='conv4_5')
    # compute the softmax output
    print(" * Output: %s" % network.outputs)
    outputs = tl.act.pixel_wise_softmax(network.outputs)
    return network, outputs

def u_net_2d_64_1024_deconv(x, n_out=2):
    from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    inputs = InputLayer(x, name='inputs')

    conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
    conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
    pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')

    conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
    conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
    pool2 = MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')

    conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
    conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
    pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')

    conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
    conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
    pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')

    conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
    conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')

    print(" * After conv: %s" % conv5.outputs)

    up4 = DeConv2d(conv5, 512, (3, 3), out_size = (nx/8, ny/8), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv4')
    up4 = ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
    conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_1')
    conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_2')

    up3 = DeConv2d(conv4, 256, (3, 3), out_size = (nx/4, ny/4), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv3')
    up3 = ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
    conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_1')
    conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_2')

    up2 = DeConv2d(conv3, 128, (3, 3), out_size = (nx/2, ny/2), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv2')
    up2 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat2')
    conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_1')
    conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_2')

    up1 = DeConv2d(conv2, 64, (3, 3), out_size = (nx/1, ny/1), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv1')
    up1 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat1')
    conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_1')
    conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_2')

    conv1 = Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
    print(" * Output: %s" % conv1.outputs)
    outputs = tl.act.pixel_wise_softmax(conv1.outputs)
    return conv1, outputs

def u_net_2d_64_1024_deconv_resnet(x, n_out=2): #TODO
    from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
    # batch_size = int(x._shape[0])
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    inputs = InputLayer(x, name='inputs')

    conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
    conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
    pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')

    conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
    conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
    pool2 = MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')

    conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
    conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
    pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')

    conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
    conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
    pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')

    conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
    conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')

    print(" * After conv: %s" % conv5.outputs)

    up4 = DeConv2d(conv5, 512, (3, 3), out_size = (nx/8, ny/8), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv4')
    up4 = ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
    conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_1')
    conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_2')

    up3 = DeConv2d(conv4, 256, (3, 3), out_size = (nx/4, ny/4), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv3')
    up3 = ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
    conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_1')
    conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_2')

    up2 = DeConv2d(conv3, 128, (3, 3), out_size = (nx/2, ny/2), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv2')
    up2 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat2')
    conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_1')
    conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_2')

    up1 = DeConv2d(conv2, 64, (3, 3), out_size = (nx/1, ny/1), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv1')
    up1 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat1')
    conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_1')
    conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_2')

    conv1 = Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
    print(" * Output: %s" % conv1.outputs)
    outputs = tl.act.pixel_wise_softmax(conv1.outputs)
    return conv1, outputs

def u_net_2d_64_2048_deconv(x, n_out=2):
    from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
    # batch_size = int(x._shape[0])
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    inputs = InputLayer(x, name='inputs')

    conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
    conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
    pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')

    conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
    conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
    pool2 = MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')

    conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
    conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
    pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')

    conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
    conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
    pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')

    conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
    conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
    pool5 = MaxPool2d(conv5, (2, 2), padding='SAME', name='pool5')

    conv6 = Conv2d(pool5, 2048, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_1')
    conv6 = Conv2d(conv6, 2048, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_2')

    print(" * After conv: %s" % conv6.outputs)

    up5 = DeConv2d(conv6, 1024, (3, 3), out_size = (nx/16, ny/16), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv5')
    up5 = ConcatLayer([up5, conv5], concat_dim=3, name='concat5')
    conv5 = Conv2d(up5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv5_1')
    conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv5_2')

    up4 = DeConv2d(conv5, 512, (3, 3), out_size = (nx/8, ny/8), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv4')
    up4 = ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
    conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_1')
    conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_2')

    up3 = DeConv2d(conv4, 256, (3, 3), out_size = (nx/4, ny/4), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv3')
    up3 = ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
    conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_1')
    conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_2')

    up2 = DeConv2d(conv3, 128, (3, 3), out_size = (nx/2, ny/2), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv2')
    up2 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat2')
    conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_1')
    conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_2')

    up1 = DeConv2d(conv2, 64, (3, 3), out_size = (nx/1, ny/1), strides = (2, 2),
                                padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv1')
    up1 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat1')
    conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_1')
    conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_2')

    conv1 = Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
    print(" * Output: %s" % conv1.outputs)
    outputs = tl.act.pixel_wise_softmax(conv1.outputs)
    return conv1, outputs

def u_net_2d_32_512_upsam(x, n_out=2):
    """
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    """
    from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
    batch_size = int(x._shape[0])
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
    ## define initializer
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    inputs = InputLayer(x, name='inputs')
    # inputs = Input((1, img_rows, img_cols))
    conv1 = Conv2d(inputs, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
    # print(conv1.outputs) # (10, 240, 240, 32)
    # conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
    # print(conv1.outputs)    # (10, 240, 240, 32)
    # conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print(pool1.outputs)    # (10, 120, 120, 32)
    # exit()
    conv2 = Conv2d(pool1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
    # conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Conv2d(conv2, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
    # conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPool2d(conv2, (2,2), padding='SAME', name='pool2')
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2d(pool2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
    # conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Conv2d(conv3, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
    # conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # print(pool3.outputs)   # (10, 30, 30, 64)

    conv4 = Conv2d(pool3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
    # print(conv4.outputs)    # (10, 30, 30, 256)
    # conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Conv2d(conv4, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
    # print(conv4.outputs)    # (10, 30, 30, 256) != (10, 30, 30, 512)
    # conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2d(pool4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
    # conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Conv2d(conv5, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
    # conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    # print(conv5.outputs)    # (10, 15, 15, 512)
    print(" * After conv: %s" % conv5.outputs)
    # print(nx/8,ny/8) # 30 30
    up6 = UpSampling2dLayer(conv5, (2, 2), name='up6')
    # print(up6.outputs)  # (10, 30, 30, 512) == (10, 30, 30, 512)
    up6 = ConcatLayer([up6, conv4], concat_dim=3, name='concat6')
    # print(up6.outputs)  # (10, 30, 30, 768)
    # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2d(up6, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_1')
    # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Conv2d(conv6, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_2')
    # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = UpSampling2dLayer(conv6, (2, 2), name='up7')
    up7 = ConcatLayer([up7, conv3] ,concat_dim=3, name='concat7')
    # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2d(up7, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_1')
    # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Conv2d(conv7, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_2')
    # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = UpSampling2dLayer(conv7, (2, 2), name='up8')
    up8 = ConcatLayer([up8, conv2] ,concat_dim=3, name='concat8')
    # up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2d(up8, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_1')
    # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Conv2d(conv8, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_2')
    # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = UpSampling2dLayer(conv8, (2, 2), name='up9')
    up9 = ConcatLayer([up9, conv1] ,concat_dim=3, name='concat9')
    # up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Conv2d(up9, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_1')
    # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Conv2d(conv9, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_2')
    # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Conv2d(conv9, n_out, (1, 1), act=None, name='conv9')
    # conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    print(" * Output: %s" % conv10.outputs)
    outputs = tl.act.pixel_wise_softmax(conv10.outputs)
    return conv10, outputs

def u_net_2d_32_1024_upsam(x, n_out=2):
    """
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    """
    from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
    batch_size = int(x._shape[0])
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
    ## define initializer
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    inputs = InputLayer(x, name='inputs')

    conv1 = Conv2d(inputs, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
    conv1 = Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
    pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')

    conv2 = Conv2d(pool1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
    conv2 = Conv2d(conv2, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
    pool2 = MaxPool2d(conv2, (2,2), padding='SAME', name='pool2')

    conv3 = Conv2d(pool2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
    conv3 = Conv2d(conv3, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
    pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')

    conv4 = Conv2d(pool3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
    conv4 = Conv2d(conv4, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
    pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')

    conv5 = Conv2d(pool4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
    conv5 = Conv2d(conv5, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
    pool5 = MaxPool2d(conv5, (2, 2), padding='SAME', name='pool6')

    # hao add
    conv6 = Conv2d(pool5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_1')
    conv6 = Conv2d(conv6, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_2')

    print(" * After conv: %s" % conv6.outputs)

    # hao add
    up7 = UpSampling2dLayer(conv6, (15, 15), is_scale=False, method=1, name='up7')
    up7 =  ConcatLayer([up7, conv5], concat_dim=3, name='concat7')
    conv7 = Conv2d(up7, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_1')
    conv7 = Conv2d(conv7, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_2')

    # print(nx/8,ny/8) # 30 30
    up8 = UpSampling2dLayer(conv7, (2, 2), method=1, name='up8')
    up8 = ConcatLayer([up8, conv4], concat_dim=3, name='concat8')
    conv8 = Conv2d(up8, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_1')
    conv8 = Conv2d(conv8, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_2')

    up9 = UpSampling2dLayer(conv8, (2, 2), method=1, name='up9')
    up9 = ConcatLayer([up9, conv3] ,concat_dim=3, name='concat9')
    conv9 = Conv2d(up9, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_1')
    conv9 = Conv2d(conv9, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_2')

    up10 = UpSampling2dLayer(conv9, (2, 2), method=1, name='up10')
    up10 = ConcatLayer([up10, conv2] ,concat_dim=3, name='concat10')
    conv10 = Conv2d(up10, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv10_1')
    conv10 = Conv2d(conv10, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv10_2')

    up11 = UpSampling2dLayer(conv10, (2, 2), method=1, name='up11')
    up11 = ConcatLayer([up11, conv1] ,concat_dim=3, name='concat11')
    conv11 = Conv2d(up11, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv11_1')
    conv11 = Conv2d(conv11, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv11_2')

    conv12 = Conv2d(conv11, n_out, (1, 1), act=None, name='conv12')
    print(" * Output: %s" % conv12.outputs)
    outputs = tl.act.pixel_wise_softmax(conv12.outputs)
    return conv10, outputs
