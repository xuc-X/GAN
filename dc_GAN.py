"""
代码模块内容:
1. 数据(Data provider):
    a.Image data
    b.random vector
2. 计算图构建(Build compute graph)
    a. generator
    b. discriminator
    c. DCGAN
        connect g and d
3. 训练流程(Training process)
"""
import os
import sys
import tensorflow as tf
from tensorflow import logging
from tensorflow import gfile
import pprint
import pickle
import numpy as np
import random
import math
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
output_dir = './local_run'
if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)


def get_default_params():
    return tf.contrib.training.HParams(
        z_dim=100,  # 用于向量变矩阵
        init_conv_size=4,  #
        g_channels=[128, 64, 32, 1],  # 生成器每层通道数目
        d_channels=[32, 64, 128, 256],  # 判别器每层通道数目
        batch_size=128,
        learning_rate=0.002,
        beta=0.5,
        img_size=32,
    )


hps = get_default_params()


# print(hps.img_size)
# print(hps.g_channels)
# print(mnist.train.images.shape) # (55000, 784)

class Mnist(object):
    def __init__(self, mnist_train, z_dim, img_size):
        self._data = mnist_train
        self._example_num = len(self._data)
        self._z_data = np.random.standard_normal((self._example_num, z_dim))
        self._indicator = 0
        self._resize_mnist_img(img_size)
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(self._example_num)
        self._z_data = self._z_data[p]
        self._data = self._data[p]

    def _resize_mnist_img(self, img_size):
        """
        Resize mnist image to goal img_size
        How?
        1.numpy -> PIL image
        2.PIL image -> resize
        3.PIL image -> numpy
        """
        data = np.asarray(self._data * 255, np.uint8)
        data = data.reshape((self._example_num, 28, 28))
        new_data = []
        for i in range(self._example_num):
            img = data[i]
            img = Image.fromarray(img)
            # 会自动插值
            img = img.resize((img_size, img_size))
            img = np.asarray(img)
            img = img.reshape((img_size, img_size, 1))
            new_data.append(img)
        new_data = np.asarray(new_data, dtype=np.float32)
        new_data = (new_data / 127.5) - 1  # 归一化到-1到1之间
        # self._data: [num_example,img_size,img_size,1]
        self._data = new_data

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._example_num:
            self._random_shuffle()
            self._indicator = 0
            end_indicator = self._indicator + batch_size
        assert end_indicator < self._example_num

        batch_data = self._data[self._indicator: end_indicator]
        batch_z = self._z_data[self._indicator: end_indicator]
        return batch_data, batch_z


mnist_data = Mnist(mnist.train.images, hps.z_dim, hps.img_size)
batch_data, batch_z = mnist_data.next_batch(5)
# print(batch_data.shape)
# print(batch_data[0][16, :])
# print(batch_z)


# ------------------------------------------------------------------------------------------------------------
def conv2d_transpose(inputs, out_channel, name, training, with_bn_relu=True):
    """Wrapper of conv2d transpose"""
    with tf.variable_scope(name):
        conv2d_trans = tf.layers.conv2d_transpose(inputs, out_channel, [5, 5], strides=(2, 2), padding='SAME')
        if with_bn_relu:
            bn = tf.layers.batch_normalization(conv2d_trans, training=training)
            relu = tf.nn.relu(bn)
            return relu
        else:
            return conv2d_trans


class Generator(object):
    def __init__(self, channels, init_conv_size):
        self._channels = channels
        self._init_conv_size = init_conv_size
        self._reuse = False

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('generator', reuse=self._reuse):
            """
            random_vector -> fc -> self._channels[0] * init_conv_size**2 ->reshape ->[init_conv_size,init_conv_size,channels]
            """
            with tf.variable_scope('inputs_conv'):
                fc = tf.layers.dense(inputs, self._channels[0] * self._init_conv_size * self._init_conv_size)
                conv0 = tf.reshape(fc, [-1, self._init_conv_size, self._init_conv_size, self._channels[0]])
                bn0 = tf.layers.batch_normalization(conv0, training=training)
                relu0 = tf.nn.relu(bn0)
            deconv_inputs = relu0
            for i in range(1, len(self._channels)):
                with_bn_relu = (i != (len(self._channels) - 1))
                deconv_inputs = conv2d_transpose(deconv_inputs, self._channels[i], 'deconv_{}'.format(i), training,
                                                 with_bn_relu)
            img_inputs = deconv_inputs
            with tf.variable_scope('generate_imgs'):
                # img values
                imgs = tf.tanh(img_inputs, name='imgs')
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return imgs


def conv2d(inputs, out_channel, name, training):
    def leak_relu(x, leak=0.2,name=''):
        return tf.maximum(x, x * leak, name=name)

    with tf.variable_scope(name):
        conv2d_output = tf.layers.conv2d(inputs, out_channel, [5, 5], strides=(2, 2), padding='SAME')
        bn = tf.layers.batch_normalization(conv2d_output, training=training)
        return leak_relu(bn, name='outputs')


class Discriminator(object):
    def __init__(self, channels):
        self._channels = channels
        self._reuse = False

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        conv_inputs = inputs
        with tf.variable_scope('discriminator', reuse=self._reuse):
            for i in range(len(self._channels)):
                conv_inputs = conv2d(conv_inputs, self._channels[i],'conv_{}'.format(i), training)
        fc_inputs = conv_inputs
        with tf.variable_scope('fc',reuse=self._reuse):
            flatten = tf.layers.flatten(fc_inputs)
            logits = tf.layers.dense(flatten, 2, name='logits')
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        print('discriminator',self.variables)
        return logits


class DCGAN(object):
    def __init__(self, hps):
        g_channels = hps.g_channels
        d_channels = hps.d_channels
        self._batch_size = hps.batch_size
        self._init_conv_size = hps.init_conv_size
        self._z_dim = hps.z_dim
        self._img_size = hps.img_size
        self._generator = Generator(g_channels, self._init_conv_size)
        self._discriminator = Discriminator(d_channels)

    def build(self):
        """Builds the whole compute graph"""
        self._z_placeholder = tf.placeholder(tf.float32, (self._batch_size, self._z_dim))
        self._img_placeholder = tf.placeholder(tf.float32, (self._batch_size, self._img_size, self._img_size, 1))
        generated_imgs = self._generator(self._z_placeholder, training=True)
        fake_img_logits = self._discriminator(generated_imgs, training=True)
        real_img_logits = self._discriminator(self._img_placeholder, training=True)
        loss_on_fake_to_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([self._batch_size], dtype=tf.int64),
                                                           logits=fake_img_logits))
        loss_on_fake_to_fake = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([self._batch_size], dtype=tf.int64),
                                                           logits=fake_img_logits))
        loss_on_real_to_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([self._batch_size], dtype=tf.int64),
                                                           logits=real_img_logits))
        tf.add_to_collection('g_losses', loss_on_fake_to_real)
        tf.add_to_collection('d_losses', loss_on_fake_to_fake)
        tf.add_to_collection('d_losses', loss_on_real_to_real)

        loss = {
            'g': tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            'd': tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')
        }
        return (self._z_placeholder,
                self._img_placeholder,
                generated_imgs,
                loss)

    def build_train_op(self, losses, learning_rate, beta1):
        """Build train op,should be called after build is called"""
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
        g_opt_op = g_opt.minimize(losses['g'], var_list=self._generator.variables)
        d_opt_op = d_opt.minimize(losses['d'], var_list=self._discriminator.variables)
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name='train')


dcgan = DCGAN(hps)
z_placeholder, img_placeholder, generated_imgs, losses = dcgan.build()
train_op = dcgan.build_train_op(losses, hps.learning_rate, hps.beta)

def combine_imgs(batch_imgs,img_size,rows=8,cols=16):
    """Combines small images in a  batch into a big pic"""
    result_big_img = []
    for i in range(rows):
        row_imgs = []
        for j in range(cols):
            img = batch_imgs[cols * i + j]
            img = img.reshape((img_size,img_size))
            img = (img + 1) * 127.5
            row_imgs.append(img)
        row_imgs = np.hstack(row_imgs)
        result_big_img.append(row_imgs)
    result_big_img = np.vstack(result_big_img)
    result_big_img = np.asarray(result_big_img,np.uint8)
    result_big_img = Image.fromarray(result_big_img)
    return result_big_img

init_op = tf.global_variables_initializer()
train_steps = 10000
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(train_steps):
        batch_imgs, batch_z = mnist_data.next_batch(hps.batch_size)
        fetches = [train_op,losses['g'],losses['d']]
        should_sample = (step + 1) % 50 == 0
        if should_sample:
            fetches += [generated_imgs]
        output_values = sess.run(fetches,feed_dict={
            z_placeholder:batch_z,
            img_placeholder:batch_imgs
        })
        _,g_loss_val,d_loss_val = output_values[0:3]
        print("step: %4d, g_loss: %4.10f,d_loss %4.10f" % (step,g_loss_val,d_loss_val))
        if should_sample:
            gen_imgs_val = output_values[3]
            gen_img_path = os.path.join(output_dir,"{}-gen.jpg".format(step+1))
            gt_img_path = os.path.join( output_dir,"{}-gt.jpg".format(step+1))
            gen_img = combine_imgs(gen_imgs_val,hps.img_size)
            gt_img = combine_imgs(batch_imgs,hps.img_size)
            gen_img.save(gen_img_path)
            gt_img.save(gt_img_path)