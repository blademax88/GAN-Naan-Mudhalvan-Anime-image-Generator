from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import prettytensor as pt

FLAGS = tf.app.flags.FLAGS

NUM_TAGS = 10
NUM_TAGS_TO_USE = 5
CHANNELS = 3
IMAGE_SIZE = 64

optimizer = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.5)
gen_optimizer = lambda: optimizer(FLAGS.learning_rate)
discrim_optimizer = lambda: optimizer(FLAGS.learning_rate)
gen_activation_fn = tf.nn.relu
discrim_activation_fn = pt.ops.leaky_rectify

def read_and_decode_cifar(filename_queue):
    label_bytes = 1
    height = 32
    width = 32
    depth = 3
    image_bytes = height * width * depth
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [depth, height, width])
    image = tf.transpose(depth_major, [1, 2, 0])
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    return image

def read_and_decode1(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'file_bytes': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([NUM_TAGS], tf.float32),
        })

    image = tf.image.decode_jpeg(features['file_bytes'], channels=3, try_recover_truncated=True)
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    if CHANNELS == 1:
        image = tf.reduce_mean(image, reduction_indices=[2], keepdims=True)
    image.set_shape([None, None, None])

    shape = tf.cast(tf.shape(image), tf.float32)
    height_pad = tf.maximum(tf.ceil((96 - shape[0]) / 2), 0)
    height_pad = tf.reshape(height_pad, [1, 1])
    width_pad = tf.maximum(tf.ceil((96 - shape[1]) / 2), 0)
    width_pad = tf.reshape(width_pad, [1, 1])
    height_pad = tf.tile(height_pad, [1, 2])
    width_pad = tf.tile(width_pad, [1, 2])
    paddings = tf.concat(0, [height_pad, width_pad, tf.zeros([1, 2])])
    paddings = tf.cast(paddings, tf.int32)
    image = tf.pad(image, paddings)

    image = tf.random_crop(image, [96, 96, CHANNELS])

    image = tf.image.resize_images(image, IMAGE_SIZE, IMAGE_SIZE, method=tf.image.ResizeMethod.AREA)

    image = tf.image.random_flip_left_right(image)

    label = features['label']
    label = tf.slice(label, [0], [NUM_TAGS_TO_USE])

    return image, label

def read_and_decode2(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'file_bytes': tf.FixedLenFeature([], tf.string),
        })

    image = tf.image.decode_png(features['file_bytes'], channels=3)
    image = tf.cast(image, tf.float32)
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    if CHANNELS == 1:
        image = tf.reduce_mean(image, reduction_indices=[2], keepdims=True)

    image = image * (2. / 255) - 1

    return image

def generator_template():
    starting_size = int(IMAGE_SIZE / (2 ** NUM_LEVELS))
    num_filters = FLAGS.gen_filter_base * (2 ** NUM_LEVELS)
    with tf.variable_scope('generator'):
        tmp = pt.template('input')
        for i in range(FLAGS.gen_fc_layers - 1):
            tmp = tmp.fully_connected(FLAGS.gen_fc_size).apply(gen_activation_fn)
        tmp = tmp.fully_connected(starting_size*starting_size*num_filters/2).apply(gen_activation_fn)
        features = tmp

        tmp = tmp.reshape([FLAGS.batch_size, starting_size, starting_size, num_filters/2])
        for i in range(NUM_LEVELS):
            num_filters = int(num_filters / 2)
            tmp = (
                tmp
                .upsample_conv(5, num_filters)
                .batch_normalize()
                .apply(gen_activation_fn)
            )
        tmp = tmp.conv2d(5, CHANNELS).apply(tf.nn.tanh)
        output = tmp

        z_prediction = (
            features
            .fully_connected(FLAGS.gen_fc_size)
            .apply(gen_activation_fn)
            .fully_connected(FLAGS.gen_fc_size)
            .apply(gen_activation_fn)
            .fully_connected(FLAGS.gen_fc_size)
            .apply(gen_activation_fn)
            .fully_connected(FLAGS.z_size)
        )

    return output, z_prediction

def discriminator_template():
    num_filters = FLAGS.discrim_filter_base
    with tf.variable_scope('discriminator'):
        tmp = pt.template('input')
        for i in range(NUM_LEVELS):
            if i > 0:
                tmp = tmp.dropout(FLAGS.keep_prob)
            tmp = tmp.conv2d(5, num_filters)
            if i > 0:
                tmp = tmp.batch_normalize()
            tmp = tmp.apply(discrim_activation_fn).max_pool(2, 2)
            num_filters *= 2
        tmp = tmp.flatten()
        features = tmp

        minibatch_discrim = features.minibatch_discrimination(100)

        for i in range(FLAGS.discrim_fc_layers-1):
            tmp = tmp.fully_connected(FLAGS.discrim_fc_size).apply(discrim_activation_fn)
        tmp = tmp.concat(1, [minibatch_discrim]).fully_connected(1)
        output = tmp

    return output

def losses(real_images):
    z = tf.truncated_normal([FLAGS.batch_size, FLAGS.z_size], stddev=1) 

    d_template = discriminator_template() 
    g_template = generator_template()

    gen_images, z_prediction = pt.construct_all(g_template, input=z)

    real_logits = d_template.construct(input=real_images)
    fake_logits = d_template.construct(input=gen_images)

    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_logits, tf.ones_like(real_logits)))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logits, tf.zeros_like(fake_logits)))
    discriminator_loss = tf.add(real_loss, fake_loss, name='discriminator_loss')

    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logits, tf.ones_like(fake_logits)), name='generator_loss')

    z_prediction_loss = tf.reduce_mean(tf.square(z - z_prediction), name='z_prediction_loss')

    tf.add_to_collection('losses', generator_loss)
    tf.add_to_collection('losses', discriminator_loss)
    tf.add_to_collection('losses', z_prediction_loss)

    return generator_loss, discriminator_loss, z_prediction_loss

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        with tf.variable_scope('input_pipeline') as scope:
            images = read_and_decode_cifar(input.filename_queue)

        gen_loss, discrim_loss, z_prediction_loss = losses(images)

        for l in tf.get_collection('losses'):
            tf.scalar_summary(l.op.name, l)

        gen_train_op = model.train(gen_loss, global_step, net='generator')
        discrim_train_op = model.train(discrim_loss, global_step, net='discriminator')
        z_predictor_train_op = model.train(z_prediction_loss, global_step, net='z_predictor')

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        discrim_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        variables_to_save = gen_vars
        if FLAGS.save_discriminator:
            variables_to_save += discrim_vars
        saver = tf.train.Saver(variables_to_save, max_to_keep=1, keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)

        summary_op = tf.merge_all_summaries()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))

        init_variables(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        net = 'discriminator'
        count = 0

        for step in range(FLAGS.max_steps):
            if net == 'discriminator':
                op, loss = discrim_train_op, discrim_loss
                count += 1
                if count == FLAGS.discriminator_steps:
                    net = 'generator'
                    count = 0
            else:
                op, loss = gen_train_op, gen_loss
                count += 1
                if count == FLAGS.generator_steps:
                    net = 'discriminator'
                    count = 0

            start_time = time.time()

            _, loss_value = sess.run([op, loss])
            assert not np.isnan(loss_value), 'Model diverged with NaN loss value'
            if net == 'generator':
                _, loss_value = sess.run([z_predictor_train_op, z_prediction_loss])
                assert not np.isnan(loss_value), 'Model diverged with NaN loss value'

            duration = time.time() - start_time

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = '{}: step {}, ({:.0f} examples/sec; {:.3f} sec/batch)'
                print(format_str.format(datetime.now(), step, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = FLAGS.train_dir
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        sess.close()

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    
    train()

if __name__ == '__main__':
    tf.app.run()
