import numpy as np
import tensorflow as tf
from .tensorlayer import *
from .tensorlayer.layers import *
from os.path import join, exists, split, isfile
from os import makedirs, environ
from shutil import rmtree
from glob import glob
from scipy.misc import imread, imresize, imsave, imrotate
import cv2
from tensorflow.python.framework import graph_util
from sklearn.metrics import roc_curve, auc

# set logging level for TensorFlow
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


class DLNLF(object):
    def __init__(
            self,
            save_dir=None,
            num_res_blocks=4,
            one_sample_num=125
    ):
        self.save_dir = save_dir
        self.num_res_blocks = int(num_res_blocks)
        self.is_model_built = False,
        self.one_sample_num = one_sample_num


    def model(
            self,
            inputs,     # LR images, in range of [-1, 1]
            is_train=True,
            reuse=False
    ):
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("classification", reuse=reuse):
            layers.set_name_reuse(reuse)

            map_in = InputLayer(inputs=inputs, name='inputs')
            net_1 = Conv3dLayer(layer=map_in, shape=[3, 3, 3, 1, 32], strides=[1, 1, 1, 1, 1],
                                padding='SAME', W_init=w_init, name='small/conv1_1', act=tf.nn.relu, b_init=None)
            net_1 = NonlocalLayer(layer=net_1, depth=16, name='nonlocal1_1')
            # net_1 = DenoiseLayer(layer=net_1, depth=16, name='denoise1_1')
            net_1 = MaxPool3d(net_1, filter_size=2, strides=2, padding='SAME', name='maxpool1_1')

            net_1 = Conv3dLayer(layer=net_1, shape=[3, 3, 3, 32, 64], strides=[1, 1, 1, 1, 1],
                               padding='SAME', W_init=w_init, name='small/conv2_1', act=tf.nn.relu, b_init=None)
            net_1 = NonlocalLayer(layer=net_1, depth=32, name='nonlocal2_1')
            # net_1 = DenoiseLayer(layer=net_1, depth=32, name='denoise2_1')
            net_1 = MaxPool3d(net_1, filter_size=2, strides=2, padding='SAME', name='maxpool2_1')

            net_1 = Conv3dLayer(layer=net_1, shape=[3, 3, 3, 64, 128], strides=[1, 1, 1, 1, 1],
                               padding='SAME', W_init=w_init, name='small/conv3_1', act=tf.nn.relu, b_init=None)
            net_1 = NonlocalLayer(layer=net_1, depth=64, name='nonlocal3_1')
            # net_1 = DenoiseLayer(layer=net_1, depth=64, name='denoise3_1')
            net_1 = MaxPool3d(net_1, filter_size=2, strides=2, padding='SAME', name='maxpool3_1')


            net_2 = Conv3dLayer(layer=map_in, shape=[3, 3, 3, 1, 32], strides=[1, 1, 1, 1, 1],
                                padding='SAME', W_init=w_init, name='small/conv1_2', act=tf.nn.relu, b_init=None)
            # net_2 = NonlocalLayer(layer=net_2, depth=16, name='nonlocal1_2')
            net_2 = DenoiseLayer(layer=net_2, depth=16, name='denoise1_2')
            net_2 = MaxPool3d(net_2, filter_size=2, strides=2, padding='SAME', name='maxpool1_2')

            net_2 = Conv3dLayer(layer=net_2, shape=[3, 3, 3, 32, 64], strides=[1, 1, 1, 1, 1],
                                padding='SAME', W_init=w_init, name='small/conv2_2', act=tf.nn.relu, b_init=None)
            # net_2 = NonlocalLayer(layer=net_2, depth=32, name='nonlocal2_2')
            net_2 = DenoiseLayer(layer=net_2, depth=32, name='denoise2_2')
            net_2 = MaxPool3d(net_2, filter_size=2, strides=2, padding='SAME', name='maxpool2_2')

            net_2 = Conv3dLayer(layer=net_2, shape=[3, 3, 3, 64, 128], strides=[1, 1, 1, 1, 1],
                                padding='SAME', W_init=w_init, name='small/conv3_2', act=tf.nn.relu, b_init=None)
            # net_2 = NonlocalLayer(layer=net_2, depth=64, name='nonlocal3_2')
            net_2 = DenoiseLayer(layer=net_2, depth=64, name='denoise3_2')
            net_2 = MaxPool3d(net_2, filter_size=2, strides=2, padding='SAME', name='maxpool3_2')

            net = BilinearLayer([net_1, net_2], name='bilinaer')
            # net = FlattenLayer(net_2, name='flatten_layer')
            # net = DenseLayer(net, n_units=256, act=tf.nn.relu, name='dense1')
            net_outputs = DenseLayer(net, n_units=2, name='dense3')

            return net_outputs.outputs

    def eta(self, time_per_iter, n_iter_remain, current_eta=None, alpha=.8):
        eta_ = time_per_iter * n_iter_remain
        if current_eta is not None:
            eta_ = (current_eta - time_per_iter) * alpha + eta_ * (1 - alpha)
        new_eta = eta_

        days = eta_ // (3600 * 24)
        eta_ -= days * (3600 * 24)

        hours = eta_ // 3600
        eta_ -= hours * 3600

        minutes = eta_ // 60
        eta_ -= minutes * 60

        seconds = eta_

        if days > 0:
            if days > 1:
                time_str = '%2d days %2d hr' % (days, hours)
            else:
                time_str = '%2d day %2d hr' % (days, hours)
        elif hours > 0 or minutes > 0:
            time_str = '%02d:%02d' % (hours, minutes)
        else:
            time_str = '%02d sec' % seconds

        return time_str, new_eta

    def get_file_name(self, dir, type):
        one_sample_num = self.one_sample_num
        high_test_num = 11 * one_sample_num  # 11 17
        low_test_num = 9 * one_sample_num  # 9 13
        files_input = sorted(glob(join(dir, '*' + type)))
        files_input.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), int(x.split('_')[-1].split('.')[0])))
        high_files = [hf for hf in files_input if int(hf.split('/')[-1].split('_')[0]) == 1]
        low_files = [lf for lf in files_input if int(lf.split('/')[-1].split('_')[0]) == 0]
        high_num, low_num = len(high_files) - high_test_num, len(low_files) - low_test_num
        train_files = np.concatenate((high_files[:high_num], low_files[:low_num]))
        train_labels = np.array([np.eye(2)[lb] for lb in np.concatenate(
            (np.ones(shape=high_num), np.zeros(shape=low_num))).astype(np.uint8)]).astype(np.float32)

        test_files = np.concatenate((high_files[high_num:], low_files[low_num:]))
        test_labels = np.array([np.eye(2)[lb] for lb in np.concatenate(
            (np.ones(shape=high_test_num), np.zeros(shape=low_test_num))).astype(np.uint8)]).astype(np.float32)

        return train_files, test_files, train_labels, test_labels

    def train(
            self,
            input_dir='data/train/input',  # original images
            batch_size=64,
            num_epochs=20,
            learning_rate=1e-4,
            beta1=0.9,
            num_train=5
    ):
        if input_dir == '':
            print('input dir is empty')
            exit(0)
        if self.save_dir == '':
            print('save dir is empty')
            exit(0)

        train_f, test_f, train_lb, test_lb = self.get_file_name(input_dir, '.npz')

        idxt = np.arange(int(self.one_sample_num / 2), test_f.shape[0], int(self.one_sample_num / 1))
        # idxt = np.arange(0, test_f.shape[0], int(self.one_sample_num / 5))
        test_img = [(np.load(im)['img']).reshape((16, 16, 16, 1)) for im in test_f[idxt]]
        test_lb = test_lb[idxt]
        print('test file num ', len(test_img))
        num_files = len(train_f)

        idxt = np.arange(0, test_f.shape[0], self.one_sample_num)
        test_name = [fn.split('/')[-1].split('_')[1] for fn in test_f[idxt]]
        print(test_name)

        # ********************************************************************************
        # *** build graph
        # ********************************************************************************
        with tf.Graph().as_default() as graph:
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, 16, 16, 16, 1])
            self.label = tf.placeholder(dtype=tf.float32, shape=[None, 2])
            self.outputs = self.model(self.input, is_train=True, reuse=False)
            self.test_outputs = self.model(self.input, is_train=False, reuse=True)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.label))

            output_position = tf.argmax(self.outputs, 1)
            label_position = tf.argmax(self.label, 1)
            predict = tf.equal(output_position, label_position)
            Accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

            test_prediction = tf.nn.softmax(self.test_outputs)
            test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.test_outputs, labels=self.label))
            test_output_position = tf.argmax(self.test_outputs, 1)
            test_label_position = tf.argmax(self.label, 1)
            test_predict = tf.equal(test_output_position, test_label_position)
            test_Accuracy = tf.reduce_mean(tf.cast(test_predict, tf.float32))

            # ********************************************************************************
            # *** optimizers
            # ********************************************************************************
            # trainable variables
            trainable_vars = tf.trainable_variables()
            var = [v for v in trainable_vars if 'classification' in v.name]

            # learning rate decay
            global_step = tf.Variable(0, trainable=False, name='global_step')
            num_batches = int(num_files / batch_size)
            print('num_batches ', num_batches)
            decayed_learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=100, #  max(num_epochs * num_batches / 2, 1),
                decay_rate=.98,
                staircase=True
            )

            # optimizer
            # optimizer = tf.train.GradientDescentOptimizer(
            #     learning_rate=decayed_learning_rate).minimize(loss, var_list=var, global_step=global_step)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=decayed_learning_rate, beta1=beta1).minimize(loss, var_list=var, global_step=global_step)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            loss_all, acc_all = [], []
            iter_time = []
            fpr, tpr, AUC = [], [], []
            for i in range(num_train):
                # print('stats before freezing')
                # stats_graph(graph)
                with tf.Session(config=config) as sess:
                    tf.global_variables_initializer().run()
                    idx = np.arange(num_files)
                    # train with all losses
                    loss_, acc_ = [], []
                    current_eta = None
                    for epoch in xrange(num_epochs):
                        # np.random.seed(epoch)
                        np.random.shuffle(idx)
                        for n_batch in xrange(num_batches):
                            step_time = time.time()
                            sub_idx = idx[n_batch * batch_size:n_batch * batch_size + batch_size]
                            batch_input = [(np.load(train_f[i])['img']).reshape((16, 16, 16, 1)) for i in sub_idx]
                            batch_labels = train_lb[sub_idx]

                            _, l, acc = sess.run(
                                fetches=[optimizer, loss, Accuracy],
                                feed_dict={self.input: batch_input, self.label: batch_labels}
                            )

                            # print
                            time_per_iter = time.time() - step_time
                            n_iter_remain = (num_epochs - epoch - 1) * num_batches + num_batches - n_batch
                            eta_str, eta_ = self.eta(time_per_iter, n_iter_remain, current_eta)
                            current_eta = eta_

                            iter_time.append(time_per_iter)

                            if n_batch % 10 == 0:
                                print('%02d Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\n'
                                      '\ttrain:\tloss = %.4f\tacc  = %.4f' %
                                      (i, epoch + 1, num_epochs, n_batch + 1, num_batches, eta_str, l, acc))

                                l, acc = sess.run(
                                    fetches=[test_loss, test_Accuracy],
                                    feed_dict={self.input: test_img, self.label: test_lb}
                                )

                                loss_.append(l), acc_.append(acc)
                    loss_all.append(loss_), acc_all.append(acc_)

                    lb_p, pre = sess.run(
                        fetches=[test_label_position, test_prediction],
                        feed_dict={self.input: test_img, self.label: test_lb}
                    )
                    fpr_, tpr_, thresholds = roc_curve(lb_p, pre[:, 1], drop_intermediate=False)
                    AUC_ = auc(fpr_, tpr_)
                    fpr.append(fpr_), tpr.append(tpr_), AUC.append(AUC_)

                    np.savez(self.save_dir, loss=loss_all, acc=acc_all, fpr=fpr, tpr=tpr, AUC=AUC)

                    # print([tensor.name for tensor in graph.as_graph_def().node])
                    # output_graph = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [tensor.name for tensor in graph.as_graph_def().node])
                    # with tf.gfile.GFile('graph.pb', "wb") as f:
                    #     f.write(output_graph.SerializeToString())

                    sess.close()

            # print('%.2f±%.2f' % (float(np.mean(iter_time[10:]))*1000, float(np.std(iter_time[10:], ddof=1))*1000))

        # with tf.Graph().as_default() as graph:
        #     graph = load_pb('./graph.pb')
        #     print('stats after freezing')
        #     stats_graph(graph)

