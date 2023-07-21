#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
from csbdeep.utils import normalize
from func import *
import os
import random
from csbdeep.io import load_training_data
from tifffile import imread, imsave
from scipy.ndimage.interpolation import zoom
sys.path.append('../../')


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 32, "The size of label to produce [21]")
flags.DEFINE_integer("label_size", 128, "The size of label to produce [21]")
flags.DEFINE_integer("shuffle_num", 200, "buffer size")
flags.DEFINE_integer("prefetch_num", 100, "prefetch buffer")
flags.DEFINE_integer("map_parallel_num", 8, "map_parallel_num")
FLAGS = flags.FLAGS


def loadData():
    axes = 'SCYX'
    datapath = '../../DataSet/Isotropic/%s/train_data/data_label.npz' % testset
    (X, Y), (X_val, Y_val), axes = load_training_data(datapath, validation_split=0.1, axes=axes, verbose=True)
    print('X.shape, Y.shape, X_val.shape, Y_val.shape', X.shape, Y.shape, X_val.shape, Y_val.shape)
    return X, Y, X_val, Y_val


def draw_features(x, savename):
    img = x[0, :, :, 0]
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
    img = img.astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(savename, img)


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d", init_value=[], trainable=True,
           hasBias=True):
    with tf.variable_scope(name):
        if (not (init_value)):
            if IS_TF_1:
                w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], trainable=trainable,
                                    initializer=tf.contrib.layers.xavier_initializer())
            else:
                w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], trainable=trainable,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        else:
            vname = tf.get_variable_scope()
            w = tf.Variable(init_value[vname.name + '/w:0'], name='w', trainable=trainable)
        _weight_decay(w)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        
        if hasBias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        return conv


def _weight_decay(var, wd=0.0001):
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)


class DataLoader():
    def __init__(self, config, X, Y, X_val, Y_val, istrain=True):
        self.patch_size = config.label_size
        self.batch_size = config.batch_size
        self.depth = X.shape[1]
        self.shuffle_num = config.shuffle_num
        self.prefetch_num = config.prefetch_num
        self.map_parallel_num = config.map_parallel_num
        if istrain:
            self.imHR = Y
            self.imLR = X
        else:
            self.imHR = Y_val
            self.imLR = X_val
        if testset == 'Isotropic_Retina':
            self.c = 2
        else:
            self.c = 1

    def get_generatorRGB(self):
        for i in range(len(self.imHR)):
            imgor = self.imHR[i]  #
            imglr = self.imLR[i]
            gt = normalize(imgor, datamin, datamax, clip=True) * 255  # [0, 1]
            lr = normalize(imglr, datamin, datamax, clip=True) * 255
            
            if np.isnan(lr).any() or np.isinf(lr).any() or np.isnan(gt).any() or np.isinf(gt).any():
                continue
            yield gt, lr
    
    def read_pngsRGB(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorRGB, (tf.float32, tf.float32))
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        if IS_TF_1:
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
            c, l = dataset.make_one_shot_iterator().get_next()
        else:
            ds = dataset.apply(
                tf.data.experimental.shuffle_and_repeat(buffer_size=self.shuffle_num))
            ds = ds.batch(self.batch_size).prefetch(self.prefetch_num)
            
            c, l = ds.make_one_shot_iterator().get_next()
        
        p = self.patch_size
        l = tf.reshape(l, [self.batch_size, p, p, self.c])
        c = tf.reshape(c, [self.batch_size, p, p, self.c])
        return c, l


def ModelfbUncer_f(x, n_filter_base=16, n_channel_out=1, residual=True, step=3, name="SR"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        f = tf.nn.relu(conv2d(x, n_filter_base // 4, 3, 3, 1, 1, 'conv1'))
        fin = tf.nn.relu(conv2d(f, n_filter_base, 3, 3, 1, 1, 'convin'))
        
        out = []
        theta = []
        SR = []
        for i in range(step):
            if i == 0:
                f0 = tf.concat([fin, fin], -1)
            f1 = tf.nn.relu(conv2d(f0, n_filter_base, 3, 3, 1, 1, 'conv1_%d' % i))
            up1 = f1
            fup1 = tf.nn.relu(conv2d(up1, n_filter_base, 3, 3, 1, 1, name='conu1_%d' % i))
            dn1 = fup1
            fdn1 = tf.nn.relu(conv2d(dn1, n_filter_base, 3, 3, 1, 1, name='convd1_%d' % i))
            
            up2 = tf.concat([fdn1, f1], -1)
            fup2 = tf.nn.relu(conv2d(up2, n_filter_base, 3, 3, 1, 1, name='conu2_%d' % i))
            dn2 = tf.concat([fup2, fup1], -1)
            fdn2 = tf.nn.relu(conv2d(dn2, n_filter_base, 3, 3, 1, 1, name='cond2_%d' % i))
            
            fcat = tf.concat([fdn2, fdn1], -1)
            fcat = tf.nn.relu(conv2d(fcat, n_filter_base, 3, 3, 1, 1, name='con2_%d' % i))
            
            # # feature perturbation
            fd0 = fcat
            funcer = tf.nn.elu(conv2d(fd0, n_filter_base, 3, 3, 1, 1, name='con2un_%d' % i))
            funcer = tf.nn.elu(conv2d(funcer, n_filter_base, 3, 3, 1, 1, name='con3un_%d' % i))
            funcer = tf.nn.elu(conv2d(funcer, 1, 3, 3, 1, 1, name='con4un_%d' % i))
            
            norm = tf.random.truncated_normal(funcer.get_shape(), mean=0, stddev=1)
            fcat = tf.concat([fd0 + funcer * norm, fd0], -1)
            fd1 = tf.nn.relu(conv2d(fcat, n_filter_base, 3, 3, 1, 1, name='concat%d' % i))
            fout = fd1 + f1  # f0 = tf.concat([fin, fout], -1)
            out.append(fout)
            theta.append(funcer)
            
            # # outimg
            if i == step - 1:
                fd = tf.nn.relu(conv2d(tf.concat(out, -1), n_filter_base, 3, 3, 1, 1, name='concat'))
                final = conv2d(fd, n_channel_out, 1, 1, 1, 1, name='conout')
            else:
                final = conv2d(fout, n_channel_out, 1, 1, 1, 1, name='conout%d' % i)
            if residual:
                final = final + x
            SR.append(final)
        return SR, theta


def ModelfbUncer_f_twostage(x, theta, n_filter_base=16, n_channel_out=1, residual=True, step=3, name="twostage-SR"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        f = tf.nn.relu(conv2d(x, n_filter_base // 4, 3, 3, 1, 1, 'conv1'))
        fin = tf.nn.relu(conv2d(f, n_filter_base, 3, 3, 1, 1, 'convin'))
        
        out = []
        SR = []
        for i in range(step):
            if i == 0:
                f0 = tf.concat([fin, fin], -1)
            f1 = tf.nn.relu(conv2d(f0, n_filter_base, 3, 3, 1, 1, 'conv1_%d' % i))
            up1 = f1
            fup1 = tf.nn.relu(conv2d(up1, n_filter_base, 3, 3, 1, 1, name='conu1_%d' % i))
            dn1 = fup1
            fdn1 = tf.nn.relu(conv2d(dn1, n_filter_base, 3, 3, 1, 1, name='convd1_%d' % i))
            
            up2 = tf.concat([fdn1, f1], -1)
            fup2 = tf.nn.relu(conv2d(up2, n_filter_base, 3, 3, 1, 1, name='conu2_%d' % i))
            dn2 = tf.concat([fup2, fup1], -1)
            fdn2 = tf.nn.relu(conv2d(dn2, n_filter_base, 3, 3, 1, 1, name='cond2_%d' % i))
            
            fcat = tf.concat([fdn2, fdn1], -1)
            fcat = tf.nn.relu(conv2d(fcat, n_filter_base, 3, 3, 1, 1, name='con2_%d' % i))
            
            # # feature perturbation
            fd0 = fcat
            funcer = theta[i]
            
            norm = tf.random.truncated_normal(funcer.get_shape(), mean=0, stddev=1)
            fcat = tf.concat([fd0 + funcer * norm, fd0], -1)
            fd1 = tf.nn.relu(conv2d(fcat, n_filter_base, 3, 3, 1, 1, name='concat%d' % i))
            fout = fd1 + f1  # f0 = tf.concat([fin, fout], -1)
            out.append(fout)
            
            # # outimg
            if i == step - 1:
                fd = tf.nn.relu(conv2d(tf.concat(out, -1), n_filter_base, 3, 3, 1, 1, name='concat'))
                final = conv2d(fd, n_channel_out, 1, 1, 1, 1, name='conout')
            else:
                final = conv2d(fout, n_channel_out, 1, 1, 1, 1, name='conout%d' % i)
            if residual:
                final = final + x
            SR.append(final)
        return SR


def train(X, Y, X_val, Y_val):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    data_loader = DataLoader(FLAGS, X, Y, X_val, Y_val)
    c, l = data_loader.read_pngsRGB()
    
    lossuncer = 0
    l2loss = 0
    # model
    if modeltype == 'FBuncertainty_f':
        sr, theta = ModelfbUncer_f(l, n_filter_base=16, n_channel_out=outchannel, residual=True, step=3)
        for i in range(len(theta)):
            s = tf.exp(-theta[i])
            sr_ = tf.multiply(sr[i], s)
            hr_ = tf.multiply(c, s)
            lossuncer += tf.reduce_mean(tf.abs(sr_ - hr_)) + 2 * tf.reduce_mean(theta[i])
            l2loss = tf.zeros(lossuncer.get_shape())  #
    elif modeltype == 'FBuncertainty_f_twostage':
        _, theta = ModelfbUncer_f(l, n_filter_base=16, n_channel_out=outchannel, residual=True, step=3)
        sr = ModelfbUncer_f_twostage(l, theta, n_filter_base=16, n_channel_out=outchannel, residual=True, step=3)
        l2loss += tf.reduce_mean(tf.abs(sr - c))
        lossuncer = tf.zeros(l2loss.get_shape())
    
    global_step = tf.Variable(0, trainable=False)
    steprate = 1000
    decay = 0.95
    lrate = tf.train.exponential_decay(5e-5, global_step, steprate, decay)
    model_vars = tf.trainable_variables()
    if modeltype == 'FBuncertainty_f_twostage':
        model_vars = []
        model_varss1 = []
        for v in tf.trainable_variables():
            if 'twostage-SR' in v.name:
                model_vars.append(v)
            else:
                model_varss1.append(v)
        saverstage1 = tf.train.Saver(var_list=model_varss1, max_to_keep=10)
    
    trainer = tf.train.AdamOptimizer(lrate).minimize(l2loss + lossuncer, var_list=model_vars)  #
    saver = tf.train.Saver(var_list=model_vars, max_to_keep=1000)
    init = tf.global_variables_initializer()
    ckpt = tf.train.get_checkpoint_state(modelpath)
    with tf.Session(config=config) as sess:
        sess.run(init)
        start_it = 1
        if modeltype == 'FBuncertainty_f_twostage':
            ckpt1 = tf.train.get_checkpoint_state(modelpath1)
            saverstage1.restore(sess, ckpt1.model_checkpoint_path)
            print("Load Stage 1 model :" + ckpt1.model_checkpoint_path)
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_str = ckpt.model_checkpoint_path
            start_it = int(ckpt_str[ckpt_str.find('-') + 1:]) + 1
            print("Continue training, start iteration:" + str(start_it))
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        for itr in range(start_it, max_iterations):
            _, MSE, lossUn, learning = sess.run([trainer, l2loss, lossuncer, lrate],
                                                feed_dict={dropout_keep_prob1: 0.9})
            if itr % 200 == 0 or (itr + 1) == max_iterations:
                print(itr, ': MSE, lossUn, learning', MSE, lossUn, learning)  #
            
            assert not np.isnan(MSE).any(), 'Model diverged with loss = NaN'
            if itr % validite == 0 or (itr + 1) == max_iterations:
                checkpoint_path = os.path.join(modelpath, 'model.ckpt')
                print('step %d, lossuncer, MSEloss = %.5f, %.5f' % (itr, lossUn, MSE), learning)
                resultth = sess.run(theta[-1], feed_dict={dropout_keep_prob1: 1.0})
                print('Max/Min of theta', np.max(resultth), np.min(resultth))
                
        saver.save(sess, checkpoint_path, global_step=max_iterations)


def test(num=1):
    filenames = os.listdir(testpath)
    
    if testset == 'Isotropic_Retina':
        inp_lr = tf.placeholder(tf.float32, [1, 32, 32, 2])
    else:
        inp_lr = tf.placeholder(tf.float32, [1, 32, 32, 1])
        
    if modeltype == 'FBuncertainty_f':
        sr, theta = ModelfbUncer_f(inp_lr, n_filter_base=16, n_channel_out=outchannel, residual=True, step=3)
    elif modeltype == 'FBuncertainty_f_twostage':
        _, theta = ModelfbUncer_f(inp_lr, n_filter_base=16, n_channel_out=outchannel, residual=True, step=3)
        sr = ModelfbUncer_f_twostage(inp_lr, theta, n_filter_base=16, n_channel_out=outchannel, residual=True, step=3)

    model_vars = tf.trainable_variables()
    if modeltype == 'FBuncertainty_f_twostage':
        model_vars = []
        model_varss1 = []
        for v in tf.trainable_variables():
            if 'twostage-SR' in v.name:
                model_vars.append(v)
            else:
                model_varss1.append(v)
        saverstage1 = tf.train.Saver(var_list=model_varss1, max_to_keep=10)
    saver = tf.train.Saver(var_list=model_vars, max_to_keep=10000)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if modeltype == 'FBuncertainty_f_twostage':
            ckpt1 = tf.train.get_checkpoint_state(modelpath1)
            saverstage1.restore(sess, ckpt1.model_checkpoint_path)
            print("Load Stage 1 model :" + ckpt1.model_checkpoint_path)

        ckpt = tf.train.get_checkpoint_state(modelpath)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load model", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            ckpt_str = ckpt.model_checkpoint_path
            start_it = int(ckpt_str[ckpt_str.rfind('-') + 1:])
            savepath = modelpath + 'results/model%d/' % (start_it // 1000)
            os.makedirs(savepath, exist_ok=True)
        else:
            print('No Model Loaded !!')
        
        for fi in range(num):
            name = filenames[fi][:-4]
            rgblr = imread(testpath + name + '.tif')
            if testset == 'Isotropic_Retina':
                rgblr = zoom(rgblr, (subsample, 1, 1, 1), order=1)
                rgblr = np.transpose(rgblr, [0, 2, 3, 1])
            else:
                rgblr = np.expand_dims(rgblr, -1)
                rgblr = zoom(rgblr, (1, 1, 1, 1), order=1)
            rgblr = np.float32(normalize(rgblr, datamin, datamax, clip=True)) * rgbrange
            
            d0, h0, w0, c0 = rgblr.shape
            curlr = rgblr[-d:]

            if modeltype == 'FBuncertainty_f':
                out, theta = ModelfbUncer_f(tf.convert_to_tensor(curlr), n_filter_base=16, n_channel_out=outchannel,
                                            residual=True, step=3)
            elif modeltype == 'FBuncertainty_f_twostage':
                _, theta = ModelfbUncer_f(tf.convert_to_tensor(curlr), n_filter_base=16, n_channel_out=outchannel,
                                          residual=True, step=3)
                out = ModelfbUncer_f_twostage(tf.convert_to_tensor(curlr), theta, n_filter_base=16,
                                              n_channel_out=outchannel, residual=True, step=3)

            result1 = sess.run(out[-1], feed_dict={dropout_keep_prob1: 1.0})
            sr = np.reshape(np.float32(np.maximum(0, np.minimum(rgbrange, result1)) * (255 / rgbrange)), [d, h, w, c0])
            if testset == 'Isotropic_Retina':
                imsave(savepath + name + '-SR255_ZYX.tif', np.transpose(sr, [0, 3, 1, 2]))
            else:
                imsave(savepath + name + '-SR255_ZYX.tif', np.squeeze(sr))
            

if __name__ == '__main__':
    datamin, datamax = 0, 100
    testsetlst = ['Isotropic_Liver']  # ['Isotropic_Retina']  # ['Isotropic_Drosophila']  #
    validite = 2000
    modeltypelst = ['FBuncertainty_f_twostage']  # ['FBuncertainty_f']  #
    istrain = False  # True  #
    max_iterations = 600001
    ntile = (4, 4, 4)  #
    rgbrange = 255
    saveVar = True
    save = True
    my_seed = 34573529
    dropout_keep_prob1 = tf.placeholder(tf.float32)

    for testset in testsetlst:
        testpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/Isotropic/' + testset + '/test_data/'
        if testset == 'Isotropic_Drosophila' or testset == 'Isotropic_Liver':
            outchannel = 1
            subsample = 1
        elif testset == 'Isotropic_Retina':
            subsample = 10.2
            outchannel = 2
            
        maxpsnr = 0
        maxseed = 0
        
        for modeltype in modeltypelst:
            modelpath = './models/epoch200/%s/%s/' % (modeltype, testset)
            os.makedirs(modelpath, exist_ok=True)
            modelpath1 = './models/epoch200/FBuncertainty_f/%s/' % (testset)
            savepath = modelpath + '/results/'
            os.makedirs(savepath, exist_ok=True)
            
            if istrain:
                np.random.seed(my_seed)
                random.seed(my_seed)
                print('******************************************************************')
                print('*** Train on %s, Model %s ***' % (testset, modeltype))
                print('******************************************************************')
                X, Y, X_val, Y_val = loadData()
                model = train(X, Y, X_val, Y_val)
                
            else:
                test()
