from model.func_md import *
from csbdeep.utils import normalize
from div2k import PercentileNormalizer, normalize
from scipy.ndimage.interpolation import zoom
import numpy as np
import utility

from model.sr_model import ModelfbUncer_f as SR_ModelfbUncer
from model.sr_model import ModelfbUncer_f_twostage as SR_ModelfbUncer_twostage

from model.is_model import ModelfbUncer_f as IS_ModelfbUncer
from model.is_model import ModelfbUncer_f_twostage as IS_ModelfbUncer_twostage

from model.de_model import ModelfbUncer_f as DE_ModelfbUncer
from model.de_model import ModelfbUncer_f_twostage as DE_ModelfbUncer_twostage


def _rotate(arr, k=1, axis=1, copy=True):
    """Rotate by 90 degrees around the first 2 axes."""
    if copy:
        arr = arr.copy()
    k = k % 4
    arr = np.rollaxis(arr, axis, arr.ndim)
    if k == 0:
        res = arr
    elif k == 1:
        res = arr[::-1].swapaxes(0, 1)
    elif k == 2:
        res = arr[::-1, ::-1]
    else:
        res = arr.swapaxes(0, 1)[::-1]

    res = np.rollaxis(res, -1, axis)
    return res


class FmirModel:
    def __init__(self, image, task_type, model_type):

        tf.reset_default_graph()

        self.image = image
        self.task_type = task_type
        self.model_type = model_type

        self.datamin = 0
        self.datamax = 100
        self.rgbrange = 255
        self.dropout_keep_prob1 = tf.placeholder(tf.float32)
        self.normalizer = PercentileNormalizer(2, 99.8)

        self.cur_lr = None
        self.network = None
        self.networkstage1 = None
        self.model_vars = []
        self.model_varss1 = []
        self.model_args = []
        self.modelpath = './experiment/%s/%s/' % (task_type, model_type)
        self.modelpath1 = './experiment/%s/FBuncertainty_f_uncer/' % (task_type)
        self.saver = None
        self.saverstage1 = None

        self.d = 0
        self.h = 0
        self.w = 0
        self.c0 = 0

        self.dn = 0
        self.hn = 0
        self.wn = 0

        self.curlr = []
        self.curlr90 = []
        self.result3 = []
        self.sr3 = None
        self.resultth3 = None


    def _pre_build(self, image, task_type, model_type):
        with tf.GradientTape() as tape:
            with tape.stop_recording():
                if 'SR' in task_type:
                    # data preset
                    scale = 2
                    inp_lr = tf.placeholder(tf.float32, [1, 32, 32, 1])
                    rgblr = np.array(image)
                    cur = rgblr[:, :]
                    h, w = cur.shape
                    self.cur_lr = np.float32(normalize(rgblr, self.datamin, self.datamax, clip=True)) * self.rgbrange
                    self.cur_lr = np.reshape(self.cur_lr, [1, h, w, 1])

                    # model preset
                    self.model_args = [32, 1, True, 3, scale]
                    if model_type == 'FBuncertainty_f_uncer':
                        self.network = SR_ModelfbUncer
                        sr, theta = self.network(inp_lr, *self.model_args)
                        self.model_vars = tf.trainable_variables()
                    elif model_type == 'FBuncertainty_f_twostage':
                        self.networkstage1 = SR_ModelfbUncer
                        self.network = SR_ModelfbUncer_twostage
                        _, theta = self.networkstage1(inp_lr, *self.model_args)
                        sr = self.network(inp_lr, theta, *self.model_args)
                        for v in tf.trainable_variables():
                            if 'twostage-SR' in v.name:
                                self.model_vars.append(v)
                            else:
                                self.model_varss1.append(v)
                        self.saverstage1 = tf.train.Saver(var_list=self.model_varss1, max_to_keep=10)
                    else:
                        raise ValueError('model_type not supported')
                    self.saver = tf.train.Saver(var_list=self.model_vars, max_to_keep=10000)

                elif 'Isotropic' in task_type:
                    # data preset
                    outchannel = 1
                    inp_lr = tf.placeholder(tf.float32, [1, 32, 32, 1])
                    rgblr = image
                    rgblr = np.expand_dims(rgblr, -1)
                    rgblr = zoom(rgblr, (1, 1, 1, 1), order=1)
                    self.wn = 4
                    self.hn = 8
                    rgblr = np.float32(normalize(rgblr, self.datamin, self.datamax, clip=True)) * self.rgbrange

                    self.d, h0, w0, self.c0 = rgblr.shape
                    self.w = np.ceil(w0 / self.wn).astype(int)
                    self.h = np.ceil(h0 / self.hn).astype(int)
                    rgblr = rgblr[:, :h0//self.hn * self.hn, :w0//self.wn * self.wn, :]
                    self.d, h0, w0, self.c0 = rgblr.shape

                    self.sr3 = np.zeros([self.d, h0, w0, self.c0])
                    self.resultth3 = np.zeros([self.d, h0, w0, 1])

                    for hi in range(self.hn):
                        for wi in range(self.wn):
                            x_rot1 = _rotate(rgblr[:, hi * self.h:(hi + 1) * self.h, wi * self.w:(wi + 1) * self.w, :], axis=1, copy=False)  # [360,768,768,2] -> [768,768,360,2]
                            self.curlr.append(x_rot1)  # [w, h, d, 1]
                            x_rot2 = _rotate(_rotate(rgblr[:, hi * self.h:(hi + 1) * self.h, wi * self.w:(wi + 1) * self.w, :], axis=2, copy=False), axis=0, copy=False)  # [768,768,360,2]
                            self.curlr90.append(x_rot2)  # [h, w, d, 1]

                    # model preset
                    self.model_args = [16, outchannel, True, 3]
                    if model_type == 'FBuncertainty_f_uncer':
                        self.network = IS_ModelfbUncer
                        sr, theta = self.network(inp_lr, *self.model_args)
                        self.model_vars = tf.trainable_variables()
                    elif model_type == 'FBuncertainty_f_twostage':
                        self.networkstage1 = IS_ModelfbUncer
                        self.network = IS_ModelfbUncer_twostage
                        _, theta = self.networkstage1(inp_lr, *self.model_args)
                        sr = self.network(inp_lr, theta, *self.model_args)
                        for v in tf.trainable_variables():
                            if 'twostage-SR' in v.name:
                                self.model_vars.append(v)
                            else:
                                self.model_varss1.append(v)
                        self.saverstage1 = tf.train.Saver(var_list=self.model_varss1, max_to_keep=10)
                    else:
                        raise ValueError('model_type not supported')
                    self.saver = tf.train.Saver(var_list=self.model_vars, max_to_keep=10000)

                elif 'Denoising' in task_type:
                    # data preset
                    inp_lr = tf.placeholder(tf.float32, [1, 16, 32, 32, 1])

                    self.dn = 2
                    self.hn = 4
                    self.wn = 4
 
                    rgblr = image.astype(np.float32)
                    d0, h0, w0 = rgblr.shape
                    self.d = np.ceil(d0 / self.dn).astype(int)
                    self.h = np.ceil(h0 / self.hn).astype(int)
                    self.w = np.ceil(w0 / self.wn).astype(int)

                    self.sr3 = np.zeros([d0, h0, w0])

                    for di in range(self.dn):
                        if di == self.dn - 1:
                            for hi in range(self.hn):
                                for wi in range(self.wn):
                                    self.curlr.append(rgblr[-self.d:, hi * self.h:(hi + 1) * self.h, wi * self.w:(wi + 1) * self.w])
                        else:
                            for hi in range(self.hn):
                                for wi in range(self.wn):
                                    self.curlr.append(rgblr[di * self.d:(di + 1) * self.d, hi * self.h:(hi + 1) * self.h, wi * self.w:(wi + 1) * self.w])

                    # model preset
                    self.model_args = [16, 1, True, 3]
                    if model_type == 'FBuncertainty_f_uncer':
                        self.network = DE_ModelfbUncer
                        sr, theta = self.network(inp_lr, *self.model_args)
                        self.model_vars = tf.trainable_variables()
                    elif model_type == 'FBuncertainty_f_twostage':
                        self.networkstage1 = DE_ModelfbUncer
                        self.network = DE_ModelfbUncer_twostage
                        _, theta = self.networkstage1(inp_lr, *self.model_args)
                        sr = self.network(inp_lr, theta, *self.model_args)
                        for v in tf.trainable_variables():
                            if 'twostage-SR' in v.name:
                                self.model_vars.append(v)
                            else:
                                self.model_varss1.append(v)
                        self.saverstage1 = tf.train.Saver(var_list=self.model_varss1, max_to_keep=10)
                    else:
                        raise ValueError('model_type not supported')
                    self.saver = tf.train.Saver(var_list=self.model_vars, max_to_keep=10000)

                else:
                    raise ValueError('task_type not supported')


    def run_model(self):
        self._pre_build(self.image, self.task_type, self.model_type)

        with tf.GradientTape() as tape:
            with tape.stop_recording():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    if self.networkstage1 is not None:
                        ckpt1 = tf.train.get_checkpoint_state(self.modelpath1)
                        self.saverstage1.restore(sess, ckpt1.model_checkpoint_path)
                    ckpt = tf.train.get_checkpoint_state(self.modelpath)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)

                    # run model
                    if 'SR' in self.task_type:
                        if self.networkstage1 is None:
                            out, theta = self.network(tf.convert_to_tensor(self.cur_lr), *self.model_args)
                        else:
                            _, theta = self.networkstage1(tf.convert_to_tensor(self.cur_lr), *self.model_args)
                            out = self.network(tf.convert_to_tensor(self.cur_lr), theta, *self.model_args)
                        result = sess.run(out[-1], feed_dict={self.dropout_keep_prob1: 1.0})

                    elif 'Isotropic' in self.task_type:
                        num = 0
                        for cur_lr1 in self.curlr:
                            if self.networkstage1 is None:
                                out, theta = self.network(tf.convert_to_tensor(cur_lr1), *self.model_args)
                                out90, theta90 = self.network(tf.convert_to_tensor(self.curlr90[num]), *self.model_args)
                            else:
                                _, theta = self.networkstage1(tf.convert_to_tensor(cur_lr1), *self.model_args)
                                _, theta90 = self.networkstage1(tf.convert_to_tensor(self.curlr90[num]), *self.model_args)
                                out = self.network(tf.convert_to_tensor(cur_lr1), theta, *self.model_args)
                                out90 = self.network(tf.convert_to_tensor(self.curlr90[num]), theta90, *self.model_args)
                            resultlst = sess.run(out, feed_dict={self.dropout_keep_prob1: 1.0})  # [w, h, d, 1]

                            num += 1

                            resultlst90 = sess.run(out90, feed_dict={self.dropout_keep_prob1: 1.0})  # [h, w, d, 1]

                            u1 = _rotate(resultlst[2], -1, axis=1, copy=False)
                            u2 = _rotate(_rotate(resultlst90[2], -1, axis=0, copy=False), -1, axis=2, copy=False)  # [360,768,768,2]
                            r2 = np.sqrt(np.maximum(u1, 0) * np.maximum(u2, 0))
                            srp3 = np.reshape(np.float32(np.maximum(0, np.minimum(self.rgbrange, r2)) * (255 / self.rgbrange)), [self.d, self.h, self.w, self.c0])
                            self.result3.append(srp3)
                        
                        i = 0
                        for hi in range(self.hn):
                            for wi in range(self.wn):
                                self.sr3[:, hi * self.h:(hi + 1) * self.h, wi * self.w:(wi + 1) * self.w, :] = self.result3[i]
                                i += 1
                    
                    elif 'Denoising' in self.task_type:
                        for cur_lr1 in self.curlr:
                            cur_lr1 = np.reshape(cur_lr1, [1, self.d, self.h, self.w, 1])
                            if self.networkstage1 is None:
                                out, theta = self.network(tf.convert_to_tensor(cur_lr1), *self.model_args)
                            else:
                                _, theta = self.networkstage1(tf.convert_to_tensor(cur_lr1), *self.model_args)
                                out = self.network(tf.convert_to_tensor(cur_lr1), theta, *self.model_args)
                            resultlst = sess.run(out, feed_dict={self.dropout_keep_prob1: 1.0})  # [w, h, d, 1]
                            self.result3.append(np.reshape(resultlst[2], [self.d, self.h, self.w]))
                        
                        i = 0
                        for di in range(self.dn):
                            if di == self.dn - 1:
                                for hi in range(self.hn):
                                    for wi in range(self.wn):
                                        self.sr3[-self.d:, hi * self.h:(hi + 1) * self.h, wi * self.w:(wi + 1) * self.w] = self.result3[i]
                                        i += 1
                            else:
                                for hi in range(self.hn):
                                    for wi in range(self.wn):
                                        self.sr3[di * self.d:(di + 1) * self.d, hi * self.h:(hi + 1) * self.h, wi * self.w:(wi + 1) * self.w] = self.result3[i]
                                        i += 1
                    
                    else:
                        raise ValueError('task_type not supported')

                    sess.close()
        
        tf.reset_default_graph()
        
        if 'SR' in self.task_type:
            sr = np.round(np.maximum(0, np.minimum(self.rgbrange, result)) * (255/self.rgbrange))
            _, c1, c2, c3 = sr.shape
            sr = np.reshape(sr, [c1, c2, c3])
            sr_norm = utility.savecolorim(None, sr, norm=True)

            return sr, [sr_norm]
        
        elif 'Isotropic' in self.task_type:
            sr = np.transpose(self.sr3, [0, 3, 1, 2])
            sr_norm = np.squeeze(np.float32(normalize(sr, self.datamin, self.datamax, clip=True)))
            clips = []
            for i in range(sr_norm.shape[0]):
                clips.append(utility.savecolorim(None, sr_norm[i], norm=True))
            
            return sr, clips
        
        elif 'Denoising' in self.task_type:
            sr = np.float32(np.maximum(0, np.minimum(self.rgbrange, self.sr3)) * (255 / self.rgbrange))
            sr = np.float32(normalize(sr, self.datamin, self.datamax, clip=True)) * self.rgbrange
            sr_norm = np.squeeze(np.float32(normalize(sr, self.datamin, self.datamax, clip=True)))
            clips = []
            for i in range(sr_norm.shape[0]):
                clips.append(utility.savecolorim(None, sr_norm[i], norm=True))
            
            return sr, clips

        else:
            raise ValueError('task_type not supported')
