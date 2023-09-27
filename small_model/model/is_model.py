from model.func_md import *


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


# inp_lr = tf.placeholder(tf.float32, [1, 32, 32, 1])
# _, theta = ModelfbUncer_f(inp_lr, n_filter_base=16, n_channel_out=1, residual=True, step=3)
# sr = ModelfbUncer_f_twostage(inp_lr, theta, n_filter_base=16, n_channel_out=1, residual=True, step=3)

IS_FB_uncer_vars = []
IS_FB_twostage_vars = []

# for var in tf.trainable_variables():
#     if 'twostage-SR' in var.name:
#         IS_FB_twostage_vars.append(var)
#     else:
#         IS_FB_uncer_vars.append(var)
