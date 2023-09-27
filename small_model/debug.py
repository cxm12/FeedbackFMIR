from csbdeep.utils.tf import tf, IS_TF_1

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')

# tensor with shape [1, 301, 752, 752, 1]
input_tensor = tf.ones([1, 301, 752, 752, 1])
output_tensors = tf.split(input_tensor, num_or_size_splits=input_tensor.shape[1], axis=1)


if __name__ == '__main__':
    print(len(output_tensors))
    print(output_tensors[0].shape)
    b = tf.squeeze(output_tensors[0], axis=1)
    print(b.shape)
    
    patches = tf.image.extract_patches(input_tensor,
                                        sizes=[1, 1, 64, 64, 1],
                                        strides=[1, 1, 64, 64, 1],
                                        rates=[1, 1, 1, 1, 1],
                                        padding='VALID')
    
    print(IS_TF_1)
