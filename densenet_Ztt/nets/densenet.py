"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)#参数初始化

#H变换 batch-relu-conv-drop
def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current

#dense block 每一层先1*1 再3*3 tmp 表示这一层的输出，net对前面所有层进行累加（拼接）
def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net

 
    

def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24 #这个应该就是K把，表示每个dense block中每层输出的feature map个数
    compression_rate = 0.5 #这个应该表示的就是论文中的reducation吧，表示把输出缩减到输入的多少倍

    #根据compression_rate 返回输出个数
    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}#这个应该是储存节点的意思吧？
    
    #Transition Layer
    def transition(net,scope='transition'):
        num=reduce_dim(net)
        
        current = slim.batch_norm(net, scope=scope + '_bn')#我看论文里面好像卷积之前要先做一次batch normlizition
        current = slim.conv2d(current,num, [1, 1],scope=scope+'_conv')
        current = slim.avg_pool2d(current,[2,2],stride=2,scope=scope+'_pool')
        return current
            
    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            
            conv_1=slim.conv2d(images,2*growth, [7, 7],stride=2,scope=scope + '_conv7x7_1')
            end_points['conv_1']=conv_1
            
            pool_1=slim.max_pool2d(conv_1, [3,3], stride=2, padding='VALID')
            end_points['pool_1']=pool_1
            
            dens_block_1=block(pool_1, 6, growth,scope='block_1')
            trans_layer_1=transition(dens_block_1,scope='transition_1')
            end_points['trans_layer_1']=trans_layer_1
            
            dens_block_2=block(trans_layer_1, 12, growth,scope='block_2')
            trans_layer_2=transition(dens_block_2,scope='transition_2')
            end_points['trans_layer_2']=trans_layer_2
            
            dens_block_3=block(trans_layer_2, 24, growth,scope='block_3')
            trans_layer_3=transition(dens_block_3,scope='transition_3')
            end_points['trans_layer_3']=trans_layer_3
            
            dens_block_4=block(trans_layer_3, 16, growth,scope='block_4')
            end_points['dens_block_4']=dens_block_4
            
            n=dens_block_4.shape[1]
            pool_global=slim.avg_pool2d(dens_block_4, [n,n],padding='VALID',scope=scope+'pool_global')
            end_points['pool_global']=pool_global
            
            logits=slim.FullyConnected(pool_global,num_classes,scope=scope+'fc')
            end_points['logits']=logits
            
            
            ##########################
            # Put your code here.
            ##########################

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
