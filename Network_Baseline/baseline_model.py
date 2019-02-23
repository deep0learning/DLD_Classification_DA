import sys

sys.path.append('../Data_Initialization/')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import Initialization as init
import baseline_utils
import time


class baseline_model(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, reload_path, epoch, num_class, learning_rate,
                 batch_size,
                 img_height, img_width, train_phase):

        self.sess = sess
        self.source_training_data = train_data[0]
        self.target_training_data = train_data[1]
        self.validation_data = val_data
        self.source_test_data = tst_data[0]
        self.target_test_data = tst_data[1]
        self.reload_path = reload_path
        self.eps = epoch
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.lr = learning_rate
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class
        self.train_phase = train_phase
        self.plt_epoch = []
        self.plt_training_accuracy = []
        self.plt_validation_accuracy = []
        self.plt_training_loss = []
        self.plt_validation_loss = []

        self.build_model()
        if self.train_phase == 'Train':
            self.saveConfiguration()

    def saveConfiguration(self):
        baseline_utils.save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        baseline_utils.save2file('model : %s' % self.model, self.ckptDir, self.model)
        baseline_utils.save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        baseline_utils.save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        baseline_utils.save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        baseline_utils.save2file('image width : %d' % self.img_w, self.ckptDir, self.model)
        baseline_utils.save2file('num class : %d' % self.num_class, self.ckptDir, self.model)
        baseline_utils.save2file('train phase : %s' % self.train_phase, self.ckptDir, self.model)

    def convLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_weight = tf.get_variable('conv_weight',
                                          [ksize, ksize, inputMap.get_shape()[-1], out_channel],
                                          initializer=layers.variance_scaling_initializer())

            conv_result = tf.nn.conv2d(inputMap, conv_weight, strides=[1, stride, stride, 1], padding=padding)

            tf.summary.histogram('conv_weight', conv_weight)
            tf.summary.histogram('conv_result', conv_result)

        return conv_result

    def bnLayer(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training)

    def reluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def avgPoolLayer(self, inputMap, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            return tf.nn.avg_pool(inputMap, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    def globalPoolLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            size = inputMap.get_shape()[1]
            return self.avgPoolLayer(inputMap, size, size, padding='VALID', scope_name=scope_name)

    def flattenLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.layers.flatten(inputMap)

    def fcLayer(self, inputMap, out_channel, scope_name):
        with tf.variable_scope(scope_name):
            in_channel = inputMap.get_shape()[-1]
            fc_weight = tf.get_variable('fc_weight', [in_channel, out_channel],
                                        initializer=layers.variance_scaling_initializer())
            fc_bias = tf.get_variable('fc_bias', [out_channel], initializer=tf.zeros_initializer())

            fc_result = tf.matmul(inputMap, fc_weight) + fc_bias

            tf.summary.histogram('fc_weight', fc_weight)
            tf.summary.histogram('fc_bias', fc_bias)
            tf.summary.histogram('fc_result', fc_result)

        return fc_result

    def residualUnitLayer(self, inputMap, out_channel, ksize, unit_name, down_sampling, is_training, first_conv=False):
        with tf.variable_scope(unit_name):
            in_channel = inputMap.get_shape().as_list()[-1]
            if down_sampling:
                stride = 2
                increase_dim = True
            else:
                stride = 1
                increase_dim = False

            if first_conv:
                conv_layer1 = self.convLayer(inputMap, out_channel, ksize, stride, scope_name='conv_layer1')
            else:
                bn_layer1 = self.bnLayer(inputMap, scope_name='bn_layer1', is_training=is_training)
                relu_layer1 = self.reluLayer(bn_layer1, scope_name='relu_layer1')
                conv_layer1 = self.convLayer(relu_layer1, out_channel, ksize, stride, scope_name='conv_layer1')

            bn_layer2 = self.bnLayer(conv_layer1, scope_name='bn_layer2', is_training=is_training)
            relu_layer2 = self.reluLayer(bn_layer2, scope_name='relu_layer2')
            conv_layer2 = self.convLayer(relu_layer2, out_channel, ksize, stride=1, scope_name='conv_layer2')

            if increase_dim:
                identical_mapping = self.avgPoolLayer(inputMap, ksize=2, stride=2, scope_name='identical_pool')
                identical_mapping = tf.pad(identical_mapping, [[0, 0], [0, 0], [0, 0],
                                                               [(out_channel - in_channel) // 2,
                                                                (out_channel - in_channel) // 2]])
            else:
                identical_mapping = inputMap

            added = tf.add(conv_layer2, identical_mapping)

        return added

    def residualStageLayer(self, inputMap, ksize, out_channel, unit_num, stage_name, down_sampling, first_conv,
                           is_training, reuse=False):
        with tf.variable_scope(stage_name, reuse=reuse):
            _out = inputMap
            _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_1', down_sampling=down_sampling,
                                          first_conv=first_conv, is_training=is_training)
            for n in range(2, unit_num + 1):
                _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_' + str(n),
                                              down_sampling=False, first_conv=False, is_training=is_training)

        return _out

    def stage0_Block(self, inputMap, scope_name, out_channel, ksize, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            stage0_conv = self.convLayer(inputMap, out_channel, ksize=ksize, stride=1, scope_name='stage0_conv')
            stage0_bn = self.bnLayer(stage0_conv, scope_name='stage0_bn', is_training=self.is_training)
            stage0_relu = self.reluLayer(stage0_bn, scope_name='stage0_relu')

        return stage0_relu

    def classifier(self, inputMap, scope_name, is_training, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            bn = self.bnLayer(inputMap, scope_name='bn', is_training=is_training)
            relu = self.reluLayer(bn, scope_name='relu')
            gap = self.globalPoolLayer(relu, scope_name='gap')
            flatten = self.flattenLayer(gap, scope_name='flatten')
            pred = self.fcLayer(flatten, self.num_class, scope_name='pred')
            pred_softmax = tf.nn.softmax(pred, name='pred_softmax')

        return pred, pred_softmax

    def task_model(self, inputMap, ksize, stage1_num, stage2_num, stage3_num, out_channel1, out_channel2, out_channel3,
                   is_training, reuse_stage0=False, reuse_stage1=False, reuse_stage2=False, reuse_stage3=False,
                   reuse_classifier=False, extra_stage1_name='', extra_stage2_name='', extra_stage3_name=''):
        stage0 = self.stage0_Block(inputMap=inputMap, scope_name='stage0', out_channel=out_channel1, ksize=ksize,
                                   reuse=reuse_stage0)

        stage1 = self.residualStageLayer(inputMap=stage0, ksize=ksize, out_channel=out_channel1,
                                         unit_num=stage1_num, stage_name='stage1' + extra_stage1_name,
                                         down_sampling=False, first_conv=True, is_training=is_training,
                                         reuse=reuse_stage1)

        stage2 = self.residualStageLayer(inputMap=stage1, ksize=ksize, out_channel=out_channel2,
                                         unit_num=stage2_num, stage_name='stage2' + extra_stage2_name,
                                         down_sampling=True, first_conv=False, is_training=is_training,
                                         reuse=reuse_stage2)

        stage3 = self.residualStageLayer(inputMap=stage2, ksize=ksize, out_channel=out_channel3,
                                         unit_num=stage3_num, stage_name='stage3' + extra_stage3_name,
                                         down_sampling=True, first_conv=False, is_training=is_training,
                                         reuse=reuse_stage3)

        pred, pred_softmax = self.classifier(stage3, scope_name='classifier', is_training=is_training,
                                             reuse=reuse_classifier)

        return pred, pred_softmax

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x')
        tf.summary.image('Image/origin', self.x)
        self.y = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.pred, self.pred_softmax = self.task_model(inputMap=self.x, ksize=3, stage1_num=3, stage2_num=3,
                                                       stage3_num=3, out_channel1=16, out_channel2=32, out_channel3=64,
                                                       is_training=self.is_training,
                                                       reuse_stage0=False,
                                                       reuse_stage1=False,
                                                       reuse_stage2=False,
                                                       reuse_stage3=False,
                                                       reuse_classifier=False)

        with tf.variable_scope('loss'):
            # supervised loss
            self.supervised_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
            tf.summary.scalar('supervised_loss', self.supervised_loss)

        with tf.variable_scope('variables'):
            self.t_var = tf.trainable_variables()

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.task_trainOp = tf.train.AdamOptimizer(self.lr).minimize(self.supervised_loss,
                                                                             var_list=self.t_var)

        with tf.variable_scope('tfSummary'):
            self.merged = tf.summary.merge_all()
            if self.train_phase == 'Train':
                self.writer = tf.summary.FileWriter(self.ckptDir, self.sess.graph)

        with tf.variable_scope('saver'):
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.eps)

        with tf.variable_scope('accuracy'):
            self.distribution = [tf.argmax(self.y, 1), tf.argmax(self.pred_softmax, 1)]
            self.correct_prediction = tf.equal(self.distribution[0], self.distribution[1])
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))

    def f_value(self, matrix):
        f = 0.0
        length = len(matrix[0])
        for i in range(length):
            recall = matrix[i][i] / np.sum([matrix[i][m] for m in range(self.num_class)])
            precision = matrix[i][i] / np.sum([matrix[n][i] for n in range(self.num_class)])
            result = (recall * precision) / (recall + precision)
            f += result
        f *= (2 / self.num_class)

        return f

    def validation_procedure(self, validation_data, distribution_op, loss_op, inputX, inputY):
        confusion_matrics = np.zeros([self.num_class, self.num_class], dtype="int")
        val_loss = 0.0

        val_batch_num = int(np.ceil(validation_data[0].shape[0] / self.bs))
        for step in range(val_batch_num):
            _validationImg = validation_data[0][step * self.bs:step * self.bs + self.bs]
            _validationLab = validation_data[1][step * self.bs:step * self.bs + self.bs]

            [matrix_row, matrix_col], tmp_loss = self.sess.run([distribution_op, loss_op],
                                                               feed_dict={inputX: _validationImg,
                                                                          inputY: _validationLab,
                                                                          self.is_training: False})
            for m, n in zip(matrix_row, matrix_col):
                confusion_matrics[m][n] += 1

            val_loss += tmp_loss

        validation_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(self.num_class)])) / float(
            np.sum(confusion_matrics))
        validation_loss = val_loss / val_batch_num

        return validation_accuracy, validation_loss

    def test_procedure(self, test_data, distribution_op, inputX, inputY, mode):
        confusion_matrics = np.zeros([self.num_class, self.num_class], dtype="int")

        tst_batch_num = int(np.ceil(test_data[0].shape[0] / self.bs))
        for step in range(tst_batch_num):
            _testImg = test_data[0][step * self.bs:step * self.bs + self.bs]
            _testLab = test_data[1][step * self.bs:step * self.bs + self.bs]

            matrix_row, matrix_col = self.sess.run(distribution_op, feed_dict={inputX: _testImg,
                                                                               inputY: _testLab,
                                                                               self.is_training: False})
            for m, n in zip(matrix_row, matrix_col):
                confusion_matrics[m][n] += 1

        test_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(self.num_class)])) / float(
            np.sum(confusion_matrics))
        detail_test_accuracy = [confusion_matrics[i][i] / np.sum(confusion_matrics[i]) for i in
                                range(self.num_class)]
        log0 = "Mode: " + mode
        log1 = "Test Accuracy : %g" % test_accuracy
        log2 = np.array(confusion_matrics.tolist())
        log3 = ''
        for j in range(self.num_class):
            log3 += 'category %s test accuracy : %g\n' % (baseline_utils.pulmonary_category[j], detail_test_accuracy[j])
        log3 = log3[:-1]
        log4 = 'F_Value : %g\n' % self.f_value(confusion_matrics)

        baseline_utils.save2file(log0, self.ckptDir, self.model)
        baseline_utils.save2file(log1, self.ckptDir, self.model)
        baseline_utils.save2file(log2, self.ckptDir, self.model)
        baseline_utils.save2file(log3, self.ckptDir, self.model)
        baseline_utils.save2file(log4, self.ckptDir, self.model)

    def getBatchData(self):
        _src_tr_img_batch, _src_tr_lab_batch = init.next_batch(self.source_training_data[0],
                                                                  self.source_training_data[1], self.bs)

        feed_dict = {self.x: _src_tr_img_batch,
                     self.y: _src_tr_lab_batch,
                     self.is_training: True}
        feed_dict_eval = {self.x: _src_tr_img_batch,
                          self.y: _src_tr_lab_batch,
                          self.is_training: False}

        return feed_dict, feed_dict_eval

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.itr_epoch = len(self.source_training_data[0]) // self.bs

        training_acc = 0.0
        training_loss = 0.0

        for e in range(1, self.eps + 1):
            for itr in range(self.itr_epoch):
                feed_dict_train, feed_dict_eval = self.getBatchData()
                _ = self.sess.run(self.task_trainOp, feed_dict=feed_dict_train)

                _training_accuracy, _training_loss = self.sess.run([self.accuracy, self.supervised_loss],
                                                                   feed_dict=feed_dict_eval)

                training_acc += _training_accuracy
                training_loss += _training_loss

            summary = self.sess.run(self.merged, feed_dict=feed_dict_eval)

            training_acc = float(training_acc / self.itr_epoch)
            training_loss = float(training_loss / self.itr_epoch)

            validation_acc, validation_loss = self.validation_procedure(
                validation_data=self.validation_data, distribution_op=self.distribution,
                loss_op=self.supervised_loss, inputX=self.x, inputY=self.y)

            log1 = "Epoch: [%d], Training Accuracy: [%g], Validation Accuracy: [%g], Training Loss: [%g], " \
                   "Validation Loss: [%g], Time: [%s]" % (
                       e, training_acc, validation_acc, training_loss, validation_loss, time.ctime(time.time()))

            self.plt_epoch.append(e)
            self.plt_training_accuracy.append(training_acc)
            self.plt_training_loss.append(training_loss)
            self.plt_validation_accuracy.append(validation_acc)
            self.plt_validation_loss.append(validation_loss)

            baseline_utils.plotAccuracy(x=self.plt_epoch,
                                        y1=self.plt_training_accuracy,
                                        y2=self.plt_validation_accuracy,
                                        figName=self.model,
                                        line1Name='training',
                                        line2Name='validation',
                                        savePath=self.ckptDir)

            baseline_utils.plotLoss(x=self.plt_epoch,
                                    y1=self.plt_training_loss,
                                    y2=self.plt_validation_loss,
                                    figName=self.model,
                                    line1Name='training',
                                    line2Name='validation',
                                    savePath=self.ckptDir)

            baseline_utils.save2file(log1, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e))

            self.test_procedure(self.source_test_data, distribution_op=self.distribution, inputX=self.x, inputY=self.y,
                                mode='source')

            training_acc = 0.0
            training_loss = 0.0

    def test(self):
        try:
            self.saver.restore(self.sess, self.reload_path)
            print('Reload parameters finish')
        except:
            print('Reload failed')

        self.test_procedure(self.target_test_data, distribution_op=self.distribution, inputX=self.x, inputY=self.y,
                            mode='target')
