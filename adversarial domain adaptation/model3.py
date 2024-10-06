import tensorflow as tf
from keras import layers
_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.3)
from cnn_util import conv1d_block,GradientReversalLayer
import numpy as np
class DANN(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(DANN, self).__init__()

        self.feature_extractor,features = self.build_FeatureExtractor(input_shape)
        self.label_classifier= self.build_LabelPredictor((features.shape[1],),num_classes)
        self.domain_classifier = self.build_domain_classifier((features.shape[1],))

        self.optimizer_task = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)
        self.optimizer_enc = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)
        self.optimizer_disc = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)

        self.loss_cls = tf.keras.losses.CategoricalCrossentropy()  # for label predictor
        self.loss_domain = tf.keras.losses.BinaryCrossentropy()  # for domain classifier
    def build_FeatureExtractor(self,input_shape):
        in_image = layers.Input(shape=input_shape)
        #first module
        x = layers.Reshape((128, 1))(in_image)

        g=conv1d_block(16,5,2,'same',x) #None,64,16
        # g = conv1d_block(32, 5, 2, 'same', g)
        # g = conv1d_block(32, 5, 2, 'same', g)
        g=conv1d_block(8,5,2,'same',g)#None, 32,8
        # g = conv1d_block(32, 5, 2, 'same', g)

        features = layers.Flatten()(g)
        # print(features.shape)
        model=tf.keras.models.Model(inputs=in_image, outputs=features)
        return model,features

    def build_LabelPredictor(self,input_shape,num_classes):
        input_features = layers.Input(shape=input_shape)

        g=layers.Dense(64)(input_features)
        g = layers.BatchNormalization()(g)
        g=layers.Activation('relu')(g)

        g = layers.Dense(32)(g)
        g = layers.BatchNormalization()(g)
        g = layers.Activation('relu')(g)

        logits = layers.Dense(num_classes, activation='softmax')(g)
        model_label = tf.keras.models.Model(inputs=input_features, outputs=logits)
        return model_label
    def build_domain_classifier(self,input_shape):  # domain classifier
        input_features = layers.Input(shape=input_shape)
        g=GradientReversalLayer()(input_features)

        g = layers.Reshape((256, 1))(g)
        g=conv1d_block(16,5,2,'same',g)
        g = layers.MaxPooling1D(strides=2)(g)
        g=conv1d_block(8,5,2,'same',g)
        g = layers.MaxPooling1D(strides=2)(g)
        g = layers.Flatten()(g)

        logits = layers.Dense(1, activation=None)(g)
        model = tf.keras.models.Model(inputs=input_features, outputs=logits)
        return model
    def label_classifier_loss(self,y_true, y_pred):
        return self.loss_cls(y_true, y_pred)
    def domain_classifier_loss(self,y_true, y_pred):
        return self.loss_domain(y_true, y_pred)

    def domain_classifier_loss_g(self,y_pred):
        g_loss=-tf.reduce_mean(y_pred)

        return g_loss

    def wasserstein_gradient_penalty(self, x, x_fake):
        # temp_shape = [x.shape[0]]+[1 for _ in  range(len(x.shape)-1)]

        # epsilon = tf.random.uniform([], 0.0, 1.0)
        batch_size = tf.shape(x)[0]  # 获取实际的 batch 大小
        epsilon = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_fake

        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.domain_classifier(x_hat, training=False)
        gradients = t.gradient(d_hat, x_hat)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        gradient_penalty = 1 * tf.reduce_mean((slopes - 1.0) ** 2)

        return gradient_penalty
    def domain_wasserstein_loss_d(self, y_true,y_pred):
        d_loss=tf.reduce_mean(y_pred)-tf.reduce_mean(y_true)

        return d_loss
    def domain_wasserstein_loss_g(self, y_pred):
        g_loss=-tf.reduce_mean(y_pred)

        return g_loss
    def train_loss(self, source_img, source_label, target_img):
        # train_classifier
        with tf.GradientTape() as task_tape,tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:
            # Forward pass
            Xs_enc = self.feature_extractor(source_img,training=True)
            ys_pred = self.label_classifier(Xs_enc, training=True)
            ys_disc = self.domain_classifier(Xs_enc, training=True)

            Xt_enc = self.feature_extractor(target_img,training=True)
            yt_disc = self.domain_classifier(Xt_enc, training=True)

            ## Compute gradient penalty
            gp = self.wasserstein_gradient_penalty(Xs_enc, Xt_enc)


            # Compute the loss value
            task_loss = self.label_classifier_loss(source_label, ys_pred)
            disc_loss_enc=tf.reduce_mean(ys_disc)-tf.reduce_mean(yt_disc)

            enc_loss=task_loss-0.1*disc_loss_enc

            disc_loss=disc_loss_enc+gp


            task_loss += sum(self.label_classifier.losses)
            disc_loss += sum(self.domain_classifier.losses)
            enc_loss += sum(self.feature_extractor.losses)

        # Compute gradients
        trainable_vars_task = self.label_classifier.trainable_variables
        trainable_vars_enc = self.feature_extractor.trainable_variables
        trainable_vars_disc = self.domain_classifier.trainable_variables

        gradients_task = task_tape.gradient(task_loss, trainable_vars_task)
        gradients_enc = enc_tape.gradient(enc_loss, trainable_vars_enc)
        gradients_disc = disc_tape.gradient(disc_loss, trainable_vars_disc)

        # Update weights
        self.optimizer_task.apply_gradients(zip(gradients_task, trainable_vars_task))
        self.optimizer_enc.apply_gradients(zip(gradients_enc, trainable_vars_enc))
        self.optimizer_disc.apply_gradients(zip(gradients_disc, trainable_vars_disc))

        return disc_loss,enc_loss,task_loss

    def train_loss2(self, source_img, source_label):
        with tf.GradientTape() as tape:
            source_features = self.feature_extractor(source_img,training=True)
            source_pred = self.label_classifier(source_features,training=True)
            loss_s_class = self.label_classifier_loss(source_label, source_pred)
            loss_main = loss_s_class

        var_g = self.feature_extractor.trainable_variables+\
                self.label_classifier.trainable_variables
        main_gradients = tape.gradient(loss_main, var_g)
        self.optimizer_task.apply_gradients(zip(main_gradients, var_g))
        return  loss_main

    @tf.function
    def train_on_step(self, source_img,source_label,target_img):
        loss = self.train_loss(source_img,source_label,target_img)
        return loss
    @tf.function
    def train_on_step2(self, source_img,source_label):
        class_loss = self.train_loss2(source_img,source_label)
        return class_loss


    def train(self,source_train_db,target_train_db,source_test_db,target_test_db,epochs):

        loss_cls=[]
        loss_domian=[]
        loss_discrimiative=[]
        accuracy_s_t=[]
        accuracy_t_t = []

        for epoch in range(epochs):
            for (source_img, source_label), (target_img, _) in zip(source_train_db, target_train_db):
                domain_loss,enc_loss,task_loss=self.train_on_step(source_img,source_label,target_img)

            #for testing
            # acc in source test
            correct, total = 0, 0
            for sx_T, sy_T in source_test_db:
                pred_sy_T = self.label_classifier(self.feature_extractor(sx_T))
                pred_y_T = tf.cast(tf.argmax(pred_sy_T, axis=-1),tf.int32)
                y_T = tf.cast(sy_T, tf.int32)
                correct += float(tf.reduce_sum(tf.cast(tf.equal(pred_y_T, y_T), tf.float32)))
                total += sx_T.shape[0]
                acc_source_test = correct / total
                accuracy_s_t.append(acc_source_test)

            #acc in target test
            correct2, total2 = 0, 0
            for tx_T, ty_T in target_test_db:
                pred_ty_T = self.label_classifier(self.feature_extractor(tx_T))
                pred_y_T2 = tf.cast(tf.argmax(pred_ty_T, axis=-1),tf.int32)
                y_T2 = tf.cast(ty_T, tf.int32)
                correct2 += float(tf.reduce_sum(tf.cast(tf.equal(pred_y_T2, y_T2), tf.float32)))
                total2 += tx_T.shape[0]
                acc_target_test = correct2 / total2
                accuracy_t_t.append(acc_target_test)

            print('epoch:', epoch, 'task_loss:', task_loss.numpy(),
                  'acc in source',acc_source_test,'acc in target',acc_target_test)


        return np.array(accuracy_s_t), np.array(accuracy_t_t), np.array(loss_cls), np.array(loss_domian), np.array(
                loss_discrimiative)

    def train_source_only(self, source_train_db, source_test_db, target_test_db, epochs):

        for epoch in range(epochs):
            for source_img, source_label in source_train_db:
                cls_loss = self.train_on_step2(source_img, source_label)

            #for testing
            # acc in source test
            correct, total = 0, 0
            for sx_T, sy_T in source_test_db:
                pred_sy_T = self.label_classifier(self.feature_extractor(sx_T))

                pred_y_T = tf.cast(tf.argmax(pred_sy_T, axis=-1),tf.int32)
                y_T = tf.cast(sy_T, tf.int32)
                correct += float(tf.reduce_sum(tf.cast(tf.equal(pred_y_T, y_T), tf.float32)))
                total += sx_T.shape[0]
                acc_source_test = correct / total

            #acc in target test
            correct2, total2 = 0, 0
            for tx_T, ty_T in target_test_db:
                pred_ty_T = self.label_classifier(self.feature_extractor(tx_T))
                pred_y_T2 = tf.cast(tf.argmax(pred_ty_T, axis=-1),tf.int32)
                y_T2 = tf.cast(ty_T, tf.int32)
                correct2 += float(tf.reduce_sum(tf.cast(tf.equal(pred_y_T2, y_T2), tf.float32)))
                total2 += tx_T.shape[0]
                acc_target_test = correct2 / total2
            print('epoch:', epoch, 'cls_loss:', cls_loss.numpy(),'acc in source',acc_source_test,'acc in target',acc_target_test)
