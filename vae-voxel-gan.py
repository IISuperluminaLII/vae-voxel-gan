import datetime
import tensorflow as tf
import os
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPool3D, AveragePooling3D, GaussianNoise, GaussianDropout, SpatialDropout3D
from keras.layers import BatchNormalization, Activation, ZeroPadding3D, concatenate, ConvLSTM2D, Layer
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import UpSampling2D, Conv3D, Conv3DTranspose, Deconvolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, Nadam
import keras.backend as K
import processVoxels as d
import visdom
from tensorboard.plugins.mesh import summary as msum
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np


class DCGAN():
    def __init__(self):
        # Input shape
        self.latent_dim = 200
        self.d_optimizer = Adam(0.00001, 0.5)
        self.g_optimizer = Adam(0.0025, 0.5)
        self.img_shape = (32, 32, 32, 1)
        self.discriminator = self.build_discriminator()
        # self.discriminator = multi_gpu_model(self.discriminator, gpus=1)'binary_crossentropy'
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.d_optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.encoder = self.build_encoder()
        self.encoder.summary()
        #self.encoder.compile(loss='mae', optimizer=self.g_optimizer, metrics=['accuracy'])
        print(self.encoder.input_shape)
        print(self.encoder.output_shape)

        # The generator takes noise as input and generates imgs

        imgs = Input(shape=(32, 32, 32, 1))

        encd = self.encoder(imgs)
        reconstructed = self.generator(encd)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # self.discriminator.compile(loss='binary_crossentropy', optimizer=self.d_optimizer, metrics=['accuracy'])

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(reconstructed)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(imgs, [reconstructed, validity])

        # self.combined = multi_gpu_model(self.combined, gpus=1)
        self.combined.compile(loss=['mse', 'binary_crossentropy'], optimizer=self.g_optimizer, metrics=['accuracy'])
        input("everythng good?")

    def build_encoder(self):
        # Encoder

        img = Input(shape=self.img_shape)
        # h = Flatten()(img)
        h = MaxPool3D(pool_size=(4, 4, 4), strides=2, input_shape=(32, 32, 32, 1), padding='same')(img)
        h = Conv3D(200, kernel_initializer='glorot_normal', kernel_size=4, strides=2, padding="same")(h)
        h = LeakyReLU(0.2)(h)
        # h = BatchNormalization()(h)
        # h = GaussianDropout(0.3)(h)
        # h = GaussianNoise(0.3)(h)
        h = MaxPool3D(pool_size=(4, 4, 4), strides=2, padding='same')(h)
        h = Conv3D(200, kernel_initializer='glorot_normal', kernel_size=4, strides=2, padding="valid")(h)
        h = LeakyReLU(0.2)(h)
        h = BatchNormalization()(h)
        h = Flatten()(h)
        # h = GaussianNoise(0.3)(h)
        mu = Dense(self.latent_dim, activation='sigmoid')(h)

        return Model(img, mu)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256 * 1 * 1 * 1, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((1, 1, 1, 256)))

        model.add(Conv3DTranspose(filters=128, kernel_size=(4, 4, 4),
                                  strides=(1, 1, 1),
                                  kernel_initializer='glorot_normal',
                                  padding='valid',
                                  name="1_C"))
        model.add(BatchNormalization(name="1_BN"))
        model.add(LeakyReLU(alpha=0.2, name="1_A"))

        model.add(Conv3DTranspose(filters=128, kernel_size=(4, 4, 4),
                                  strides=(2, 2, 2),
                                  kernel_initializer='glorot_normal',
                                  padding='same',
                                  name="2_C"))
        model.add(BatchNormalization(name="2_BN"))
        model.add(LeakyReLU(alpha=0.2, name="2_A"))

        model.add(Conv3DTranspose(filters=128, kernel_size=(4, 4, 4),
                                  strides=(2, 2, 2),
                                  kernel_initializer='glorot_normal',
                                  padding='same',
                                  name="3_C"))
        model.add(BatchNormalization(name="3_BN"))
        model.add(LeakyReLU(alpha=0.2, name="3_A"))

        model.add(Conv3DTranspose(filters=128, kernel_size=(4, 4, 4),
                                  strides=(2, 2, 2),
                                  kernel_initializer='glorot_normal',
                                  padding='same',
                                  name="4_C"))
        model.add(BatchNormalization(name="4_BN"))
        model.add(LeakyReLU(alpha=0.2, name="4_A"))

        model.add(Conv3D(filters=1, kernel_size=(4, 4, 4),
                         strides=(1, 1, 1),
                         kernel_initializer='glorot_normal',
                         padding='same',
                         name="5_C"))
        model.add(BatchNormalization(name="5_BN"))
        model.add(LeakyReLU(alpha=0.1, name="5_A"))

        model.add(Conv3DTranspose(filters=1, kernel_size=(4, 4, 4),
                                  strides=(1, 1, 1),
                                  kernel_initializer='glorot_normal',
                                  padding='same',
                                  name="7_C"))
        model.add(BatchNormalization(name="7_BN"))
        model.add(Activation("sigmoid", name="7_A"))


        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv3D(64, kernel_initializer='glorot_normal', kernel_size=4, input_shape=(32, 32, 32, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.4))
        #model.add(MaxPool3D(pool_size=(4, 4, 4), strides=2, padding='same'))
        model.add(SpatialDropout3D(0.2))

        model.add(Conv3D(128, kernel_initializer='glorot_normal', kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.4))
        model.add(SpatialDropout3D(0.2))

        model.add(Conv3D(128, kernel_initializer='glorot_normal', kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.4))
        model.add(SpatialDropout3D(0.4))

        model.add(Conv3D(128, kernel_initializer='glorot_normal', kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.4))
        model.add(SpatialDropout3D(0.2))

        model.add(Conv3D(128, kernel_initializer='glorot_normal', kernel_size=4, strides=2, padding="valid"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.4))
        model.add(SpatialDropout3D(0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        img = Input(shape=(32, 32, 32, 1))
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=16, save_interval=100):

        obj = 'chair'
        obj_ratio = 1.0
        volumes = d.getAll(obj=obj, train=True, cube_len=32, is_local=False, obj_ratio=obj_ratio)
        print('Using ' + obj + ' Data')
        volumes = volumes[..., np.newaxis].astype(np.float)
        # Rescale -1 to 1
        X_train = volumes
        # X_train = np.expand_dims(X_train, axis=3)
        #np.save("chairs_bruh_bruh_2", X_train)
        print(len(volumes))
        #X_train = np.load("chairs_bruh_bruh_2.npy")
        #print(X_train.max())
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # tensorboard = TensorBoard(log_dir=("gan/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"),
        #                           write_graph=True, update_freq='batch')
        #
        # tensorboard.set_model(self.generator)
        # tensorboard.set_model(self.discriminator)
        # tensorboard.set_model(self.combined)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 0.3, (batch_size, self.latent_dim)).astype(np.float32)
            #noise = np.asarray(tf.random.normal((batch_size, self.latent_dim), dtype='float32'))
            # with tf.device('/gpu:0'):

            # latent_fake = self.encoder.predict(imgs)
            # latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            encoded = self.encoder.predict(imgs)
            # if epoch % 500 == 0:
            #     print(encoded)
            #     print(np.min(encoded))
            #     print(noise)
            #     print(np.min(noise))
            gen_imgs = self.generator.predict(encoded)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(imgs, [imgs, valid])

            # Plot the progress
            print("%d [D[] loss: %f, acc.: %6.2f%%] [G loss: %f, mse: %f] [real: %f, acc:%6.2f%%][fake %f, acc:%.4f%%]"
                  % (epoch,
                     d_loss[0],
                     100 * d_loss[1],
                     g_loss[0],
                     g_loss[1],
                     d_loss_real[0],
                     100 * d_loss_real[1],
                     d_loss_fake[0],
                     (100 * d_loss_fake[1])
                     )
                  )

            #tensorboard.on_batch_begin(self.named_logs([self.generator, self.discriminator], [d_loss, g_loss]))

            # If at save interval => save generated image samples
            acc = int(100 * d_loss[1])
            new_env = str(str(int(epoch/100000)) + "-" + str(int(epoch/10000)) + "-" + str(int(epoch/1000)) + "-" + str(epoch))
            if epoch % save_interval == 0:
                vis = visdom.Visdom(env=new_env, log_to_filename="logs/backup.log")
                self.sample_images(epoch, vis, batch_size, volumes)
        #tensorboard.writer.flush()
    def named_logs(self, model, logs):
        result = {}
        for l in model:
            for j in zip(['Discriminator Loss', 'Discriminator Acc', 'Generator Loss','Generator Acc'], logs):
                result[j[0]] = j[1]
        return result

    def sample_images(self, epoch, vis, batch_size, volumes):

        mu, kappa = 0.0, 5.0
        #noise = np.random.normal(0, 0.3, (batch_size, self.latent_dim)).astype(np.float32)
        noise = np.random.normal(0, 0.3, (batch_size, self.latent_dim)).astype(np.float32)
        #noise_lv5 = np.random.wald(3, 2, (16, 200))
        #concat_noise = noise_lv4 + noise_lv5
        # noise_lv4 = np.random.vonmises(mu, kappa, (64, self.latent_dim))

        def write_session(voxels, strings):
            with tf.Session() as sess:
                mesh_sum = msum.op(strings, vertices=voxels)
                summaries = sess.run(mesh_sum)
                #writer.add_summary(summaries)

        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        #gen_imgs = 0.5 * gen_imgs + 0.5
        id_ch = np.random.randint(0, 16, 5)
        for i in range(4):
            try:
                voxels = np.squeeze(gen_imgs[id_ch[i]])
                d.plotVoxelVisdom_2(voxels, vis, '_'.join(map(str, [epoch, i, "rand_samp"])))
            except Exception as e:
                print("No")
            noise = np.random.normal(0, 0.3, (batch_size, self.latent_dim)).astype(np.float32)
            gen_imgs = self.generator.predict(noise)

        try:
            encded = self.encoder.predict(volumes[np.random.randint(0, volumes.shape[0], batch_size)])
            gen_imgs = self.generator.predict(encded)
            voxels = np.squeeze(gen_imgs[id_ch[4]])
            #print(voxels.shape)
            d.plotVoxelVisdom_2(voxels, vis, '_'.join(map(str, [epoch, 4, "REFERENCE"])))
        except Exception as e:
            print("Err: ", e)

        # try:
        #     voxels = np.squeeze(volumes[id_ch[1]])
        #     #print(voxels.shape)
        #     d.plotVoxelVisdom(voxels, vis, '_'.join(map(str, [epoch, 6, 0.5])))
        # except:
        #     print("Couldint print image: ", 5)


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=1000001, batch_size=16, save_interval=500)
