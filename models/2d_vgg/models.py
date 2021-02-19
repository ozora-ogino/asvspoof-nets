import tensorflow as tf

class VGG_2D(tf.keras.models.Model):
    def __init__(self):
        super(VGG_2D, self).__init__()
        self.conv_1_1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.conv_1_2 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,1), strides=None, padding='same')

        self.conv_2_1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.conv_2_2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(2,1), strides=None, padding='same')

        self.conv_3_1 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.conv_3_2 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='same')

        self.conv_4_1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.conv_4_2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.maxpool_4 = tf.keras.layers.MaxPool2D(pool_size=(2,1), strides=None, padding='same')

        self.conv_5_1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.conv_5_2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.maxpool_5 = tf.keras.layers.MaxPool2D(pool_size=(2,1), strides=None, padding='same')

        self.conv_6_1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.conv_6_2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.maxpool_6 = tf.keras.layers.MaxPool2D(pool_size=(2,1), strides=None, padding='same')

        self.meanpool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense_2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(2, activation=tf.nn.softmax)


    def call(self, inputs, training=False):
        x_1 = self.conv_1_1(inputs[0])
        x_2 = self.conv_1_2(inputs[1])
        x_1 = self.maxpool_1(x_1)
        x_2 = self.maxpool_1(x_2)
        
        x_1 = self.conv_2_1(x_1)
        x_2 = self.conv_2_2(x_2)
        x_1 = self.maxpool_2(x_1)
        x_2 = self.maxpool_2(x_2)
        
        x_1 = self.conv_3_1(x_1)
        x_2 = self.conv_3_2(x_2)
        x_1 = self.maxpool_3(x_1)
        x_2 = self.maxpool_3(x_2)

        x_1 = self.conv_4_1(x_1)
        x_2 = self.conv_4_2(x_2)
        x_1 = self.maxpool_4(x_1)
        x_2 = self.maxpool_4(x_2)

        x_1 = self.conv_5_1(x_1)
        x_2 = self.conv_5_2(x_2)
        x_1 = self.maxpool_5(x_1)
        x_2 = self.maxpool_4(x_2)

        x_1 = self.conv_6_1(x_1)
        x_2 = self.conv_6_2(x_2)
        x_1 = self.maxpool_6(x_1)
        x_2 = self.maxpool_4(x_2)

        x = tf.keras.layers.concatenate([x_1, x_2], axis=3)
        x = self.meanpool(x)
        x = self.flatten(x)
        
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.out(x)

        return x



