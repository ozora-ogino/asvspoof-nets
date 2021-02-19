import unittest
import tensorflow as tf
import numpy as np
from models import VGG_2D 

class TestLCNN(unittest.TestCase):
    def test_build(self):
        data_1 = np.random.randn(100, 100, 100, 1)
        data_2 = np.random.randn(100, 100, 100, 1)
        labels = np.random.randint(0, 2, 100)
        model = VGG_2D()
        model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(lr=0.01),
                      metrics=['acc'])
        model.fit([data_1, data_2], labels, epochs=1, batch_size=64, verbose=0)

if __name__ == '__main__':
    unittest.main()