import os
import sys

def init_new_model(name):
    dir = './models/' + name
    if os.path.exists(dir):
        raise Exception('./models/' + name + ' already exists.')
    os.mkdir(dir)
    with open(dir+'/models.py', 'w') as f:
        f.write(template_models)
    
    with open(dir+'/description.md', 'w') as f:
        f.write(template_description)
    
    with open(dir+'/test_models.py', 'w') as f:
        f.write(template_test)


template_test = '''
import unittest
import  tensorflow as tf
from models import #YOUR_MODE

class TestModel(unittest.TestCase):
    def test_build(self):
        # Example: data = np.random.randn(100, 100, 100, 1)
        data = # Test data
        # Example: labels = np.random.randint(0, 2, 100)
        labels = # Labels

        model = # Your model here
        model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(lr=0.01),
                      metrics=['acc'])
        model.fit(data, labels, epochs=1, verbose=0)

if __name__ == '__main__':
    unittest.main()
'''

template_models = '''
import  tensorflow as tf

class YOUR_MODEL(tf.keras.models.Model):
    def __init__(self):
        super(YOUR_MODEL, self).__init__()

    def call(self, inputs):

'''

template_description = '''
# Model Name

## Reference
***[Referense](https://something.com)***
[ Authors ]

## Description

'''


if __name__ == '__main__':
    name = sys.argv[1]
    init_new_model(name)