import tensorflow as tf
import numpy as np

print('TensorFlow version: ', tf.__version__)
print('Is eager execution enabled? ', tf.executing_eagerly())

def fizzBuzz(limit):

    print('Is limit a tensor? %s' %tf.is_tensor(limit))

    if (not tf.is_tensor(limit)):
        limit = tf.convert_to_tensor(limit)
        print('Is limit a tensor? %s' %tf.is_tensor(limit))

    for i in tf.range(1, limit + 1):
        if (i % 3 == 0 and i % 5 == 0):
            print('FizzBuzz')
        elif (i % 3 == 0):
            print('Fizz')
        elif (i % 5 == 0):
            print('Buzz')
        else:
            print(i.numpy())

fizzBuzz(tf.constant(15))