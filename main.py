import tensorflow.compat.v1 as tf
import numpy as np
import cv2

tf.disable_eager_execution()

def main():
    image_size = 64
    num_channels = 3
    images = []

    file_name = 'cat.jpg'
    # file_name = 'dog.jpg'
    image = cv2.imread(file_name)
    cv2.imshow('Image for prediction', image)

    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)

    x_batch = images.reshape(1, image_size, image_size, num_channels)

    sess = tf.Session()
    saver = tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-7975.meta')

    saver.restore(sess, './dogs-cats-model/dog-cat.ckpt-7975')

    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")

    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    res_label = ['dog', 'cat']
    print(res_label[result.argmax()])
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
