import math

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from chainer.functions.math import exponential, average
from tensorflow import Variable
from tensorflow.keras.optimizers import Adam, RMSprop
tf.compat.v1.enable_eager_execution()
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


from model import generator_model,discriminator_model
import tensorflow.keras.datasets.mnist as mnist
(train_image, train_label), (_, _) = mnist.load_data()
train_image = train_image / 127.5  - 1
train_image = np.expand_dims(train_image, -1)
dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))
BATCH_SIZE = 256
n_categorical = 10
n_continuous = 2
n_z = 64
image_count = train_image.shape[0]
im_shape = train_image.shape[1:]
dataset = dataset.shuffle(image_count).batch(BATCH_SIZE)


def gaussian_nll(x, mean, ln_var, reduce='sum'):
    """Computes the negative log-likelihood of a Gaussian distribution.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function computes in
    elementwise manner the negative log-likelihood of :math:`x` on a
    Gaussian distribution :math:`N(\\mu, S)`,

    .. math::

        -\\log N(x; \\mu, \\sigma^2) =
        \\log\\left(\\sqrt{(2\\pi)^D |S|}\\right) +
        \\frac{1}{2}(x - \\mu)^\\top S^{-1}(x - \\mu),

    where :math:`D` is a dimension of :math:`x` and :math:`S` is a diagonal
    matrix where :math:`S_{ii} = \\sigma_i^2`.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'sum'`` or ``'mean'``, loss values are summed up
    or averaged respectively.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        mean (:class:`~chainer.Variable` or :ref:`ndarray`): A variable
            representing mean of a Gaussian distribution, :math:`\\mu`.
        ln_var (:class:`~chainer.Variable` or :ref:`ndarray`): A variable
            representing logarithm of variance of a Gaussian distribution,
            :math:`\\log(\\sigma^2)`.
        reduce (str): Reduction option. Its value must be either
            ``'sum'``, ``'mean'`` or ``'no'``. Otherwise, :class:`ValueError`
            is raised.

    Returns:
        ~chainer.Variable:
            A variable representing the negative log-likelihood.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum'`` or ``'mean'``, the output variable holds a
            scalar value.

    """
    if reduce not in ('sum', 'mean', 'no'):
        raise ValueError(
            'only \'sum\', \'mean\' and \'no\' are valid for \'reduce\', but '
            '\'%s\' is given' % reduce)

    x_prec = exponential.exp(-ln_var)
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (ln_var + math.log(2 * math.pi)) / 2 - x_power
    if reduce == 'sum':
        return sum.sum(loss)
    elif reduce == 'mean':
        return average.average(loss)
    else:
        return loss


def rnd_categorical(n, n_categorical):
    """
    :param n: batch size
    :param n_categorical:  分类数量
    :return: one_hot编码和其为数字0-9的实际数字
    """
    indices = np.random.randint(n_categorical, size=n)
    one_hot = np.zeros((n, n_categorical))
    one_hot[np.arange(n), indices] = 1
    return np.asarray(one_hot,dtype=np.float32), np.asarray(indices,dtype=np.float32)

def rnd_continuous(n, n_continuous, mu=0, std=1):
    return np.random.normal(mu, std, size=(n, n_continuous))


def discriminator_loss(real_y,real_categories_out,fake_y,real_categories_indices,n_continuous_out,n_continuous_in):
    pass
def generator_loss(fake_y, fake_categories_out, real_categories_indices, n_continuous_out, n_continuous_in):
    pass



# def discriminator_loss(real_output, real_cat_out, fake_output, label, con_out, cond_in):
#     real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
#     cat_loss = category_cross_entropy(label, real_cat_out)
#
#     con_loss = tf.reduce_mean(tf.square(con_out - cond_in))
#     total_loss = real_loss + fake_loss + cat_loss + con_loss
#     return total_loss
#
#
# def generator_loss(fake_output, fake_cat_out, label, con_out, cond_in):
#     print(label.shape)
#     print(fake_cat_out.shape)
#     fake_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)
#     cat_loss = category_cross_entropy(label, fake_cat_out)
#
#     con_loss = tf.reduce_mean(tf.square(con_out - cond_in))
#     return fake_loss + cat_loss + con_loss

generator = generator_model(n_z+n_continuous+n_categorical)
discriminator = discriminator_model(im_shape)
generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
# binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# category_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
@tf.function
def train_step(image):
    noise = np.random.uniform(-1,1,(BATCH_SIZE,n_z)).astype(np.float32)
    cat_onehot, cat_indices = rnd_categorical(BATCH_SIZE, n_categorical)
    continuous = np.asarray(rnd_continuous(BATCH_SIZE, n_continuous), dtype=np.float32)
    gz = np.concatenate((noise,continuous,cat_onehot),axis=1)


    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        fake_x  = generator(gz, training=True)
        fake_y,mi = discriminator(fake_x,training = True)
        real_y,_ = discriminator(image,training = True)


        generator_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
        generator_loss = tf.nn.softmax_cross_entropy(fake_y, np.ones(BATCH_SIZE, dtype=np.int32))
        discriminator_loss = tf.nn.softmax_cross_entropy(fake_y, np.zeros(BATCH_SIZE, dtype=np.int32))
        discriminator_loss += tf.nn.softmax_cross_entropy(real_y, np.ones(BATCH_SIZE, dtype=np.int32))

        # Mutual Information loss
        mi_cat_onehot, mi_continuous_mean = mi[:,0:n_categorical],mi[:,n_categorical-1:]

        # Categorical loss
        categorical_loss = tf.nn.softmax_cross_entropy(mi_cat_onehot, cat_indices)

        # Continuous loss - Fix standard deviation to 1, i.e. log variance is 0
        mi_continuous_ln_var = np.empty_like(mi_continuous_mean.data, dtype=np.float32)
        mi_continuous_ln_var.fill(1)

        # mi_continuous_ln_var.fill(1e-6)
        # 高斯分布的负对数似然函数
        continuous_loss = gaussian_nll(mi_continuous_mean, Variable(continuous), Variable(mi_continuous_ln_var))
        continuous_loss /= BATCH_SIZE

        generator_loss += categorical_loss
        generator_loss += continuous_loss

    gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))





# def generate_and_save_images(model, test_noise_input, test_cat_input, epoch,cond):
#     print('Epoch:', epoch + 1)
#     # Notice `training` is set to False.
#     # This is so all layers run in inference mode (batchnorm).
#     cond_seed = cond
#     predictions = model((test_noise_input, cond_seed, test_cat_input), training=False)
#     predictions = tf.squeeze(predictions)
#     # fig = plt.figure(figsize=(10, 1))
#
#     # for i in range(predictions.shape[0]):
#     #     plt.subplot(1, 10, i + 1)
#     #     plt.imshow((predictions[i, :, :] + 1) / 2, cmap='gray')
#     #     plt.axis('off')
#     #
#     #     plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#     # plt.show()


def train(dataset, epochs):
    label = np.random.randint(0, 10, size=(10, 1))
    c_continuous = np.asarray(rnd_continuous(1, 2), dtype=np.float32)
    noise = np.random.uniform(-1, 1, (BATCH_SIZE, n_z)).astype(np.float32)

    for epoch in range(epochs):
        for image in dataset:
            loss = train_step(image[0])
        tf.print("gen_loss", loss[0])
        tf.print("disc_loss", loss[1])
    #     if epoch % 10 == 0:
    #         generate_and_save_images(generator,
    #                                  noise,
    #                                  label,
    #                                  epoch,c_continuous)
    #
    # generate_and_save_images(generator,
    #                          noise,
    #                          label,
    #                          epoch,c_continuous)


EPOCHS = 200
train(dataset, EPOCHS)

generator.save('generate_infogan.h5')
# generator = load_model('generate_infogan.h5')
# num = 10
# noise_seed = tf.random.normal([num, noise_dim])
# cat_seed = np.arange(10).reshape(-1, 1)
# print(cat_seed.T)
# cond1 = tf.convert_to_tensor(np.ones([num,con_dim]))
# generate_and_save_images(generator,noise_seed,cat_seed,0,cond)
#




