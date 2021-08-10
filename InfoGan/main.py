from model import *

# mi_loss
def mi_loss(c, q_of_c_give_x):
    """mi_loss = -c * log(Q(c|x))
    """
    return K.mean(-K.sum(K.log(q_of_c_give_x + K.epsilon()) * c, axis=1))


def build_and_train_models(latent_size=100):
    """Load the dataset, build InfoGAN models,
    Call the InfoGAN train routine.
    """
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255.
    num_labels = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train)

    # 超参数
    model_name = 'infogan_mnist'
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels,)
    code_shape = (1,)

    # discriminator model
    inputs = keras.layers.Input(shape=input_shape, name='discriminator_input')
    # discriminator with 4 outputs
    discriminator_model = discriminator(inputs, num_labels=num_labels, num_codes=2)
    optimizer = keras.optimizers.RMSprop(lr=lr, decay=decay)
    loss = ['binary_crossentropy', 'categorical_crossentropy', mi_loss, mi_loss]
    loss_weights = [1.0, 1.0, 0.5, 0.5]
    discriminator_model.compile(loss=loss,
                                loss_weights=loss_weights,
                                optimizer=optimizer,
                                metrics=['acc'])
    discriminator_model.summary()
    input_shape = (latent_size,)
    inputs = keras.layers.Input(shape=input_shape, name='z_input')
    labels = keras.layers.Input(shape=label_shape, name='labels')
    code1 = keras.layers.Input(shape=code_shape, name='code1')
    code2 = keras.layers.Input(shape=code_shape, name='code2')
    generator_model = generator(inputs, image_size, labels=labels, codes=[code1, code2])
    generator_model.summary()
    optimizer = keras.optimizers.RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    discriminator_model.trainable = False
    inputs = [inputs, labels, code1, code2]
    adversarial_model = keras.Model(inputs,
                                    discriminator_model(generator_model(inputs)),
                                    name=model_name)
    adversarial_model.compile(loss=loss, loss_weights=loss_weights,
                              optimizer=optimizer,
                              metrics=['acc'])
    adversarial_model.summary()

    models = (generator_model, discriminator_model, adversarial_model)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, data, params)


def train(models, data, params):
    """Train the network
    #Arguments
        models (Models): generator,discriminator,adversarial model
        data (tuple): x_train,y_train data
        params (tuple): Network params
    """
    generator, discriminator, adversarial = models
    x_train, y_train = data
    batch_size, latent_size, train_steps, num_labels, model_name = params

    save_interval = 500
    code_std = 0.5
    noise_input = np.random.uniform(-1.0, 1., size=[16, latent_size])
    noise_label = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    noise_code1 = np.random.normal(scale=code_std, size=[16, 1])
    noise_code2 = np.random.normal(scale=code_std, size=[16, 1])
    train_size = x_train.shape[0]
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_label, axis=1))
    for i in range(train_steps):
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        real_labels = y_train[rand_indexes]
        # random codes for real images
        real_code1 = np.random.normal(scale=code_std, size=[batch_size, 1])
        real_code2 = np.random.normal(scale=code_std, size=[batch_size, 1])
        # 生成假图片，标签和编码
        noise = np.random.uniform(-1., 1., size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]
        fake_code1 = np.random.normal(scale=code_std, size=[batch_size, 1])
        fake_code2 = np.random.normal(scale=code_std, size=[batch_size, 1])
        inputs = [noise, fake_labels, fake_code1, fake_code2]
        fake_images = generator.predict(inputs)
        x = np.concatenate((real_images, fake_images))
        labels = np.concatenate((real_labels, fake_labels))
        codes1 = np.concatenate((real_code1, fake_code1))
        codes2 = np.concatenate((real_code2, fake_code2))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0
        # train discriminator network
        outputs = [y, labels, codes1, codes2]
        # metrics = ['loss', 'activation_1_loss', 'label_loss',
        # 'code1_loss', 'code2_loss', 'activation_1_acc',
        # 'label_acc', 'code1_acc', 'code2_acc']
        metrics = discriminator.train_on_batch(x, outputs)
        fmt = "%d: [dis: %f, bce: %f, ce: %f, mi: %f, mi:%f, acc: %f]"
        log = fmt % (i, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[6])
        # train the adversarial network
        noise = np.random.uniform(-1., 1., size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]
        fake_code1 = np.random.normal(scale=code_std, size=[batch_size, 1])
        fake_code2 = np.random.normal(scale=code_std, size=[batch_size, 1])
        y = np.ones([batch_size, 1])
        inputs = [noise, fake_labels, fake_code1, fake_code2]
        outputs = [y, fake_labels, fake_code1, fake_code2]
        metrics = adversarial.train_on_batch(inputs, outputs)
        fmt = "%s [adv: %f, bce: %f, ce: %f, mi: %f, mi:%f, acc: %f]"
        log = fmt % (log, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[6])

        print(log)
        if (i + 1) % save_interval == 0:
            # plot generator images on a periodic basis
            plot_images(generator,
                        noise_input=noise_input,
                        noise_label=noise_label,
                        noise_codes=[noise_code1, noise_code2],
                        show=False,
                        step=(i + 1),
                        model_name=model_name)

        # save the model
        if (i + 1) % (2 * save_interval) == 0:
            generator.save(model_name + ".h5")


def plot_images(generator,
                noise_input,
                noise_label=None,
                noise_codes=None,
                show=False,
                step=0,
                model_name="gan"):
    """Generate fake images and plot them

    For visualization purposes, generate fake images
    then plot them in a square grid

    # Arguments
        generator (Model): The Generator Model for
            fake images generation
        noise_input (ndarray): Array of z-vectors
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images
        model_name (string): Model name

    """
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    rows = int(math.sqrt(noise_input.shape[0]))
    if noise_label is not None:
        noise_input = [noise_input, noise_label]
        if noise_codes is not None:
            noise_input += noise_codes

    images = generator.predict(noise_input)
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')
#模型训练
build_and_train_models(latent_size=62)
