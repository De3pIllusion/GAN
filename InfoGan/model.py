
from tensorflow.keras import *
from  tensorflow.keras.models import Model
from  tensorflow.keras.initializers import RandomNormal
import  tensorflow.keras.backend as K
from  tensorflow.keras.utils import plot_model

def generator_model(gz):
    input = Input(gz)
    # noise_seed = Input((BATCH_SIZE,noise_dim))
    # con_seed = Input((BATCH_SIZE,2))
    # label = Input(())
    # x = layers.Embedding(10, 256, input_length=1)(label)
    # x = layers.Flatten()(x)
    # x = layers.concatenate([noise_seed, con_seed, label],axis=1)
    x = layers.Dense(3 * 3 * 128, use_bias=False)(input)
    x = layers.Reshape((3, 3, 128))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)  # 7*7

    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)  # 14*14

    x = layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.Activation('tanh')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    return model


def discriminator_model(imshape):
    image =  Input(imshape)

    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(image)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(32 * 2, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(32 * 4, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x1 = layers.Dense(1)(x)
    x2 = layers.Dense(12,activation="softmax")(x)

    model = Model(inputs=image, outputs=[x1, x2])

    return model
