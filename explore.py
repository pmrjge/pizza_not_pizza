from typing import Optional
import jax
import jax.numpy as jn
import jax.random as jr
import jax.nn as jnn

import haiku as hk
import haiku.initializers as hki

import numpy as np

import cv2
import os

import functools as ft

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            i = img_rgb.resize((384, 384, 3))
            images.append(i)

    return np.array(images)

pizza_imgs = load_images_from_folder('data/pizza')
not_pizza_imgs = load_images_from_folder('data/not_pizza')

aux_pizza_imgs = np.random.shuffle(pizza_imgs)
aux_not_pizza_imgs = np.random.shuffle(not_pizza_imgs)

N1 = aux_pizza_imgs.shape[0]
N2 = aux_not_pizza_imgs.shape[0]

limit1 = int(0.25 * N1)
limit2 = int(0.25 * N2)

test_pizza = aux_pizza_imgs[:limit1]
train_pizza = aux_pizza_imgs[limit1:]

test_not_pizza = aux_not_pizza_imgs[:limit2]
train_not_pizza = aux_not_pizza_imgs[limit2:]
        
class RandomSampler:
    def __init__(self, pizzas, not_pizzas, batch_size, num_devices, *, key):
        self.pizzas = pizzas
        self.not_pizzas = not_pizzas
        self.key = key
        self.batch_size = batch_size
        self.num_devices = num_devices

    def sample(self):
        pizzas = self.pizzas
        not_pizzas = self.not_pizzas
        key = self.key
        batch_size = self.batch_size
        num_devices = self.num_devices
        kk = batch_size // num_devices
        dd = batch_size // 2
        n = pizzas.shape[0]
        def generator():
            while True:
                key, k1, k2 = jr.split(key, 3)
                perm1 = jax.random.choice(k1, n, shape=(dd,))
                perm2 = jax.random.choice(k2, n, shape=(dd,))
                p = pizzas[perm1]
                not_p = not_pizzas[perm2]
                xx = jn.stack([p, not_p], axis=0)
                yy = jn.stack([jn.ones(shape=(dd,)), jn.zeros(shape=(dd,))], axis=0).expand_dims(axis=1)
                return xx.reshape(num_devices, kk, *xx.shape[1:]), yy.reshape(num_devices, kk, *yy.shape[1:])

        return generator()


class ConvResBlock(hk.Module):
    def __init__(self, kernel_size, strides, padding, dropout, name: Optional[str] = None):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dropout = dropout

    def __call__(self, inputs):
        features = x.inputs[-1]
        init = hki.VarianceScaling(0.01)
        x = hk.Conv2D(output_channels=features, kernel_shape=self.kernel_size, stride=self.strides, padding="same", w_init=init)(inputs)
        x = jnn.gelu(x)
        x = hk.Conv2D(output_channels=2 * features, kernel_shape=self.kernel_size, stride=self.strides, padding="same", w_init=init)(x)
        x = jnn.gelu(x)
        x = hk.Conv2D(output_channels=2 * features, kernel_shape=self.kernel_size, stride=self.strides, padding="same",w_init=init)(x)
        x = jnn.gelu(x)
        x = hk.Conv2D(output_channels=features, kernel_shape=self.kernel_size, stride=self.strides, padding="same",w_init=init)(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = jnn.gelu(x)
        return x + inputs


class ConvNet(hk.Module):
    def __init__(self, dropout, name: Optional[str]=None):
        super().__init__(name)
        self.dropout = dropout

    def __call__(self, x, is_training=True):
        dropout = self.dropout if is_training else 0.0
        x = hk.Conv2D(output_channels=32, kernel_shape=(3, 3), stride=(1, 1), padding="same")
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.gelu(x)
        x = hk.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="same")
        x = hk.Conv2D(output_channels=64, kernel_shape=(3, 3), stride=(1, 1), padding="same")
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.gelu(x)
        x = hk.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="same")
        x = hk.Conv2D(output_channels=128, kernel_shape=(3, 3), stride=(1, 1), padding="same")
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.gelu(x)
        x = hk.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="same")

        for i in range(4):
            x = ConvResBlock((3, 3), (1, 1), "same", dropout=dropout)(x)
        
        x = hk.Conv2D(output_channels=256, kernel_shape=(3, 3), stride=(1, 1), padding="same")
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.gelu(x)
        x = hk.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="same")

        for i in range(4):
            x = ConvResBlock((3, 3), (1, 1), "same", dropout=dropout)(x)

        x = hk.Flatten()(x)
        logits = hk.Linear(1)(x)
        return logits - jnn.logsumexp(logits)



@ft.partial(jax.jit, static_argnums=(0, 5))
def binary_crossentropy_loss(forward_fn, params, rng, batch_x, batch_y, is_training: bool = True):
    y_pred = forward_fn(params, rng, batch_x, is_training)
    return -jn.mean(y_pred * batch_y)
        
