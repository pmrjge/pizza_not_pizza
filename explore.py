import logging
from turtle import forward
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

import optax

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            i = cv2.resize(img_rgb, (384, 384), interpolation=cv2.INTER_CUBIC)
            i[:, :, 0:3] = (i[:, :, 0:3]- 128.0) / 255.0
            images.append(i)

    return np.array(images)

pizza_imgs = load_images_from_folder('data/pizza')
not_pizza_imgs = load_images_from_folder('data/not_pizza')

np.random.shuffle(pizza_imgs)
np.random.shuffle(not_pizza_imgs)

N1 = pizza_imgs.shape[0]
N2 = not_pizza_imgs.shape[0]

limit1 = int(0.25 * N1)
limit2 = int(0.25 * N2)

test_pizza = pizza_imgs[:limit1]
train_pizza = pizza_imgs[limit1:]

test_not_pizza = not_pizza_imgs[:limit2]
train_not_pizza = not_pizza_imgs[limit2:]
        

def compute_sampler(pizzas, not_pizzas, batch_size, num_devices, *, rng_key):

        def generator():
            kk = batch_size // num_devices
            dd = batch_size // 2
            n = pizzas.shape[0]
            bs = batch_size
            pz = pizzas
            npz = not_pizzas
            key = rng_key
            while True:
                key, k1, k2 = jr.split(key, 3)
                perm1 = jax.random.choice(k1, n, shape=(dd,))
                perm2 = jax.random.choice(k2, n, shape=(dd,))
                p = pz[perm1]
                not_p = npz[perm2]
                xx = jn.vstack([p, not_p])
                a = jn.ones(shape=(dd,))
                b = jn.zeros(shape=(dd,))
                yy = jn.ones(shape=(bs,))
                yy = yy.at[dd:].set(0.0)
                yield xx.reshape(num_devices, kk, *xx.shape[1:]), yy.reshape(num_devices, kk, 1)

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


def build_estimator(dropout):
    def forward_fn(x: jn.ndarray, is_training: bool = True) -> jn.ndarray:
        convn = ConvNet(dropout)
        return convn(x, is_training=is_training)

    return forward_fn


@ft.partial(jax.jit, static_argnums=(0, 5))
def binary_crossentropy_loss(forward_fn, params, rng, batch_x, batch_y, is_training: bool = True):
    y_pred = forward_fn(params, rng, batch_x, is_training)
    return -jn.mean(y_pred * batch_y)


class GradientUpdater:
    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, master_rng, x):
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, x)
        opt_state = self._opt.init(params)
        return jn.array(0), out_rng, params, opt_state

    def update(self, num_steps, rng, params, opt_state, x:jn.ndarray, y: jn.ndarray):
        rng, new_rng = jax.random.split(rng)

        loss, grads = jax.value_and_grad(self._loss_fn)(params, rng, x, y)

        grads = jax.lax.pmean(grads, axis_name='i')

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, opt_state, metrics
        

def replicate(t, num_devices):
    return jax.tree_map(lambda x: jnp.array([x] * num_devices), t)


def main():
    max_steps = 150
    dropout = 0.6
    grad_clip_value = 1.0
    learning_rate = 0.001
    batch_size = 32

    num_devices = jax.local_device_count()

    print("Num devices :::::: ", num_devices)

    rng1, rng = jr.split(jax.random.PRNGKey(0))

    train_dataset = compute_sampler(train_pizza, train_not_pizza, batch_size=batch_size, num_devices=num_devices, rng_key = rng1)

    forward_fn = build_estimator(dropout)
    forward_fn = hk.transform(forward_fn)

    forward_apply = forward_fn.apply

    loss_fn = ft.partial(binary_crossentropy_loss, forward_apply, is_training=True)

    optimizer = optax.chain(
        optax.adaptive_grad_clip(grad_clip_value),
        #optax.sgd(learning_rate=learning_rate, momentum=0.95, nesterov=True),
        optax.radam(learning_rate=learning_rate)
    )

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    logging.info('Initializing parameters...')
    rng1, rng = jr.split(rng)
    a = next(train_dataset)
    w, z = a
    num_steps, rng, params, opt_state = updater.init(rng1, w[0, :, :, :])

    params_multi_device = params
    opt_state_multi_device = opt_state
    num_steps_replicated = replicate(num_steps, num_devices)
    rng_replicated = rng

    fn_update = jax.pmap(updater.update, axis_name="i", in_axes=(0, None, None, None, 0, 0), out_axes=(0, None, None, None, 0))

    logging.info('Starting training loop +++++++++++++++')
    for i, (w, z) in zip(range(max_steps), train_dataset):
        if (i + 1) % 10 == 0:
            logging.info(f'Step {i} of the computation of the forward-backward pass')
        num_steps_replicated, rng_replicated, params_multi_device, opt_state_multi_device, metrics = \
              fn_update(num_steps_replicated, rng_replicated, params_multi_device, opt_state_multi_device, w, z)

        if (i + 1) % 10 == 0:
            logging.info(f'Loss at step {i} :::::::::::: {metrics}')

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()