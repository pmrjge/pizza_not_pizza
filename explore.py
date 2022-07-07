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
            i = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
            i = cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(i.astype(np.float32))

    return np.array(images)

pizza_imgs = load_images_from_folder('data/pizza')
not_pizza_imgs = load_images_from_folder('data/not_pizza')

np.random.shuffle(pizza_imgs)
np.random.shuffle(not_pizza_imgs)

N1 = pizza_imgs.shape[0]
N2 = not_pizza_imgs.shape[0]

limit1 = int(0.3 * N1)
limit2 = int(0.3 * N2)

test_pizza = pizza_imgs[:limit1]
train_pizza = pizza_imgs[limit1:]

train_pizza = jn.ones_like(train_pizza) * 0.0001

test_not_pizza = not_pizza_imgs[:limit2]
train_not_pizza = not_pizza_imgs[limit2:]
train_not_pizza = jn.ones_like(train_not_pizza) * 0.0003
        

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
                key, k1, k2, k3 = jr.split(key, 4)
                perm1 = jr.choice(k1, n, shape=(dd,))
                perm2 = jr.choice(k2, n, shape=(dd,))
                p = pz[perm1]
                not_p = npz[perm2]
                xx = jn.vstack([p, not_p])
                a = jn.ones(shape=(dd,), dtype=jn.int32)
                b = jn.zeros(shape=(dd,))
                yy = jn.ones(shape=(bs,))
                yy = yy.at[dd:].set(0)
                perm3 = jr.permutation(k3, bs)
                xx = xx[perm3]
                yy = yy[perm3]
                yield xx.reshape(num_devices, kk, *xx.shape[1:]), jn.array(yy.reshape(num_devices, kk, *yy.shape[1:]), dtype=jn.int32)

        return generator()


class ConvResBlock(hk.Module):
    def __init__(self, kernel_size, strides, dropout, name: Optional[str] = None):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout = dropout
        self.ks = 1.0
        self.ko = 1e-8

    def __call__(self, inputs, is_training=True):
        features = inputs.shape[-1]
        ks = self.ks
        ko = self.ko
        init_scale = hki.RandomUniform(ks)
        init_offset = hki.RandomUniform(ko)
        init = hki.VarianceScaling(scale=1.0, mode='fan_avg', distribution='truncated_normal')
        x = hk.Conv2D(output_channels=features, kernel_shape=self.kernel_size, stride=self.strides, padding="SAME", w_init=init)(inputs)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = jnn.relu(x)
        x = hk.Conv2D(output_channels= 2 * features, kernel_shape=self.kernel_size, stride=self.strides, padding="SAME", w_init=init)(x)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = jnn.relu(x)
        x = hk.Conv2D(output_channels=2 * features, kernel_shape=self.kernel_size, stride=self.strides, padding="SAME",w_init=init)(x)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = jnn.relu(x)
        x = hk.Conv2D(output_channels=features, kernel_shape=self.kernel_size, stride=self.strides, padding="SAME",w_init=init)(x)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = jnn.relu(x)
        return jnn.relu(x + inputs)


class ConvNet(hk.Module):
    def __init__(self, dropout, name: Optional[str]=None):
        super().__init__(name)
        self.dropout = dropout
        self.ks = 1.0
        self.ko = 1e-8

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0.0
        ks, ko = self.ks, self.ko
        init_scale = hki.RandomUniform(ks)
        init_offset = hki.RandomUniform(ko)
        init = hki.VarianceScaling(scale=1.0, mode='fan_avg', distribution='truncated_normal')
        x = hk.Conv2D(output_channels=32, kernel_shape=(3, 3), stride=(1, 1), padding="SAME", w_init=init)(inputs)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = jnn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding="SAME")(x)
        x = hk.Conv2D(output_channels=64, kernel_shape=(3, 3), stride=(1, 1), padding="SAME", w_init=init)(x)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = jnn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding="SAME")(x)
        x = hk.Conv2D(output_channels=128, kernel_shape=(3, 3), stride=(1, 1), padding="SAME", w_init=init)(x)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = jnn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding="SAME")(x)
        x = hk.Conv2D(output_channels=128, kernel_shape=(3, 3), stride=(1, 1), padding="SAME", w_init=init)(x)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = jnn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding="SAME")(x)
        x = hk.Conv2D(output_channels=128, kernel_shape=(3, 3), stride=(1, 1), padding="SAME", w_init=init)(x)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = jnn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding="SAME")(x)
        x = hk.Conv2D(output_channels=256, kernel_shape=(3, 3), stride=(1, 1), padding="SAME", w_init=init)(x)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = jnn.relu(x)

        for i in range(4):
            x = ConvResBlock((1 + i, 1 + i), (1, 1), dropout=dropout)(x, is_training)
        
        x = hk.Conv2D(output_channels=192, kernel_shape=(3, 3), stride=(1, 1), padding="SAME", w_init=init)(x)
        x = hk.BatchNorm(True, True, 0.95, scale_init=init_scale, offset_init=init_offset)(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding="SAME")(x)

        for i in range(4):
            x = ConvResBlock((2 + i, 2 + i), (1, 1), dropout=dropout)(x, is_training)

        x = hk.Flatten()(x)
        init = hki.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        x = hk.Linear(64, w_init=init)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        return hk.Linear(2, w_init=init)(x)
        


def build_estimator(dropout):
    def forward_fn(x: jn.ndarray, is_training: bool = True) -> jn.ndarray:
        #convn = ConvNet(dropout)
        convn = hk.nets.ResNet200(2,resnet_v2=True)
        return convn(x, is_training=is_training)

    return forward_fn


@ft.partial(jax.jit, static_argnums=(0, 6))
def binary_crossentropy_loss(forward_fn, params, state, rng, batch_x, batch_y, is_training: bool = True):
    logits, state = forward_fn(params, state, rng, batch_x, is_training)
    labels = jnn.one_hot(batch_y, 2)

    return (jn.mean(optax.softmax_cross_entropy(logits=logits, labels=labels)), state)


class GradientUpdater:
    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, master_rng, x):
        out_rng, init_rng = jax.random.split(master_rng)
        params, state = self._net_init(init_rng, x)
        opt_state = self._opt.init(params)
        return jn.array(0), out_rng, params, state, opt_state

    def update(self, num_steps, rng, params, state, opt_state, x:jn.ndarray, y: jn.ndarray):
        rng, new_rng = jax.random.split(rng)

        (loss, state), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(params, state, rng, x, y)

        grads = jax.lax.pmean(grads, axis_name='i')

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, state, opt_state, metrics
        

def replicate(t, num_devices):
    return jax.tree_map(lambda x: jn.array([x] * num_devices), t)


def main():
    max_steps = 666
    dropout = 0.5
    grad_clip_value = 1.0
    learning_rate = 0.001
    batch_size = 8

    num_devices = jax.local_device_count()

    print("Num devices :::::: ", num_devices)

    rng1, rng = jr.split(jax.random.PRNGKey(0))

    train_dataset = compute_sampler(train_pizza, train_not_pizza, batch_size=batch_size, num_devices=num_devices, rng_key = rng1)

    forward_fn = build_estimator(dropout)
    forward_fn = hk.transform_with_state(forward_fn)

    forward_apply = forward_fn.apply

    loss_fn = ft.partial(binary_crossentropy_loss, forward_apply, is_training=True)

    optimizer = optax.chain(
        
        optax.adaptive_grad_clip(grad_clip_value, eps=0.01),
        #optax.sgd(learning_rate=learning_rate, momentum=0.95, nesterov=True),
        optax.radam(learning_rate=learning_rate)
    )

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    logging.info('Initializing parameters...')
    rng1, rng = jr.split(rng)
    a = next(train_dataset)
    w, z = a
    num_steps, rng, params, state, opt_state = updater.init(rng1, w[0, :, :, :])

    params_multi_device = replicate(params, num_devices)
    opt_state_multi_device = replicate(opt_state, num_devices)
    num_steps_replicated = replicate(num_steps, num_devices)
    state_multi_device = replicate(state, num_devices)
    rng_replicated = rng

    fn_update = jax.pmap(updater.update, axis_name="i", in_axes=(0, None, 0, 0, 0, 0, 0), out_axes=(0, None, 0, 0, 0, 0))

    logging.info('Starting training loop +++++++++++++++')
    for i, (w, z) in zip(range(max_steps), train_dataset):
        if (i + 1) % 10 == 0:
            logging.info(f'Step {i} of the computation of the forward-backward pass')
        num_steps_replicated, rng_replicated, params_multi_device, state_multi_device,  opt_state_multi_device, metrics = \
              fn_update(num_steps_replicated, rng_replicated,  params_multi_device, state_multi_device, opt_state_multi_device, w, z)

        if (i + 1) % 10 == 0:
            logging.info(f'Loss at step {i} :::::::::::: {metrics}')

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()