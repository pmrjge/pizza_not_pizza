import jax
import jax.numpy as jn
import jax.random as jr
import jax.nn as jnn

import numpy as np

import cv2
import os

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

