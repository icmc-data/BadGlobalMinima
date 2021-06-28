
# Flexible integration for any Python script
from tqdm.auto import tqdm
import tree
import wandb
import os
import haiku as hk
import optax
import jax
from jax import numpy as jnp
from jax import random
from typing import NamedTuple
from functools import partial
import pickle

# Number of classes in CIFAR10
CLASS_NUM = 10

def get_forward_fn(net):
    def _forward(batch, is_training):
        """Forward application of the resnet."""
        images = batch[0].reshape(-1, 32, 32, 3)
        return net(num_classes=CLASS_NUM)(images, is_training=is_training)
    return _forward

def make_optimizer(momentum=True, schedule_fn = lambda x:-1e-3):
    """SGD with momentum and a fixed lr."""
    if momentum:
        return optax.chain(
            optax.trace(decay=0.9, nesterov=False),  # momentum
            optax.scale_by_schedule(schedule_fn))
    else:
        return optax.chain(
            optax.scale(schedule_fn))


def l2_loss(params):
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)

def lp_path_norm(forward, train_state, p=2, input_size=[3, 32, 32]):
    weights = train_state.weights
    state = train_state.state
    pw_model = jax.tree_map(lambda w: jnp.power(jnp.abs(w), p), weights)
    data_ones = jnp.ones(input_size)
    return (forward.apply(pw_model, state, data_ones).sum() ** (1 / p ))

class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


def loss_fn(forward, params, state, batch, l2=True):
    """Computes a regularized loss for the given batch."""
    logits, state = forward.apply(
        params, state, None, batch, is_training=True)
    labels = jax.nn.one_hot(batch[1], CLASS_NUM)
    logits = logits.reshape(len(labels), CLASS_NUM)  # match labels shape
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
    acc = (labels.argmax(1) == logits.argmax(1)).mean()

    if l2:
        l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params)
                     if 'batchnorm' not in mod_name]
        loss = loss + 5e-4 * l2_loss(l2_params)
    return loss, (loss, state, acc)


loss_fn_grad = jax.grad(loss_fn, has_aux=True, argnums = 1)


@partial(jax.jit, static_argnums = (0,1,4))
def train_step(forward, opt, train_state, batch, l2=True):
    """Applies an update to parameters and returns new state."""
    params, state, opt_state = train_state
    grads, (loss, new_state, acc) = loss_fn_grad(forward, params, state, batch, l2=l2)

    # Compute and apply updates via our optimizer.
    updates, new_opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    train_state = TrainState(new_params, new_state, new_opt_state)
    return train_state, loss, acc


def initial_state(forward, rng, opt, batch):
    """Computes the initial network state."""
    params, state = forward.init(rng, batch, is_training=True)
    opt_state = opt.init(params)
    return TrainState(params, state, opt_state)


def train(net, epochs, dataloader, dataloader_test, schedule_fn = lambda x: -1e-3, l2=True, 
        momentum=True, seed = 0, weights_file="", run_weights_name=None):
    # Transform our forwards function into a pair of pure functions.
    forward = hk.transform_with_state(get_forward_fn(net))

    opt = make_optimizer(momentum, schedule_fn)

    rng = random.PRNGKey(seed)

    train_state = initial_state(forward, rng, opt, (jnp.zeros((64, 32, 32, 3)),))

    if run_weights_name is not None:
        temp_params = pickle.load(open(weights_file, 'rb'))
        train_state = TrainState(temp_params, train_state.state, train_state.opt_state)

    if not os.path.exists(f'./run/'):
        os.mkdir(f'./run/')

    with open(f"./run/weights_init.pkl", 'wb') as f:
        pickle.dump(train_state, f)
    wandb.save(f"./run/weights_init.pkl", )

    for e in range(epochs):
        print('Epoch', e+1)
        losses = []
        accs = []

        for batch in tqdm(dataloader):
            batch[0] = jnp.array(batch[0])
            batch[1] = jnp.array(batch[1])
            train_state, loss, acc = train_step(forward, opt, train_state, batch, l2)
            losses.append(loss)
            accs.append(acc)
            wandb.log({'loss': float(loss), 'acc': float(acc), 'lr' : float(schedule_fn(train_state.opt_state[1].count))})

        if dataloader_test != None:
            losses = []
            accs = []

            for batch in dataloader_test:
                batch[0] = jnp.array(batch[0])
                batch[1] = jnp.array(batch[1])
                params, state, opt_state = train_state

                _, (loss, state, acc) = loss_fn(forward, params, state, batch, l2)
                losses.append(loss)
                accs.append(acc)

            wandb.log({'test_loss': float(jnp.mean(jnp.array(losses))), 'test_acc': float(jnp.mean(jnp.array(accs)))})

    with open(f"./run/weights_final.pkl", 'wb') as f:
        pickle.dump(train_state.params, f)
    wandb.save(f"./run/weights_final.pkl", )

    return train_state
