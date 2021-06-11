
# Flexible integration for any Python script
import wandb
import os
import haiku as hk
import optax
import jax
from jax import numpy as jnp
from jax import random
from typing import NamedTuple

# Number of classes in CIFAR10
CLASS_NUM = 10


def _forward(net, batch, is_training):
    """Forward application of the resnet."""
    images = batch['image'].reshape(-1, 32, 32, 3)
    return net(num_classes=CLASS_NUM)(images, is_training=is_training)


# Transform our forwards function into a pair of pure functions.
forward = hk.transform_with_state(_forward)


def make_optimizer(momentum=True):
    """SGD with momentum and a fixed lr."""
    if momentum:
        return optax.chain(
            optax.trace(decay=0.9, nesterov=False),  # momentum
            optax.scale(-1e-3))
    else:
        return optax.chain(
            optax.scale(-1e-3))


def l2_loss(params):
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


def loss_fn(params, state, net, batch, l2=True):
    """Computes a regularized loss for the given batch."""
    logits, state = forward.apply(
        params, state, None, net, batch, is_training=True)
    labels = jax.nn.one_hot(batch['label'], CLASS_NUM)
    logits = logits.reshape(len(labels), 1, CLASS_NUM)  # match labels shape
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
    acc = (labels.argmax(0, axis=2) == logits.argmax(0, axis=2)).mean()

    if l2:
        l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params)
                     if 'batchnorm' not in mod_name]
        loss = loss + 1e-4 * l2_loss(l2_params)
    return loss, (loss, state, acc)


loss_fn_grad = jax.grad(loss_fn, has_aux=True)


@jax.jit
def train_step(opt, train_state, batch, net, l2=True):
    """Applies an update to parameters and returns new state."""
    params, state, opt_state = train_state
    grads, (loss, new_state, acc) = loss_fn_grad(
        params, state, net, batch, l2=l2)

    # Compute and apply updates via our optimizer.
    updates, new_opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    train_state = TrainState(new_params, new_state, new_opt_state)
    return train_state, loss, acc


def initial_state(rng, opt, net, batch):
    """Computes the initial network state."""
    params, state = forward.init(rng, net, batch, is_training=True)
    opt_state = opt.init(params)
    return TrainState(params, state, opt_state)


def train(run_name, description, net, epochs, dataloader, dataloader_test, l2=True, momentum=True):
    opt = make_optimizer(momentum)

    rng = random.PRNGKey(0)

    train_state = initial_state(
        rng, opt, net, {"image": jnp.zeros((64, 32, 32, 3))})

    artifact = wandb.Artifact(run_name, type='model', description=description)

    os.mkdir(f'../run/{run_name}/')

    for e in range(epochs):
        losses = []
        accs = []

        for batch in dataloader:
            train_state, loss, acc = train_step(
                opt, train_state, batch, net, l2)
            losses.append(loss)
            accs.append(acc)
            wandb.log({'loss': loss, 'acc': acc})

        losses = []
        accs = []

        for batch in dataloader_test:
            params, state, opt_state = train_state

            _, (loss, state, acc) = loss_fn(params, state, net, batch, l2)
            losses.append(loss)
            accs.append(acc)

        wandb.log({'test_loss': jnp.mean(losses), 'test_acc': jnp.mean(accs)})

        if e % 50 == 0:
            pickle.dump(run_name, open(
                f"../run/{run_name}/{run_name}_{e}", 'wb'))

    artifact.add_dir(f'./run/{run_name}/')
    wandb.log_artifact(artifact)

    return train_state
