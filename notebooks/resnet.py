import types
from typing import Mapping, Optional, Sequence, Union, Any

from haiku._src import basic
from haiku._src import batch_norm
from haiku._src import conv
from haiku._src import module
from haiku._src import pool
import jax
import jax.numpy as jnp

# If forking replace this block with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.Module = module.Module
hk.BatchNorm = batch_norm.BatchNorm
hk.Conv2D = conv.Conv2D
hk.Linear = basic.Linear
hk.max_pool = pool.max_pool
del basic, batch_norm, conv, module, pool
FloatStrOrBool = Union[str, float, bool]


class BlockV1(hk.Module):
  """ResNet V1 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bn_config: Mapping[str, FloatStrOrBool],
      bottleneck: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection

    bn_config = dict(bn_config)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)
    bn_config.setdefault("decay_rate", 0.999)

    if self.use_projection:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv")

      self.proj_batchnorm = hk.BatchNorm(name="shortcut_batchnorm", **bn_config)

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        with_bias=False,
        padding="SAME",
        name="conv_0")
    bn_0 = hk.BatchNorm(name="batchnorm_0", **bn_config)

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        with_bias=False,
        padding="SAME",
        name="conv_1")

    bn_1 = hk.BatchNorm(name="batchnorm_1", **bn_config)
    layers = ((conv_0, bn_0), (conv_1, bn_1))

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          with_bias=False,
          padding="SAME",
          name="conv_2")

      bn_2 = hk.BatchNorm(name="batchnorm_2", scale_init=jnp.zeros, **bn_config)
      layers = layers + ((conv_2, bn_2),)

    self.layers = layers

  def __call__(self, inputs, is_training, test_local_stats):
    out = shortcut = inputs

    if self.use_projection:
      shortcut = self.proj_conv(shortcut)
      shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)

    for i, (conv_i, bn_i) in enumerate(self.layers):
      out = conv_i(out)
      out = bn_i(out, is_training, test_local_stats)
      if i < len(self.layers) - 1:  # Don't apply relu on last layer
        out = jax.nn.relu(out)

    return jax.nn.relu(out + shortcut)


class BlockGroup(hk.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      bn_config: Mapping[str, FloatStrOrBool],
      resnet_v2: bool,
      bottleneck: bool,
      use_projection: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    block_cls = BlockV2 if resnet_v2 else BlockV1

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(
          block_cls(channels=channels,
                    stride=(1 if i else stride),
                    use_projection=(i == 0 and use_projection),
                    bottleneck=bottleneck,
                    bn_config=bn_config,
                    name="block_%d" % (i)))

  def __call__(self, inputs, is_training, test_local_stats):
    out = inputs
    for block in self.blocks:
      out = block(out, is_training, test_local_stats)
    return out


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(hk.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        self.conv1 = hk.Conv2d(64, kernel_shape=3, stride=1,
                               padding='SAME', bias=False)  # OK
        self.bn1 = hk.BatchNorm2d()

        # I'll leave these lines commented just because _make_layer is not working
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = hk.Linear(512*block.expansion, num_classes)

    # Let's pretend this works for a moment...
    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return hk.Sequential(*layers)

    def __call__(self, x):
        out = jax.nn.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = hk.avg_pool(out, 4) # This is probably wrong.
        out = hk.Reshape(output_shape=(-1, out.size(0)))(out)
        out = self.linear(out)
        return out


def ResNet18(x):
    return ResNet(BasicBlock, [2, 2, 2, 2], 10)(x)
