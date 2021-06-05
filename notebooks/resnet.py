import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
class BasicBlock(hk.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = hk.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = hk.BatchNorm2d()
        self.conv2 = hk.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = hk.BatchNorm2d(planes)

        self.shortcut = hk.Sequential()

        # I didn't figure it out yet what self.expansion does in pytorch
        # thus I didn't translate it into jax...
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = hk.Sequential([
                hk.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                hk.BatchNorm2d(self.expansion*planes)
            ])

    def __call__(self, x):
        out = jax.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = jax.relu(out)
        return out


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
