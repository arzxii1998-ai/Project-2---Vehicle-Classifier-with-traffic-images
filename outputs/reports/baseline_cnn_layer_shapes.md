| Layer | Type | Output shape (C×H×W) | Params |
|---|---|---|---:|
| input | - | 3×288×224 | 0 |
| features.0.0 | Conv2d | 32×288×224 | 864 |
| features.0.1 | BatchNorm2d | 32×288×224 | 64 |
| features.0.2 | ReLU | 32×288×224 | 0 |
| features.0.3 | MaxPool2d | 32×144×112 | 0 |
| features.1.0 | Conv2d | 64×144×112 | 18,432 |
| features.1.1 | BatchNorm2d | 64×144×112 | 128 |
| features.1.2 | ReLU | 64×144×112 | 0 |
| features.1.3 | MaxPool2d | 64×72×56 | 0 |
| features.2.0 | Conv2d | 128×72×56 | 73,728 |
| features.2.1 | BatchNorm2d | 128×72×56 | 256 |
| features.2.2 | ReLU | 128×72×56 | 0 |
| features.2.3 | MaxPool2d | 128×36×28 | 0 |
| features.3.0 | Conv2d | 256×36×28 | 294,912 |
| features.3.1 | BatchNorm2d | 256×36×28 | 512 |
| features.3.2 | ReLU | 256×36×28 | 0 |
| features.3.3 | MaxPool2d | 256×18×14 | 0 |
| global_pool | AdaptiveAvgPool2d | 256×1×1 | 0 |
| classifier.0 | Flatten | 256 | 0 |
| classifier.1 | Dropout | 256 | 0 |
| classifier.2 | Linear | 8 | 2,056 |

Trainable parameters: 390,952 | Total parameters: 390,952