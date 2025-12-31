### Data Generators for Tensorflow

#### Requirements:
- `numpy`
- `cv2`
- `albumentations`
- `tensorflow`

#### Instruction:
##### 0. Video samplers
Some video samplers are implemented in `datasets.utils.video_sampler` package.
These are used in initialization of data generator classes (see below).
Convention:
- `datasets.utils.video_sampler.RandomTemporalSegmentSampler` for training set.
- `datasets.utils.video_sampler.SystematicSampler` for testing set.
Reference: https://github.com/inspiros/multi_stream_videonet

##### 1. IPN RGB/Flow dataset
`datasets.tf.IPNFramesDataGenerator` \
__TODO__

##### 2. AFORS RGB/Flow dataset

Initialization of `datasets.tf.AFORSVideoDataGenerator` class:

```python
import albumentations as A

from datasets.tf.afors import AFORSVideoDataGenerator
from datasets.utils.video_sampler import *


train_generator = AFORSVideoDataGenerator(
    video_dir='/mnt/disk3/datasets/afors2022/data',
    annotation_file_path='/mnt/disk3/datasets/afors2022/train.txt',
    sampler=RandomTemporalSegmentSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
    data_format='channels_last',
    batch_size=16,
    shuffle=True,
)
test_generator = AFORSVideoDataGenerator(
    video_dir='/mnt/disk3/datasets/afors2022/data',
    annotation_file_path='/mnt/disk3/datasets/afors2022/val.txt',
    sampler=SystematicSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
    data_format='channels_last',
    batch_size=16,
    shuffle=False,
)
```

Usage:
```python
# compile a keras model then fit/evaluate with the above generators
model.fit(x=train_generator, steps_per_epoch=len(train_generator), epochs=30)
model.evaluate(x=test_generator, steps=len(test_generator))
```

#### Note:

- Most `keras`'s layers only support `data_format='channels_last'`.
