import albumentations as A
import tensorflow as tf
import os
from datasets.tf.afors import AFORSVideoDataGenerator
from datasets.utils.video_sampler import *

"""
export PATH=/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/:${PATH}
export LD_LIBRARY_PATH=/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/:${LD_LIBRARY_PATH}
"""
def main():
    # configurations
    video_dir = '../data'
    train_annotation_file = '../train.txt'
    test_annotation_file = '../val.txt'
    #os.environ["LD_LIBRARY_PATH"]="/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/"
    #tf.config.list_physical_devies('GPU')
    #exit()
    temporal_slice = 16  # number of sampled frames
    batch_size = 8
    data_format = 'channels_last'  # channels_first or [channels_last]

    # image transform
    transform = A.Compose([
        A.Resize(128, 171, always_apply=True),
        A.CenterCrop(112, 112, always_apply=True),
        A.ToFloat(),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    always_apply=True),
    ])

    train_generator = AFORSVideoDataGenerator(
        video_dir=video_dir,
        annotation_file_path=train_annotation_file,
        sampler=RandomTemporalSegmentSampler(n_frames=temporal_slice),
        to_rgb=True,
        transform=transform,
        use_albumentations=True,
        data_format=data_format,
        batch_size=batch_size,
        shuffle=True,
    )
    test_generator = AFORSVideoDataGenerator(
        video_dir=video_dir,
        annotation_file_path=test_annotation_file,
        sampler=SystematicSampler(n_frames=temporal_slice),
        to_rgb=True,
        transform=transform,
        use_albumentations=True,
        data_format=data_format,
        batch_size=batch_size,
        shuffle=False,
    )

    # properties
    print('\nDataset loaded:')
    print(f'Number of classes: {train_generator.n_classes}')
    print(f'Number of training instances: {len(train_generator.clips)}, '
          f'number of training batches: {len(train_generator)}')
    print(f'Number of testing instances: {len(test_generator.clips)}, '
          f'number of testing batches: {len(test_generator)}')
    print(f'Shape of train generator: {train_generator.shape}')
    # sample loop
    print('\nSample loop:')
    for batch_id, (X, y) in enumerate(train_generator):
        print(f'[{batch_id + 1}/{len(train_generator)}] X: {X.shape}, y: {y}')
        if batch_id == 3:
            break

    # sample training code with keras
    print('\nSample training:')
    tf.keras.backend.set_image_data_format(data_format)
    # create a simple foo model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=train_generator[0][0].shape[1:]),
        tf.keras.layers.Conv3D(48, kernel_size=(1, 5, 5)),
        tf.keras.layers.Conv3D(64, kernel_size=(3, 1, 1),
                               activation=tf.keras.activations.relu),
        tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=[_ for _ in range(1, len(x.shape))])),
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Nadam(),
                  metrics=[tf.keras.metrics.categorical_accuracy])
    model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=test_generator,
        validation_steps=len(test_generator),
    )
    model.evaluate(
        x=test_generator,
        steps=len(test_generator),
    )


if __name__ == '__main__':
    main()
