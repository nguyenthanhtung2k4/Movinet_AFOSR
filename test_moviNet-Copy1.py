import albumentations as A
from datasets.tf.afors import AFORSVideoDataGenerator
from datasets.utils.video_sampler import *
from six.moves import urllib
import tensorflow as tf
#os.environ["LD_LIBRARY_PATH"]="/home/kientt/.conda/envs/tensor2/lib/"
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from official.projects.movinet.configs import movinet as movinet_configs
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_layers
from official.projects.movinet.modeling import movinet_model

from datetime import datetime
from sklearn.metrics import confusion_matrix

"""
export PATH=/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/:${PATH}
export LD_LIBRARY_PATH=/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/kientt/.conda/envs/tensor2/lib/:${LD_LIBRARY_PATH}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/
"""
#def get_flops(model_h5_path,movinet):
def get_flops(test_generator):

    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        

    with graph.as_default():
        with session.as_default():
            #model = tf.keras.models.load_model(model_h5_path,custom_objects={'MovinetClassifier':movinet})
            model_id = 'a5'
            model_type = 'base'
            temporal_slice = 16  # number of sampled frames
            batch_size = 8
            data_format = 'channels_last'  # channels_first or [channels_last]

            batch_size = 8
            num_frames = 8
            frame_stride = 10
            resolution = 171 #112
            print("Create new backbone=========================================")
            loss_obj = tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True,
                    label_smoothing=0.1)

            metrics = [
                    tf.keras.metrics.TopKCategoricalAccuracy(
                        k=1, name='top_1', dtype=tf.float32),
                    tf.keras.metrics.TopKCategoricalAccuracy(
                        k=5, name='top_5', dtype=tf.float32)    ]

            initial_learning_rate = 0.01
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate, decay_steps=1,
                )
            optimizer = tf.keras.optimizers.RMSprop(
                    learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)
                #optimizer = tf.keras.optimizers.Nadam()

            backbone = movinet.Movinet(model_id=model_id, causal=True)

            model = movinet_model.MovinetClassifier(
                backbone=backbone,
                num_classes=12)

            model.call = tf.function(model.call)
            #model.run_eagerly = True

            model.build([batch_size, temporal_slice, resolution, resolution, 3])
            model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)
            #model.run_eagerly = True

            checkpoint_filepath = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/cp.cpkt'
            checkpoint_graph =  f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/cp.cpkt.meta'
            checkpoint_dir  = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/'
            h5_filepath = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/weight.h5'

            #checkpoint_filepath = f'/AFOSR/afosr2022/movinet/logs/{model_id}/checkpoint/'
            #checkpoint_dir  = f'/AFOSR/afosr2022/movinet/logs/{model_id}/'


            latest = tf.train.latest_checkpoint(checkpoint_dir)

            model.load_weights(latest)             
            #Y_pred = model.predict_generator(test_generator)
            #y_pred = np.argmax(Y_pred, axis=1)
            acc = model.evaluate(
                x=test_generator,
                steps=len(test_generator),

            )
            print(acc)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
        
            return flops.total_float_ops
        
def main():
    # configurations
    video_dir = '/AFOSR/afosr2022/data/'
    train_annotation_file = '/AFOSR/afosr2022/train.txt'
    test_annotation_file = '/AFOSR/afosr2022/val.txt'
    #os.environ["LD_LIBRARY_PATH"]="/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/"
    #tf.config.list_physical_devies('GPU')
    #exit()
    temporal_slice = 16  # number of sampled frames
    batch_size = 8
    data_format = 'channels_last'  # channels_first or [channels_last]

    batch_size = 8
    num_frames = 8
    frame_stride = 10
    resolution = 171 #112

    # image transform
    transform = A.Compose([
        #A.Resize(128, 171, always_apply=True),
        A.Resize(resolution, resolution, always_apply=True),
        #A.CenterCrop(112, 112, always_apply=True),
        A.ToFloat(),
        #A.Normalize(mean=[0.485, 0.456, 0.406],
        #            std=[0.229, 0.224, 0.225],
        #            always_apply=True),
    ])


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
    print(f'Number of classes: {test_generator.n_classes}')
    
    print(f'Number of testing instances: {len(test_generator.clips)}, '
          f'number of testing batches: {len(test_generator)}')
    #print(f'Shape of train generator: {train_generator[0]}')

    """
    model_id = 'a5'
    model_type = 'base'
    print("Create new backbone=========================================")
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.1)

    metrics = [
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=1, name='top_1', dtype=tf.float32),
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=5, name='top_5', dtype=tf.float32)    ]

    initial_learning_rate = 0.01
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate, decay_steps=1,
        )
    optimizer = tf.keras.optimizers.RMSprop(
            learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)
        #optimizer = tf.keras.optimizers.Nadam()
       
    backbone = movinet.Movinet(model_id=model_id, causal=True)

    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=12)
    
    model.call = tf.function(model.call)
    #model.run_eagerly = True

    model.build([batch_size, temporal_slice, resolution, resolution, 3])
    model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics,run_eagerly=True)
    #model.run_eagerly = True

    checkpoint_filepath = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/cp.cpkt'
    checkpoint_graph =  f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/cp.cpkt.meta'
    checkpoint_dir  = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/'
    h5_filepath = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/weight.h5'

    #checkpoint_filepath = f'/AFOSR/afosr2022/movinet/logs/{model_id}/checkpoint/'
    #checkpoint_dir  = f'/AFOSR/afosr2022/movinet/logs/{model_id}/'
    
  
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    model.load_weights(latest)
   
    model.save(h5_filepath)
    """
    #tf.train.import_meta_graph(checkpoint_graph)
    #saver = tf.train.Saver()
    #with tf.Session() as sess:
    #    with tf.Session() as sess:
            # The session is binding to the default global graph
    #        tf.profiler.profile(
    #            sess.graph,
    #            options=tf.profiler.ProfileOptionBuilder.float_operation())
    #        parameters = tf.profiler.profile(sess.graph,
    #                                         options=tf.profiler.ProfileOptionBuilder
    #                                         .trainable_variables_parameter())
    #        print ('total parameters: {}'.format(parameters.total_parameters))
    #model.summary()
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    
    get_flops(test_generator)

    exit()
    
    Y_pred = model.predict_generator(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    acc = model.evaluate(
        x=test_generator,
        steps=len(test_generator),

    )
    print(acc)
    y_class = np.argmax(test_generator.labels,axis=1)
    predict_result_filepath = f'/AFOSR/afosr2022/movinet/result/{model_id}_{model_type}.csv'
    
    cf = confusion_matrix(y_class, y_pred)
    np.savetxt(predict_result_filepath, cf, delimiter=",")

    print("===================")
    print(cf)
    print("===================")
    #print(y_class)
    #print(loss)

if __name__ == '__main__':
    main()
