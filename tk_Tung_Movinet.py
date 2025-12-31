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

def get_flops(models, test_generator):

    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        

    with graph.as_default():
        with session.as_default():
            #model = tf.keras.models.load_model(model_h5_path,custom_objects={'MovinetClassifier':movinet})
            model=models
            Y_pred = model.predict_generator(test_generator)
            y_pred = np.argmax(Y_pred, axis=1)
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
            print(flops.total_float_ops)
            return flops.total_float_ops
        
def main():
    # configurations
    video_dir = '/AFOSR/afosr2022/data/'
    train_annotation_file = '/AFOSR/afosr2022/train_hieu.txt'
    test_annotation_file = '/AFOSR/afosr2022/val_hieu.txt'
    #os.environ["LD_LIBRARY_PATH"]="/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/"
    #tf.config.list_physical_devies('GPU')
    #exit()
    temporal_slice = 16  # number of sampled frames
    data_format = 'channels_last'  # channels_first or [channels_last]

    model_id = 'a5'
    model_type = 'base'
    
    batch_size = 2
    num_frames = 8
    frame_stride = 10
    if model_id=='a5':
        resolution = 320 #112
    elif model_id=='a2':
        resolution= 224
    else:
        resolution = 172
    resolution=172
    #a0=94.87
    #a2_224 = 92.73
    #a2-172 = 
    num_epochs = 40

     # Create log directory
    logdir = f'/AFOSR/afosr2022/movinet/savedmodel/cross_env/Hieu/{model_id}/{model_type}/{resolution}/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = f'/AFOSR/afosr2022/movinet/savedmodel/cross_env/Hieu/{model_id}/{model_type}/{resolution}/checkpoint/cp.cpkt'
    savedmodel_filepath = f'/AFOSR/afosr2022/movinet/savedmodel/cross_env/Hieu/{model_id}/{model_type}/{resolution}/model'
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

    train_steps = len(train_generator.clips) // batch_size
    total_train_steps = train_steps * num_epochs
    test_steps = len(test_generator.clips) // batch_size
    
    # properties
    print('\nDataset loaded:')
    print(f'Number of classes: {train_generator.n_classes}')
    print(f'Number of training instances: {len(train_generator.clips)}, '
          f'number of training batches: {len(train_generator)}')
    print(f'Number of testing instances: {len(test_generator.clips)}, '
          f'number of testing batches: {len(test_generator)}')
    #print(f'Shape of train generator: {train_generator[0]}')

    # sample loop
    print('\nSample loop:')
    for batch_id, (X, y) in enumerate(train_generator):
        #print(f'[{batch_id + 1}/{len(train_generator)}] X: {X.shape}, y: {y}')
        if batch_id == 3:
            break

    tf.keras.backend.clear_session()
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    print(gpus)
    gpus = ["GPU:2"]
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        print("Create new backbone=========================================")
        backbone = movinet.Movinet(model_id=model_id, causal=True)
        model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=600)
        model.build([batch_size, temporal_slice, resolution, resolution, 3])
        
        #print("Download pretrained network=================================")
        # Load pretrained weights from TF Hub
        movinet_hub_url = f'https://tfhub.dev/tensorflow/movinet/{model_id}/{model_type}/kinetics-600/classification/3'
        movinet_hub_model = hub.KerasLayer(movinet_hub_url, trainable=True)
        pretrained_weights = {w.name: w for w in movinet_hub_model.weights}
        model_weights = {w.name: w for w in model.weights}
        #print(model.weights)
        #print("len pretrained weights:", len(pretrained_weights))
        #print("len model_weights:",len(model_weights))
        #print(pretrained_weights.keys())
        #print("============================================")
       
        #print(model_weights.keys())
        #print("============================================")
        #for i in range(334):
        #    print(i, model_weights[i],pretrained_weights[i])

        #for name in pretrained_weights:
        for name in model_weights:
            replaced_name = name
            #print(name)
            replaced_name = name.replace("block","b")
            replaced_name = replaced_name.replace("_layer","/l")
            model_weights[name].assign(pretrained_weights[replaced_name])
        #print("After assigning")
        #print(model.weights)
        # Wrap the backbone with a new classifier to create a new classifier head
        # with num_classes outputs
        model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=12)
        #tf.compat.v1.disable_eager_execution()
        #print("[INFO] training with {} GPUs...".format(G))
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        #with tf.device("/cpu:0"):
        model.build([batch_size, temporal_slice, resolution, resolution, 3])
        #model = multi_gpu_model(model, gpus=G)

        # Freeze all layers except for the final classifier head
        for layer in model.layers[:-1]:
            layer.trainable = True #False
        model.layers[-1].trainable = True
        #print("After recreating new classifier")
        #print(model.weights)
        #exit()
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
            initial_learning_rate, decay_steps=total_train_steps,
        )
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)
        #optimizer = tf.keras.optimizers.Nadam()
       

        # Define a TensorBoard callback. Use the log_dir parameter
        # to specify the path to the directory where you want to save the
        # log files to be parsed by TensorBoard.
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir, histogram_freq = 1)

        #file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        callbacks = [
            tensorboard_callback, model_checkpoint_callback
        ]
        """
        results = model.fit(
            x=train_dataset,
            validation_data=test_dataset,
            epochs=num_epochs,
            steps_per_epoch=train_steps,
            validation_steps=test_steps,
            callbacks=callbacks,
            validation_freq=1,
            verbose=1)
        """

        model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)
        results =model.fit(
            x=train_generator,
            steps_per_epoch=len(train_generator),
            epochs=num_epochs,
            validation_data=test_generator,
            validation_steps=len(test_generator),
            callbacks=callbacks,
            validation_freq=1,
            verbose=1
        )
        model.evaluate(
            x=test_generator,
            steps=len(test_generator),
            verbose=1,
        )
    tf.saved_model.save(model,savedmodel_filepath)
    Y_pred = model.predict_generator(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    y_class = np.argmax(test_generator.labels,axis=1)
    predict_result_filepath = f'/AFOSR/afosr2022/movinet/result/cross_env/Hieu/{model_id}_{model_type}_{resolution}.csv'
    model_summary_filepath = f'/AFOSR/afosr2022/movinet/result/cross_env/Hieu/{model_id}_{model_type}_{resolution}_flops.txt'

    cf = confusion_matrix(y_class, y_pred)
    np.savetxt(predict_result_filepath, cf, delimiter=",")

    print("===================")
    print(cf)
    print("===================")
    print("Calculate FLOPS")
    #flops = get_flops(model, test_generator)
    #np.savetxt(model_summary_filepath,model_summary_filepath)
    
if __name__ == '__main__':
    main()
