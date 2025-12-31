import albumentations as A
from datasets.tf.afors import AFORSVideoDataGeneratorMultistream
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
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score


"""import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

export PATH=/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/:${PATH}
export LD_LIBRARY_PATH=/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/kientt/.conda/envs/tensor2/lib/:${LD_LIBRARY_PATH}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/
"""
# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def main():
    # configurations
    video_dir = '/AFOSR/afosr2022/data/'
    train_annotation_file = '/AFOSR/afosr2022/train.txt'
    test_annotation_file = '/AFOSR/afosr2022/val.txt'
    video_dir = '/AFOSR/afosr2022/data/'
    train_annotation_file = '/AFOSR/afosr2022/train_hieu.txt'
    test_annotation_file = '/AFOSR/afosr2022/val_hieu.txt'
    
    #os.environ["LD_LIBRARY_PATH"]="/home/kientt/.conda/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/"
    #tf.config.list_physical_devies('GPU')
    #exit()
    temporal_slice = 16  # number of sampled frames
    batch_size = 8
    data_format = 'channels_last'  # channels_first or [channels_last]

    model_id = 'a2'
    model_type = 'base'
    batch_size = 8
    num_frames = 8
    frame_stride = 10
    if model_id=='a5':
        resolution = 320 #112
    elif model_id=='a2':
        resolution= 224
    else:
        resolution = 172
    #resolution = 171 #112

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


    test_generator = AFORSVideoDataGeneratorMultistream(
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

    model.build([batch_size, temporal_slice, resolution, resolution, 3])
    model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)
    
    checkpoint_filepath = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/cp.cpkt'
    rgb_checkpoint_filepath = f'/AFOSR/afosr2022/movinet/savedmodel/cross_env/Hieu/{model_id}/{model_type}/checkpoint/cp.cpkt'
    of_checkpoint_filepath = f'/AFOSR/afosr2022/movinet/OF_checkpoint/{model_id}_cross_env_Hieu/{model_type}/checkpoint/cp.cpkt'
    cross_subj_rgb_checkpoint_filepath = f'/AFOSR/afosr2022/movinet/savedmodel/{model_id}/{model_type}/checkpoint/cp.cpkt'
    cross_subj_of_checkpoint_filepath = f'/AFOSR/afosr2022/movinet/OF_checkpoint/{model_id}_cross_subject/{model_type}/checkpoint/cp.cpkt'

    
    checkpoint_graph =  f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/cp.cpkt.meta'
    checkpoint_dir  = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/'
    h5_filepath = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/checkpoint/weight.h5'
    pb_filepath = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/model/'
    savedmodel_filepath = f'/AFOSR/afosr2022/movinet/ckpoint/{model_id}/{model_type}/model'
    

    checkpoint_dir  = f'/AFOSR/afosr2022/movinet/savedmodel/{model_id}/{model_type}/checkpoint/'
    checkpoint_filepath = f'/AFOSR/afosr2022/movinet/savedmodel/{model_id}/{model_type}/checkpoint/cp.cpkt'
    savedmodel_filepath = f'/AFOSR/afosr2022/movinet/savedmodel/{model_id}/{model_type}/model'
    
    
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    
    #latest = tf.train.latest_checkpoint(checkpoint_dir)

    #model.load_weights(latest)
    #model.save(h5_filepath)import matplotlib.pyplot as plt 


    #tf.saved_model.save(model,savedmodel_filepath)
    #model = tf.saved_model.load(savedmodel_filepath)
    #model = load_model(checkpoint_filepath)
    #model.summary()
    #exit()
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
    # Save model to SavedModel format
    #model = tf.saved_model.load(savedmodel_filepath)
    
    #Y_pred = model.predict_generator(test_generator)
    #Y_pred = model.predict_generator(test_generator)
    #
    
    model_rgb = load_model(cross_subj_rgb_checkpoint_filepath)
    model_rgb = Model(inputs=model_rgb.inputs,
                outputs=model_rgb.outputs,
                name='MoviNet_RGB')
    for layer in model_rgb.layers:
            layer._name = layer._name + '_RGB' # <===========
            
    out_rgb = model_rgb.output
    model_of = load_model(cross_subj_of_checkpoint_filepath)
    model_of = Model(inputs=model_of.inputs,
                outputs=model_of.outputs,
                name='MoviNet_OF')
    for layer in model_of.layers:
            layer._name = layer._name + '_OF' # <===========
    out_of = model_of.output
    #models = [model_rgb, model_of]
    #model_input = Input(shape=(image_width, image_height, 3))
    #model_outputs = [model(model_input) for model in models]
    model_outputs = [out_rgb, out_of]
    
    ensemble_output = Average()(model_outputs)
    #ensemble_model = Model(inputs=model_input, outputs=ensemble_output, name='ensemble')
    ensemble_model = Model(inputs=[model_rgb.input, model_of.input], outputs=ensemble_output, name='ensemble' )

    #Y_pred = model.predict(test_generator)
    Y_pred = ensemble_model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_class = np.argmax(test_generator.labels,axis=1)
    # function for scoring roc auc score for multi-class
    print(y_pred)
    print(Y_pred)
    print(y_class)
    y1_pred = softmax(Y_pred)
    print(y1_pred)
    print("="*50)
    
    
    ensemble_model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)
    acc = ensemble_model.evaluate(
        x=test_generator,
        steps=len(test_generator),
    )
    print(acc)
    #predict_result_filepath = f'/AFOSR/afosr2022/movinet/result/{model_id}_{model_type}_predict.csv'
    #real_result_filepath = f'/AFOSR/afosr2022/movinet/result/{model_id}_{model_type}_real.csv'
    result_filepath = f'/AFOSR/afosr2022/movinet/result/cross_subj_ensemble_{model_id}_{model_type}_report.csv'

    confusion_result_filepath = f'/AFOSR/afosr2022/movinet/result/cross_subj_ensemble_{model_id}_{model_type}_confusion.csv'
    plot_filepath = f'/AFOSR/afosr2022/movinet/result/cross_subj_ensemble_{model_id}_{model_type}_AUC_ROC.png'
    #y = np.concatenate((y_class, y_pred), axis=1)
    log = "True class:" + str(y_class.tolist()) + "\n"
    log +="Predicted class: " + str(y_pred.tolist()) + "\n"
    log +="Predict: " + str(acc) + "\n"
    #np.savetxt(predict_result_filepath,y_pred)
    #np.savetxt(real_result_filepath,y_class)
    cf = confusion_matrix(y_class, y_pred)
    log +="Confusion report: " + str(cf)
    np.savetxt(confusion_result_filepath, cf, delimiter=",")
    
    class_report = classification_report(y_class, y_pred, labels=[0, 1, 2, 3, 4,5,6,7,8,9,10,11])

    log +="Report:" + str(class_report) + "\n"
    
    target= ['G1', 'G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12']

    # set plot figure size
    fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
    
    # function for scoring roc auc score for multi-class
    print(y_pred)
    print(Y_pred)
    y1_pred = softmax(Y_pred)
    print(y1_pred)
    print("="*50)
    #print(y_class.shape())
    print(y_class.tolist())
    #print(Y_pred.shape())

    #print(y_class)
    def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        print(y_test)
        #y_pred = lb.transform(y_pred)
   
        for (idx, c_label) in enumerate(target):
            fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
            c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
        c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
        return roc_auc_score(y_test, y_pred, average=average)
    #print(Y_pred)
    mrauc =  multiclass_roc_auc_score(y_class, Y_pred)
    log +="ROC AUC score:" +  str(mrauc) + "\n"

    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.savefig(plot_filepath)
    f = open(result_filepath, 'w')
    f.write(log)
    f.close()

if __name__ == '__main__':
    main()
