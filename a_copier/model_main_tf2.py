# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Creates and runs TF2 object detection models.

For local training/evaluation run:
PIPELINE_CONFIG_PATH=path/to/pipeline.config
MODEL_DIR=/tmp/model_outputs
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
"""
from absl import flags
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2
#import tensorboard.summary._tf as tf_summary
from object_detection.utils import config_util
from object_detection.builders import model_builder, optimizer_builder
import os


flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_bool('eval_on_train_data', False, 'Enable evaluating on train '
                  'data (only supported in distributed training).')
flags.DEFINE_integer('sample_1_of_n_eval_examples', None, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
                       'where event and checkpoint files will be written.')


flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')

flags.DEFINE_integer('eval_timeout', 3600, 'Number of seconds to wait for an'
                     'evaluation checkpoint before exiting.')

flags.DEFINE_bool('use_tpu', False, 'Whether the job is executing on a TPU.')
flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_integer(
    'num_workers', 1, 'When num_workers > 1, training uses '
    'MultiWorkerMirroredStrategy. When num_workers = 1 it uses '
    'MirroredStrategy.')
flags.DEFINE_integer(
    'checkpoint_every_n', 100, 'Integer defining how often we checkpoint.')
flags.DEFINE_boolean('record_summaries', True,
                     ('Whether or not to record summaries defined by the model'
                      ' or the training pipeline. This does not impact the'
                      ' summaries of the loss values which are always'
                      ' recorded.'))

FLAGS = flags.FLAGS


def main(_):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')
  tf.config.set_soft_device_placement(True)
  
  

  if FLAGS.checkpoint_dir:
    model_lib_v2.eval_continuously(
        pipeline_config_path=FLAGS.pipeline_config_path,
        model_dir=FLAGS.model_dir,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples),
        checkpoint_dir=FLAGS.checkpoint_dir,
        wait_interval=300, timeout=FLAGS.eval_timeout)
  else:
    if FLAGS.use_tpu:
      # TPU is automatically inferred if tpu_name is None and
      # we are running under cloud ai-platform.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name)
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif FLAGS.num_workers > 1:
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
      strategy = tf.compat.v2.distribute.MirroredStrategy()

    with strategy.scope():
        pipeline_config = config_util.get_configs_from_pipeline_file(FLAGS.pipeline_config_path)
        model_config = pipeline_config['model']
        train_config = pipeline_config['train_config']
        
        detection_model = model_builder.build(model_config=model_config, is_training=True)
        global_step = tf.compat.v2.Variable(0, trainable=False, dtype=tf.int64)
        optimizer, _ = optimizer_builder.build(train_config.optimizer, global_step=global_step)
        
        # Créer des writers de résumés pour l'entraînement et l'évaluation
        train_summary_writer = tf.summary.create_file_writer(FLAGS.model_dir + '/train')
        eval_summary_writer = tf.summary.create_file_writer(FLAGS.model_dir + '/eval')
        
        def write_eval_summary(eval_learning_rate, step):
            with eval_summary_writer.as_default():
                tf.summary.scalar('learning_rate/eval', eval_learning_rate, step=step)
                eval_summary_writer.flush()
        
        
        for step in range(0, FLAGS.num_train_steps, FLAGS.checkpoint_every_n):
            # Entraînement
            #print(f"Training from step {step} to {step + FLAGS.checkpoint_every_n}")
            model_lib_v2.train_loop(
                pipeline_config_path=FLAGS.pipeline_config_path,
                model_dir=FLAGS.model_dir,
                train_steps=step + FLAGS.checkpoint_every_n,
                use_tpu=FLAGS.use_tpu,
                checkpoint_every_n=FLAGS.checkpoint_every_n,
                record_summaries=FLAGS.record_summaries)
            
            # Enregistrer le taux d'apprentissage pour l'entraînement
            with train_summary_writer.as_default():
                
                learning_rate = optimizer._decayed_lr(tf.float32).numpy()  # Récupérer le taux d'apprentissage décadé
                tf.summary.scalar('learning_rate/train', learning_rate, step=step)
                train_summary_writer.flush()
                       
            # Évaluation périodique
            if (step + FLAGS.checkpoint_every_n) <= FLAGS.num_train_steps:
                # **Nouvelle boucle explicite pour l'évaluation**
                for eval_step in range(step, step + FLAGS.checkpoint_every_n, FLAGS.checkpoint_every_n):
                    # Recalculer et enregistrer le taux d'apprentissage pour l'évaluation
                    eval_learning_rate = optimizer._decayed_lr(tf.float32).numpy()
                    write_eval_summary(eval_learning_rate, eval_step)
                    
                    model_lib_v2.eval_continuously(
                        pipeline_config_path=FLAGS.pipeline_config_path,
                        model_dir=FLAGS.model_dir,
                        train_steps=step + FLAGS.checkpoint_every_n,
                        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
                        sample_1_of_n_eval_on_train_examples=FLAGS.sample_1_of_n_eval_on_train_examples,
                        checkpoint_dir=FLAGS.model_dir,
                        wait_interval=0,
                        timeout=0)
                           
                                    
    # Filtrer les métriques inutiles
    '''metrics_to_exclude = [
        'DetectionBoxes_Recall/AR@1',
        'DetectionBoxes_Recall/AR@10',
        'DetectionBoxes_Recall/AR@100',
        'DetectionBoxes_Recall/AR@100 (small)',
        'DetectionBoxes_Recall/AR@100 (medium)',
        'DetectionBoxes_Recall/AR@100 (large)',
        'DetectionBoxes_Precision/mAP',
        'DetectionBoxes_Precision/mAP@.50IOU',
        'DetectionBoxes_Precision/mAP@.75IOU',
        'DetectionBoxes_Precision/mAP (small)',
        'DetectionBoxes_Precision/mAP (medium)',
        'DetectionBoxes_Precision/mAP (large)'
        ]
    for metric in metrics_to_exclude:
        tf.compat.v1.summary.scalar(metric, None)'''
        
                    
                
                
                #with tf.summary.create_file_writer(FLAGS.model_dir).as_default():
                    # Ajouter les métriques de validation ici
                    #tf.summary.scalar('learning_rate/eval', learning_rate, step=step + FLAGS.checkpoint_every_n)
                    #tf.summary.scalar('validation/mAP', .0, step=step + FLAGS.checkpoint_every_n)
                    #tf.summary.scalar('validation/AR', .0, step=step + FLAGS.checkpoint_every_n)
      

if __name__ == '__main__':
  tf.compat.v1.app.run()
