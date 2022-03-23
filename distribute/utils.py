import tensorflow as tf

def setup_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print('strategy with tpu..')
    except:
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
        print('running on cpu instead..')
    num_replica = strategy.num_replicas_in_sync

    return strategy, num_replica

