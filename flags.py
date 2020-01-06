from absl import flags

# TODO remove unnecessary flags

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.005, help="Learning rate value.")
flags.DEFINE_integer("epochs", default=400, help="Training epochs.")
flags.DEFINE_integer("batch_size", default=1, help="Batch size.")
flags.DEFINE_integer("val_batch_size", default=2, help="Validation/Test Batch size.")
flags.DEFINE_integer("hidden_dim", default=256, help="Node feature dimension.")
flags.DEFINE_integer("num_layers", default=4, help="Number of GeniePathLayers to stack.")
flags.DEFINE_string("checkpoint_dir", default="./checkpoints", help="Directory to save checkpoint files.")
flags.DEFINE_string("checkpoint_path", default="", help="Checkpoint file to load (by default train from scratch).")
flags.DEFINE_string("tensorboard_path", default="./runs", help="Path to tensorboard log directory")
flags.DEFINE_bool("use_gpu", default=False, help="Enable gpu usage.")