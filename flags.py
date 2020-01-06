from absl import flags

# TODO remove unnecessary flags

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.005, help="Learning rate value.")
flags.DEFINE_integer("epochs", default=400, help="Training epochs.")
flags.DEFINE_integer("batch_size", default=1, help="Batch size.")
flags.DEFINE_integer("val_batch_size", default=2, help="Validation/Test Batch size.")
flags.DEFINE_integer("embed_dim", default=50, help="TransE embed dimension.")
flags.DEFINE_float("margin", default=1.0, help="Margin ranking loss margin size.")
flags.DEFINE_integer("norm", default=1, help="Margin ranking loss norm type (1 or 2).")
flags.DEFINE_string("checkpoint_dir", default="./checkpoints", help="Directory to save checkpoint files.")
flags.DEFINE_string("checkpoint_path", default="", help="Checkpoint file to load (by default train from scratch).")
flags.DEFINE_string("tensorboard_path", default="./runs", help="Path to tensorboard log directory")
flags.DEFINE_bool("use_gpu", default=False, help="Enable gpu usage.")