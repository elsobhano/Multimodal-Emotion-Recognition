# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16

# CLIP-Bert Configs
max_temporal_position_embeddings = 100
backbone_channel_in_size = 2048
max_grid_row_position_embeddings = 100
max_grid_col_position_embeddings = 100
attention_probs_dropout_prob = 0.1
hidden_act = "relu"
hidden_dropout_prob = 0.1
hidden_size= 768
initializer_range= 0.02
intermediate_size= 3072
layer_norm_eps= 1e-12
max_position_embeddings= 512
model_type= "bert"
num_attention_heads= 12
num_hidden_layers= 12
pad_token_id= 0
type_vocab_size= 2
vocab_size= 30522
cls_hidden_scale= 2  # mlp intermediate layer hidden size scaler
classifier="mlp"  # classfied type, [mlp, linear]
num_labels=3  # number of labels for classifier output
loss_type="ce"  # [BCE, CE, KLDivLoss] only used when num_labels > 1