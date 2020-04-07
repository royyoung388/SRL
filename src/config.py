PAD = "<pad>"
UNK = "<unk>"
# pad id can be automatically detected, so you don't need to modify it.
WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_PAD_ID = 0

# parameter
label_smoothing = 0.1

residual_dropout = 0.2
attention_dropout = 0.1
relu_dropout = 0.1

# train
# subprocesses to use for data loading. = cpu counts
num_workers = 16
batch_size = 2048
# batch_size = 4

# dimensions
# feature_dim * 2 = model_dim
feature_dim = 100
model_dim = 200
filter_dim = 400
# feature_dim = 4
# model_dim = 8
# filter_dim = 16

head_num = 8
layer_num = 4
# head_num = 2
# layer_num = 1

epoch = 1000
clipping = 3
warmup_step = 4000
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-8

# output
plot_step = 5
plot_epoch = 10
