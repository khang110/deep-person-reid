import torch
import torchreid

#-----------------------------------------------------
# Step 1: Build the Data Manager
#-----------------------------------------------------
# Here we assume you want to evaluate on 'market1501'.
# The root directory should contain the dataset folder.
# For Market1501, it usually has:
# market1501/
#     bounding_box_train/
#     bounding_box_test/
#     query/
datamanager = torchreid.data.ImageDataManager(
    root='dataset',       # Path to the folder containing the dataset
    sources='market1501',           # Source dataset for training (if needed)
    targets='market1501',           # Target dataset for evaluation
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms='random_flip',       # Apply basic transformations
    combineall=False,
    workers=4
)

#-----------------------------------------------------
# Step 2: Build and Load the Model
#-----------------------------------------------------
# Suppose your model is built as follows:
model = torchreid.models.build_model(
    name='osnet_x1_0',                         # Example model architecture
    num_classes=datamanager.num_train_pids,    # Number of training person IDs
    loss='softmax',
    pretrained=True                            # Load imagenet pretraining
)

model = model.cuda()

# Load the trained weights (model.pth is your checkpoint)
torchreid.utils.load_pretrained_weights(model, 'log/osnet_x1_0_market1501_softmax_cosinelr/model/model.pth.tar-20')

# You don't necessarily need an optimizer or scheduler if you're just evaluating,
# but Torchreid's engines typically require them. Here we set them up anyway.
optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)
scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

#-----------------------------------------------------
# Step 3: Evaluate the Model
#-----------------------------------------------------
# Torchreid provides built-in engines for training and evaluation.
# If you just want to evaluate (test-only), you can use an Image-based engine.
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer,
    scheduler=scheduler,
    label_smooth=True
)

# Run evaluation on the target dataset. Setting test_only=True
# means it won't do any training, just evaluation.
engine.run(
    test_only=True,
    dist_metric='euclidean',    # distance metric for evaluation
    normalize_feature=True,     # normalize features before computing distance
    ranks=[1, 5, 10, 20],       # which Rank-k accuracies to print
    rerank=False                # whether to use re-ranking
)
