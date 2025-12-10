from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "swin_moe"
config.resume = False
config.output = "/cis/home/knaraya4/FaceMoE/weights/swin4m_briar_exp_3_k_2_fc"
config.pretrained = "/cis/home/knaraya4/FaceMoE/pretrained_weights/swin4m_exp_3_k_2"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 1e-1
config.batch_size = 64
config.size = 120
config.gradient_acc = 24 # total batchsize is 256 * 12

config.optimizer = "adamw"
config.lr = 0.001
config.verbose = 2000
config.dali = False

config.num_experts = 3
config.k = 2

config.rec = "/cis/home/knaraya4/data/BRIAR/train_set_1/"
config.num_classes = 778
config.num_image = 301000
config.num_epoch = 20
config.warmup_epoch = 2
config.val_targets = []