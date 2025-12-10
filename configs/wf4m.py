from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "swin_moe"
config.output = "/cis/home/knaraya4/FaceMoE/pretrained_weights/swin4m_exp_3_k_2"
config.pretrained = "/cis/home/knaraya4/FaceMoE/pretrained_weights/swin4m_exp_3_k_2"
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = True
config.weight_decay = 5e-2
config.batch_size = 128
config.gradient_acc = 1
config.size = 120
config.optimizer = "adamw"
config.lr = 0.001
config.verbose = 2000
config.dali = False
config.resume = False
config.save_all_states = True
config.use_moe = True

config.num_experts = 3
config.k = 2

config.rec = "/cis/home/knaraya4/data/WebFace4M/"
config.num_classes = 205990
config.num_image = 4235242
config.num_epoch = 26
config.warmup_epoch = 1
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
