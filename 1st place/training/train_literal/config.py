import torch
class CFG:
    DIM = (512,512)
    NUM_WORKERS = 0
    TRAIN_BATCH_SIZE = 6
    VALID_BATCH_SIZE = 4
    EPOCHS = 20
    SEED = 42
    device = torch.device('cuda')
    model_name = 'tf_efficientnet_b2_ns'
    loss_module = 'arcface' #'cosface' #'adacos'
    s = 30.0
    m = 0.5 
    ls_eps = 0.0
    n_fold = 5
    easy_margin = False
    scheduler_params = {
          "lr_start": 1e-5,
          "lr_max": 1e-3 ,
          "lr_min": 1e-6,
          "lr_ramp_ep": 15,
          "lr_sus_ep": 0,
          "lr_decay": 0.8,
      }
    model_params = {
      'n_classes':788,
      'model_name':'tf_efficientnet_b5_ns',
      'use_fc':False,
      'fc_dim':2048,
      'dropout':0.0,
      'loss_module':loss_module,
      's':30.0,
      'margin':0.50,
      'ls_eps':0.0,
      'theta_zero':0.785,
      'pretrained':True
    }
    model_params2 = {
      'n_classes':788,
      'model_name':'efficientnetv2_rw_m',
      'use_fc':False,
      'fc_dim':2048,
      'dropout':0.0,
      'loss_module':loss_module,
      's':30.0,
      'margin':0.50,
      'ls_eps':0.0,
      'theta_zero':0.785,
      'pretrained':True
    }