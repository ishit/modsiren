exp_name: ms_ae_large
train_filenames: '/home/ishit/localImages/img_generalization/img_align_celeba/filenames.txt'
val_filenames: '/home/ishit/localImages/img_generalization/img_align_celeba/val_filenames.txt'
root_dir: '/home/ishit/localImages/img_generalization/img_align_celeba/'

model:
    hyper: False
    local: True # True, False
    layers: 4
    width: 512
    embedding: False # pe, ffn, False
    N_freqs: 256
    synthesis_activation: sine # relu, sine
    modulation_activation: relu # relu, false
    encoder: True # True, False

training:
    epochs: 10000
    lr: 0.0001
    batch_size: 8

image:
    size: 64
    channels: 3
    crop_size: 128
    
latent:
    dim: 256
    var_factor: 0.001

loss:
    reg_lam: 0.001

logging:
    image_step: 1000
    scalars_step: 100
    image_vis_count: 10

log_dir: logs
gpu: 0

