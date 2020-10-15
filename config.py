
class config(object):
    k = 2
    # 当γ增加的时候，a需要减小一点
    alpha = 0.5
    gamma = 2
#    data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v8/segmentation_data'
#     data_path_train = '/home/chenxiaojing/PycharmProjects/v6.1/segmentation_data'
#    data_path_train = '/home/chenxiaojing/PycharmProjects/v9/segmentation_data'
    data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v9/segmentation_data'
    # data_path_test = '../dataset/eye/images_cropped_divide_index_5cls/test'
    validation_split = .1
    shuffle_dataset = True
    random_seed = 42
    num_workers = 1
    lr = 0.0001

    lr_decay = 0.95
    weight_decay = 0.0
    use_gpu = True
    # model_path = "checkpoints/225epoch.pkl"
    save_path = "./checkpoints"
    date = '0903v1'
    num_epoch = 300
    num_classes = 1
    batch_size = 5
    save_epoch = 20
