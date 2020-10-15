import torch
import torch.utils.data as DATA
from PIL import Image
import numpy as np
from models.attention_model import Unet_att
from utils.load import Glaucoma_Dataset
from utils.criterion import dice_coeff3

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter



def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn

if __name__ == '__main__':

    # net = UNet(n_channels=3, n_classes=1)
    # 1.导入网络模型
    net2 = Unet_att(3, 1)

    # 2.

    # 3.加载训练好的模型
    net2.load_state_dict(torch.load('/home/chenxiaojing/PycharmProjects/B4_attUnetv4/checkpoints/0901v1/CP300_k1.pth'))

    # 4. 放入gpu
    net2.cuda()

    # 默认操作
    net2.eval()

    # 5. 开始测试、
    # 5.1 封装数据集
    #dataset_test = Glaucoma_Dataset(root='/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v3', phase='test')
    dataset_test = Glaucoma_Dataset(root='/home/chenxiaojing/PycharmProjects/v9/segmentation_data/image_data', phase='test')
    # 5.2 加载数据集
    test_loader = DATA.DataLoader(
        dataset_test,
        batch_size=1,  # 测试时，batchsize是1
        drop_last=False,  # 是false
        shuffle=False,  # 是false
        num_workers=1)

    # 开始测试（算分）
    score_ave = 0

    for step, [img, label] in enumerate(test_loader):
        # 数据放入 cuda()
        img = img.cuda()
        label = label.float().cuda()

        # 图像放入网络 的到输出
        output = net2(img)
        # 归一化
        output = normPRED(output)
        # 二值化
        output = (output > 0.5).float()
        print(output)

        # 计算 dice
        score = dice_coeff3(output, label)

        # 打印score，显示图像
        print('step{}'.format(step), ':', score.item())

        # 计算总分
        score_ave += score.item()


        # 打印图像并保存
        plot_img_and_mask(step, img, output, label)
        plt.show()
        break

    # 打印平均分
    print("score_ave:", score_ave / (step + 1))




