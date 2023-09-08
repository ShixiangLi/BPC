"""
Author: Benny
Date: Nov 2019
"""
import json

from common import setup_logging
from dataset import ModelNetDataLoader
import numpy as np
import os
import torch
import torch.nn as nn
from datetime import datetime
import logging
from tqdm import tqdm

import provider

from model_bin import PointTransformerCls


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main():
    # --------------------------------------------读取训练配置文件-------------------------------------------------
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # --------------------------------------------配置log路径-------------------------------------------------
    save_path = os.path.join(config['results_dir'], str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'config.txt'), 'w') as args_file:
        args_file.write(str(datetime.now()) + '\n\n')
        for args_n, args_v in config.items():
            args_v = '' if not args_v and not isinstance(args_v, int) else args_v
            args_file.write(str(args_n) + ':  ' + str(args_v) + '\n')

    setup_logging(os.path.join(save_path, 'logging.log'), filemode='a')

    # --------------------------------------------准备数据集-------------------------------------------------------
    logging.info('Load dataset ...')
    DATA_PATH = config['data_path']

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=config['num_point'], split='train',
                                       normal_channel=config['normal'])
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=config['num_point'], split='test',
                                      normal_channel=config['normal'])
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=config['batch_size'], shuffle=True,
                                                  num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=config['batch_size'], shuffle=False,
                                                 num_workers=4)

    # --------------------------------------------准备模型-------------------------------------------------------
    config['num_class'] = 40
    config['input_dim'] = 6 if config['normal'] else 3

    model = PointTransformerCls(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    try:
        checkpoint = torch.load('best/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info('Use pretrain model')
    except:
        logging.info('No existing model, starting training from scratch...')
        start_epoch = 0


    # --------------------------------------------准备优化器-------------------------------------------------------
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': config['learning_rate']}],
                                    lr=config['learning_rate'],
                                    momentum=config['momentum'],
                                    weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': config['learning_rate']}],
                                     lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])
    else:
        logging.error("Optimizer '%s' not defined.", config['optimizer'])
        raise ValueError('Optimizer %s is not supported.' % config['optimizer'])

    # --------------------------------------------准备学习率调整器-------------------------------------------------------
    if config['lr_type'] == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'] - config['warm_up'] * 4,
                                                                  eta_min=0,
                                                                  last_epoch=config['start_epoch'])
    elif config['lr_type'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config['lr_decay_step'], gamma=0.1,
                                                            last_epoch=-1)
    elif config['lr_type'] == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (
                1.0 - (epoch - config['warm_up'] * 4) / (config['epochs'] - config['warm_up'] * 4)), last_epoch=-1)
    else:
        logging.error("lr_type '%s' not defined.", config['lr_type'])
        raise ValueError('lr_type %s is not supported.' % config['lr_type'])

    # --------------------------------------------准备损失函数-------------------------------------------------------
    criterion = nn.CrossEntropyLoss().cuda()
    criterion = criterion.type(config['type'])

    # --------------------------------------------设置随机种子-------------------------------------------------------
    # np.random.seed(42)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # --------------------------------------------开始训练-------------------------------------------------------
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

    logging.info('Start training...')
    for epoch in range(start_epoch, config['epoch']):
        logging.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, config['epoch']))

        model.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = model(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        logging.info('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(model.eval(), testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logging.info('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            logging.info('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logging.info('Save model...')
                savepath = './checkpoints/best_model.pth'
                logging.info('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logging.info('End of training...')


if __name__ == '__main__':
    main()
