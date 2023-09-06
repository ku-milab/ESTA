import os
import random

import matplotlib.pyplot as plt
import torch
import os
import util
import argparse
import numpy as np
from dataset import *
from main_model import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import glob




def writelog(file, line):
    file.write(line + '\n')
    print(line)




def step(model, criterion, inputs, label, device='cpu', optimizer=None):
    if optimizer is None: model.eval()
    else: model.train()



    # run model
    logit, st_attention = model(inputs.to(device))
    loss = criterion(logit, label.to(device))


    # optimize model
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss, logit, st_attention


def training_function(args):
    # make directories
    os.makedirs(os.path.join(args.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(args.targetdir, 'summary'), exist_ok=True)

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)



    # define dataset
    dataset = DatasetHCPTask(args.sourcedir, roi=args.roi, crop_length=args.standardized_length, k_fold=args.k_fold)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)



    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(args.targetdir, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(args.targetdir, 'checkpoint.pth'), map_location=device)
    else:
        checkpoint = {
            'fold': 0,
            'loss' : 0,
            'epoch': 0,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'scheduler': None}


    # start experiment
    for k in range(checkpoint['fold'], args.k_fold):
        # make directories per fold
        os.makedirs(os.path.join(args.targetdir, 'model', str(k)), exist_ok=True)
        os.makedirs(os.path.join(args.targetdir, 'model_weights', str(k)), exist_ok=True)

        # set dataloader
        dataset.set_fold(k, train=True)

        # define model
        batch = args.batch_size
        time = dataset.standardized_length
        roi = dataset.num_nodes
        n_labels= dataset.num_classes
        spa_hidden_dim = 64
        hidden_dim = 128
        model = STEAMNetwork(roi, hidden_dim, spa_hidden_dim, time, n_labels)
        model.to(device)



        if checkpoint['model_state_dict'] is not None: model.load_state_dict(checkpoint['model_state_dict'])
        criterion = torch.nn.CrossEntropyLoss()


        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr_ae, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs_ae, steps_per_epoch=len(dataloader), pct_start=0.2, div_factor=args.max_lr/args.lr_ae, final_div_factor=1000)
        if checkpoint['optimizer_state_dict'] is not None: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])


        # define logging objects
        summary_writer = SummaryWriter(os.path.join(args.targetdir, 'summary', str(k), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join(args.targetdir, 'summary', str(k), 'val'), )


        best_score = 0.0


        # start training
        for epoch in range(checkpoint['epoch'], args.epochs_ae):


            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0
            acc_accumulate = 0.0


            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):
                # process input data
                inputs = x['timeseries']
                label = x['label']


                loss, logit, attention = step(
                    model = model,
                    criterion = criterion,
                    inputs = inputs,
                    label = label,
                    device=device,
                    optimizer=optimizer
                )



                loss_accumulate += loss.detach().cpu().numpy()
                pred = logit.argmax(1)
                acc_accumulate += ( torch.sum(pred.cpu() == label).item()  / batch )


            if scheduler is not None:
                scheduler.step()


            total_loss = loss_accumulate / len(dataloader)
            total_acc = acc_accumulate / len(dataloader)


            # summarize results
            summary_writer.add_scalar('training loss', total_loss, epoch)
            summary_writer.add_scalar("training acc", total_acc, epoch)
            print()
            print('loss for epoch {} is : {}'.format(epoch, total_loss))
            print('acc for epoch {} is : {}'.format(epoch, total_acc))



            # eval
            dataset.set_fold(k, train=False)
            predict_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)
            for i, x in enumerate(dataloader):
                with torch.no_grad():
                    # process input data
                    inputs = x['timeseries']
                    label = x['label']


                    val_loss, val_logit, val_attention = step(
                        model=model,
                        criterion=criterion,
                        inputs = inputs,
                        label=label,
                        device=device,
                        optimizer=None,
                    )

                    pred = val_logit.argmax(1).cpu().numpy()
                    predict_all = np.append(predict_all, pred)
                    labels_all = np.append(labels_all, label)



            val_acc = metrics.accuracy_score(labels_all, predict_all)
            print('val_acc for epoch {} is : {}'.format(epoch, val_acc))
            summary_writer_val.add_scalar('val acc', val_acc, epoch)

            if best_score < val_acc:
                best_score = val_acc
                best_epoch = epoch

                torch.save({
                    'fold': k,
                    'loss': total_loss,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.targetdir, 'model_weights', str(k), 'checkpoint_epoch_{}.pth'.format(epoch)))


                f = open(os.path.join(args.targetdir, 'model', str(k), 'best_acc.log'), 'a')
                writelog(f, 'best_acc: %.4f for epoch: %d' % (best_score, best_epoch))
                f.close()
                print()
                print('-----------------------------------------------------------------')

        # finalize fold
        torch.save(model.state_dict(), os.path.join(args.targetdir, 'model', str(k), 'model.pth'))
        checkpoint.update({'loss' : 0, 'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})


    summary_writer.close()
    summary_writer_val.close()


def tt_function(args):
    os.makedirs(os.path.join(args.targetdir, 'attention'), exist_ok=True)



    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # define dataset
    dataset = DatasetHCPTask_test(args.sourcedir, roi=args.roi, crop_length=args.standardized_length, k_fold=args.k_fold)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)


    for k in range(args.k_fold):
        os.makedirs(os.path.join(args.targetdir, 'attention', str(k)), exist_ok=True)


        # define model
        batch = args.batch_size
        time = dataset.standardized_length
        roi = dataset.num_nodes
        n_labels = dataset.num_classes
        spa_hidden_dim = 64
        hidden_dim = 128
        model = STEAMNetwork(roi, hidden_dim, spa_hidden_dim, time, n_labels)

        # load model
        path = os.path.join(args.targetdir, 'model_weights', str(k))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getmtime)[-1]
        print(full_path)
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device)


        criterion = torch.nn.CrossEntropyLoss()

        # eval
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        fold_st_attention = {'temporal-result': [], 'spatial-result': []}


        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():

                inputs = x['timeseries']
                label = x['label']

                test_loss, test_logit, test_attention = step(
                    model=model,
                    criterion=criterion,
                    inputs=inputs,
                    label=label,
                    device=device,
                    optimizer=None,
                )

                pred = test_logit.argmax(1).cpu().numpy()
                predict_all = np.append(predict_all, pred)
                labels_all = np.append(labels_all, label)

                fold_st_attention['temporal-result'].append(np.stack(test_attention['temporal-result'], axis=0))
                fold_st_attention['spatial-result'].append(np.stack(test_attention['spatial-result'], axis=0))




        test_acc = metrics.accuracy_score(labels_all, predict_all)
        f1_score = metrics.f1_score(labels_all, predict_all, average='macro')

        f = open(os.path.join(args.targetdir, 'model', str(k), 'test_acc.log'), 'a')
        writelog(f, 'test_acc: %.4f' % (test_acc))
        f.close()
        print('test_acc is : {}'.format(test_acc))

        print('f1_score is : {}'.format(f1_score))
        f = open(os.path.join(args.targetdir, 'model', str(k), 'f1_score.log'), 'a')
        writelog(f, 'f1_score: %.4f' % (f1_score))
        f.close()
        print('---------------------------')

        for key, value in fold_st_attention.items():
            os.makedirs(os.path.join(args.targetdir, 'attention', str(k), key), exist_ok=True)
            for idx, task in enumerate(dataset.task_list):
                np.save(os.path.join(args.targetdir, 'attention', str(k), key, f'{task}.npy'), np.concatenate([v for (v, l) in zip(value, labels_all) if l == idx]))








if __name__=='__main__':
    # parse options and make directories
    def get_arguments():
        parser = argparse.ArgumentParser(description='EAM-NETWORK')
        parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
        parser.add_argument('-n', '--exp_name', type=str, default='experiment_1')
        parser.add_argument('-k', '--k_fold', type=int, default=5)
        parser.add_argument('-ds', '--sourcedir', type=str, default='../data')
        parser.add_argument('-dt', '--targetdir', type=str, default='../result')



        # model args
        parser.add_argument("--model_name", default="STEAM", help="model_name")
        parser.add_argument('--roi', type=str, default='aal', choices=['scahefer', 'aal', 'destrieux', 'harvard_oxford'])
        parser.add_argument('--standardized_length', type=int, default=176)

        # training args

        parser.add_argument("--batch_size", default=8, type=int, help="batch size")
        parser.add_argument("--epochs_ae", type=int, default=50, help="Epochs number of training", )
        parser.add_argument("--lr_ae", type=float, default=1e-5, help="Learning rate of training", )
        parser.add_argument('--max_lr', type=float, default=3e-5)
        parser.add_argument("--weight_decay", type=float, default=5e-6)



        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument("--weights", default=True, help='pre-trained STEAM weights')

        return parser


    parser = get_arguments()
    args = parser.parse_args()
    args.targetdir = os.path.join(args.targetdir, args.exp_name)
    print(args.exp_name)

    training_function(args)

    if args.weights is not None:
        tt_function(args)  #test function
