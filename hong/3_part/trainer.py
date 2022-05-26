import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from pickle import TRUE
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice
from sklearn.model_selection import KFold
from torchvision import transforms
from tqdm import tqdm


# trainer 함수
from pickle import TRUE
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice
from sklearn.model_selection import KFold
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 



def train_model(model_class, DriveDataset, csv_file, transforms, BATCH_SIZE, epochs, lr, is_1d = None, 
                reshape_size = None, fold_num=4, CODER=None, 
                bagging_num=1,
                savepath="model.pth"):
    # bagging_num 만큼 모델 학습을 반복 수행한다
    for b in range(bagging_num):
        print("bagging num : ", b)

        # 교차 검증
        kfold = KFold(n_splits=fold_num, shuffle=True)
        for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(csv_file),1):
            print(f'[fold: {fold_index}]')
            torch.cuda.empty_cache()
            
            # kfold dataset 구성
            train_answer = csv_file.iloc[trn_idx]
            test_answer  = csv_file.iloc[val_idx]

            #Dataset 정의
            train_dataset = DriveDataset(train_answer, transforms)
            valid_dataset = DriveDataset(test_answer, transforms)

            #DataLoader 정의
            train_data_loader = DataLoader(
                train_dataset,
                batch_size = BATCH_SIZE,
                shuffle = True,
                num_workers = 8,
            )
            valid_data_loader = DataLoader(
                valid_dataset,
                batch_size = int(BATCH_SIZE / 2),
                shuffle = False,
                num_workers = 4,
            )

            # model setup
            model = model_class()
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(),lr = lr)
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                             step_size = 5,
            #                                             gamma = 0.75)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, 
                                                                            eta_min=0.001, last_epoch=-1)                                       
            criterion = torch.nn.MSELoss()


            # train 시작
            early_stopping = EarlyStopping(patience=3, verbose = True)

            for epoch in range(epochs):
                model.train()
                print("-" * 50)
                # trainloader를 통해 batch_size 만큼의 훈련 데이터를 읽어온다
                with tqdm(train_data_loader,total=train_data_loader.__len__(), unit="batch") as train_bar:
                    for batch_idx, batch_data in enumerate(train_bar):
                        train_bar.set_description(f"Train Epoch {epoch}")
                        images, steering = batch_data['image'], batch_data['steering']
                        images, steering = Variable(images.cuda()), Variable(steering.cuda())
                        
                        # steering = batch_data['steering'] ,batch_data['throttle'], batch_data['brake'], batch_data['speed']
                        # steering=  Variable(steering.cuda()),Variable(throttle.cuda()), Variable(brake.cuda()), Variable(speed.cuda())
                        
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(True):
                            # center, left, right중 뭐가 들어가는거지? 랜덤?
                            probs  = model()
                            loss = criterion(probs, steering)
                            loss.backward()
                            optimizer.step()

                        # 현재 progress bar에 현재 미니배치의 loss 결과 출력
                        train_bar.set_postfix(train_loss= loss.item())
                                                                                                   
                # epoch마다 valid 계산
                valid_losses = []
                model.eval()
                with tqdm(valid_data_loader,total=valid_data_loader.__len__(), unit="batch") as valid_bar:
                    for batch_idx, batch_data in enumerate(valid_bar):
                        valid_bar.set_description(f"Valid Epoch {epoch}")
                        images, steering = batch_data['image'], batch_data['steering']
                        images, steering = Variable(images.cuda()), Variable(steering.cuda())
                        
                        # steering = batch_data['steering'] ,batch_data['throttle'], batch_data['brake'], batch_data['speed']
                        # steering=  Variable(steering.cuda()),Variable(throttle.cuda()), Variable(brake.cuda()), Variable(speed.cuda())steering, throttle, brake, speed =  Variable(steering.cuda()), Variable(throttle.cuda()), Variable(brake.cuda()), Variable(speed.cuda())
                        
                        with torch.no_grad():
                            probs  = model()
                            valid_loss = criterion(probs, steering)


                        valid_losses.append(valid_loss.item())
                        valid_bar.set_postfix(valid_loss = valid_loss.item())
                
                # Learning rate 조절
                lr_scheduler.step()  
                early_stopping(np.average(valid_losses), model)
                
                if early_stopping.early_stop or epoch == epochs:
                    print("Early stopping")
                    create_directory("model")
                    torch.save(model.state_dict(), "model/" + savepath)
                    break


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss