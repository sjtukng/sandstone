from keras.utils import np_utils
import keras
from nn.PetroNetModel import get_model
from data import load_data
import matplotlib.pyplot as plt


#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def train_nn():
    model = get_model()

    datas1, labels = load_data(64, 64)
    datas2, labels = load_data(98, 98)
    labels = np_utils.to_categorical(labels, 3)
    bs = 100
    ep = 50
    for i in range(ep):
        print ("iteration ", str(i + 1))
        model.fit(datas1, labels, batch_size = bs, epochs = 1)
        model.fit(datas2, labels, batch_size = bs, epochs = 1)

    # model.save('petro_model(bs=' + str(bs) + ',ep=' + str(ep) +').h5')
    model.save('model11_roi.h5')
    # 绘制acc-loss曲线


train_nn()
