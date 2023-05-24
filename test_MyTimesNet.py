import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np
import pytorch_lightning as pl
from MyTimesNet import MyPL_TimesNet
from LoadStockData import OneStockAllDayData



if __name__ == '__main__':
    import argparse
    import random
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
















    from LoadStockData import OneStockAllDayData

    torch.set_default_dtype(torch.float32)
    from pytorch_lightning.callbacks import ModelCheckpoint
    # fp = DataStorageBase('/mnt/weka/home/shannon_public/StockData/1000ms_aligned/')
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    # model = MyLightingModule.load_from_checkpoint(PATH)

    # load dataset
    # mnistdata = AllStockAllDayData1(fp, path = 'stock_date_id_test.txt')
    # mnistdata = AllStockAllDayData('stock_date_test.txt', 'stock_num_test.txt')
    mnistdata = OneStockAllDayData('/home/dongyikai/stock_date_test.txt', '/home/dongyikai/stock_num_test.txt', window_size = args.seq_len, pred_len = args.pred_len, label_len = args.label_len, batch_size = 1024)


    my_model = MyPL_TimesNet(args)
    checkpoint_path = r'/home/dongyikai/Time-Series-Library/lightning_logs/version_57537/checkpoints/last.ckpt'
    checkpoint = torch.load(checkpoint_path)
    
    my_model.load_state_dict(checkpoint['state_dict'])
    mnistdata.setup('test')
    test_data = mnistdata.test_dataloader()
    ans1 = []
    ans2 = []
    ans3 = []
    ans_gt1 = []
    ans_gt2 = []
    ans_gt3 = []
    for x, y, y_hat in test_data:
        y1 = my_model(x)
        print(y1[:10, :])
        if y1.shape[0] != 1024:break
        ans1.append(y1[:, 0].detach().numpy().reshape(-1))
        ans2.append(y1[:, 1].detach().numpy().reshape(-1))
        ans3.append(y1[:, 2].detach().numpy().reshape(-1))
        ans_gt1.append(y_hat[:, 0].detach().numpy().reshape(-1))
        ans_gt2.append(y_hat[:, 1].detach().numpy().reshape(-1))
        ans_gt3.append(y_hat[:, 2].detach().numpy().reshape(-1))
        print(np.corrcoef(y1[:, 0].detach().numpy().reshape(-1), y_hat[:, 0].detach().numpy().reshape(-1)))
        print(np.corrcoef(y1[:, 1].detach().numpy().reshape(-1), y_hat[:, 1].detach().numpy().reshape(-1)))
        print(np.corrcoef(y1[:, 2].detach().numpy().reshape(-1), y_hat[:, 2].detach().numpy().reshape(-1)))
    print(np.corrcoef(np.array(ans1).reshape(-1), np.array(ans_gt1).reshape(-1)))
    print(np.corrcoef(np.array(ans2).reshape(-1), np.array(ans_gt2).reshape(-1)))
    print(np.corrcoef(np.array(ans3).reshape(-1), np.array(ans_gt3).reshape(-1)))


    # my_model.load_from_checkpoint(r'D:\vscode_python\lightning_logs\version_116\checkpoints\epoch=19-step=34380.ckpt', encoder = Encoder(), decoder = Decoder())
    # trainer = pl.Trainer(max_epochs=15, callbacks=[checkpoint_callback], accelerator='gpu')
    # trainer.fit(my_model, mnistdata)

    # trainer.test(my_model, datamodule=mnistdata, ckpt_path=r'/home/dongyikai/Time-Series-Library/lightning_logs/version_56269/checkpoints/last-v1.ckpt')
    # trainer.predict(my_model, mnistdata)
    # autoencoder = LitAutoEncoder(Encoder(), Decoder())
    # trainer = pl.Trainer(max_epochs=2, default_root_dir=r"D:\vscode_python", profiler="advanced")
    # trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # trainer.test(my_model, mnistdata)
