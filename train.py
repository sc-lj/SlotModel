import os
import shutil
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
# from utils.Callback import EMACallBack
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from utils.utils import statistics_text_length,update_arguments
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(2023)
def parser_args():
    parser = argparse.ArgumentParser(description='各个模型公共参数')
    parser.add_argument('--model_type', default="prgc", type=str, help='specify max sample triples', choices=["prgc"])
    parser.add_argument('--pretrain_path', type=str, default="rtb3", help='specify the model name')
    parser.add_argument('--data_dir', type=str, default="data", help='specify the train path')
    parser.add_argument('--lr', default=5e-04, type=float, help='定义非bert模块的学习率')
    parser.add_argument('--bert_lr', default=5e-05, type=float, help='定义bert模块的学习率')
    parser.add_argument('--epoch', default=30, type=int, help='specify the epoch size')
    parser.add_argument('--batch_size', default=24, type=int, help='specify the batch size')
    parser.add_argument('--output_path', default="event_extract", type=str, help='将每轮的验证结果保存的路径')
    parser.add_argument('--float16', default=False,type=bool, help='是否采用浮点16进行半精度计算')

    # 不同scheduler的参数
    parser.add_argument('--decay_rate', default=0.999, type=float, help='StepLR scheduler 相关参数')
    parser.add_argument('--decay_steps', default=100,type=int, help='StepLR scheduler 相关参数')

    parser.add_argument('--T_mult', default=1.0, type=float, help='CosineAnnealingWarmRestarts scheduler 相关参数')
    parser.add_argument('--rewarm_epoch_num', default=2,type=int, help='StepLR scheduler 相关参数')

    args = parser.parse_args()
    # 根据超参数文件更新参数
    config_file = os.path.join("config","{}.yaml".format(args.model_type))
    if os.path.exists(config_file):
        with open(config_file,'r') as f:
            config = yaml.load(f,Loader=yaml.Loader)
        args = update_arguments(args,config)

    return args


def main():
    args = parser_args()
    tb_logger =  TensorBoardLogger(save_dir="lightning_logs",name=args.model_type)

    from PRGC import PRGCPytochLighting, PRGCDataset, collate_fn
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_path, cache_dir="./bertbaseuncased")
    filename = os.path.join(args.data_dir, "train_data.csv")
    max_length = statistics_text_length(filename, tokenizer)
    print("最大文本长度为:", max_length)
    args.max_length = max_length

    train_dataset = PRGCDataset(args, is_training=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn,
                                    batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = PRGCDataset(args, is_training=False)
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)

    intent_number = train_dataset.intent_number
    args.intent_number = intent_number
    args.seq_tag_size = len(train_dataset.slot2id)
    model = PRGCPytochLighting(args)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=8,
        verbose=True,
        monitor='n_f1',  # 监控val_acc指标
        mode='max',
        save_last=True,
        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"),
        every_n_epochs=1,
        # filename = "{epoch:02d}{f1:.3f}{acc:.3f}{recall:.3f}",
        filename="{epoch:02d}{int_acc:.3f}{n_f1:.3f}{n_acc:.3f}{n_rec:.3f}{g_f1:.3f}{g_acc:.3f}{g_rec:.3f}",
        # filename = "{epoch:02d}{loss:.3f}",
    )

    early_stopping_callback = EarlyStopping(monitor="n_f1",
                                            patience=8,
                                            mode="max",
                                            )
    # ema_callback = EMACallBack()
    swa_callback = StochasticWeightAveraging()

    trainer = pl.Trainer(max_epochs=args.epoch,
                         gpus=[0],
                         logger=tb_logger,
                        #  accelerator = 'dp',
                        #  plugins=DDPPlugin(find_unused_parameters=True),
                         check_val_every_n_epoch=1,  # 每多少epoch执行一次validation
                         callbacks=[checkpoint_callback,
                                    early_stopping_callback,
                                    ],
                         accumulate_grad_batches=1,  # 累计梯度计算
                        #  precision=16, # 半精度训练
                         gradient_clip_val=3,  # 梯度剪裁,梯度范数阈值
                         progress_bar_refresh_rate=5,  # 进度条默认每几个step更新一次
                        # O0：纯FP32训练,
                        # O1：混合精度训练，根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
                        # O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算
                        # O3：纯FP16训练，很不稳定，但是可以作为speed的baseline；
                         amp_level="O1",
                         move_metrics_to_cpu=True,
                         amp_backend="apex",
                        #  resume_from_checkpoint =""
                         )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()

