import os
import torch
from dataset.Sim_dataset import get_dual_data_loader
from trainer.addSPE_trainer import DualTrainer
import argparse
from model.mymodel import initialize_weights, DualLSTMClassifier
from utils.read_data import read_data
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import utils
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import glob


def load_or_generate_data(mat_folder_path, excel_file_path, output_file_path, if_load):
    if if_load == 0:
        data = torch.load(output_file_path)
        all_data = data['all_data']
        all_labels = data['all_labels']
        all_ids = data['all_ids']
    else:
        read_data(mat_folder_path, excel_file_path, output_file_path)
        data = torch.load(output_file_path)
        all_data = data['all_data']
        all_labels = data['all_labels']
        all_ids = data['all_ids']

    return all_data, all_labels, all_ids


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
if __name__ == "__main__":
    m_parser = argparse.ArgumentParser(description="")
    m_parser.add_argument("--classes", type=int, default=2)  
    m_parser.add_argument("--lr", type=float, default=1e-3)  # 1e-3
    m_parser.add_argument("--batch_size", type=int, default=16)
    m_parser.add_argument("--window_size", type=int, default=90)  
    m_parser.add_argument("--window_step", type=int, default=1)  
    m_parser.add_argument("--wavelet_level", type=int, default=1)  
    m_parser.add_argument("--wavelet_fun", type=str, default="db4")  
    m_parser.add_argument("--if_load", type=int, default=0)
    m_parser.add_argument("--sum_fold", type=int, default=10)
    m_parser.add_argument("--only_test", type=int, default=0)
    m_parser.add_argument("--seed", type=int, default=9)
    m_parser.add_argument("--max_epoch", type=int, default=300)
    m_parser.add_argument("--label_smoothing", type=float, default=0.1)
    m_parser.add_argument("--embedding_dim", type=int, default=64)

    args = m_parser.parse_args()

    fast_dev_run = False
    utils.set_seed(args.seed)

    mat_folder_path = "data/ADNI2"
    excel_file_path = "data/ADNI2_processed_subjects.xlsx"
    output_file_path = "data/data_pt/ADNI2_output_data.pt"
    save_txt_name = os.path.join("results/MCI2/", "my_data_save", "log.txt")

    all_data, all_labels, all_ids = load_or_generate_data(mat_folder_path, excel_file_path, output_file_path,
                                                          args.if_load)
    all_data = all_data.transpose(-1, -2)  

    time_points = all_data.shape[-1]
    num_brain_areas = all_data.shape[1]  # all_data.shape[1]
    num_samples = all_data.shape[0]

    skf = KFold(n_splits=args.sum_fold, shuffle=True, random_state=args.seed)
    results = np.array([])
    sen = np.array([])
    spe = np.array([])
    auc = np.array([])
    f1score = np.array([])

    with open(save_txt_name, 'a') as f:
        f.write("----------------------------\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    for k, (train_idx, test_idx) in enumerate(skf.split(all_data, all_labels)):      
        try:
            del train_data_loader, test_data_loader, val_data_loader, MyTrainer, train_data, train_label, test_data, test_label, val_data, val_label
        except:
            pass
        torch.cuda.empty_cache()
        train_data = all_data[train_idx]
        train_label = all_labels[train_idx]
        test_data = all_data[test_idx]
        test_label = all_labels[test_idx]

        train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1,
                                                                        train_size=0.9, random_state=args.seed)

        # data_loader
        train_data_loader = get_dual_data_loader(train_data, train_label, batch_size=args.batch_size,
                                                 window_size=args.window_size
                                                 , window_step=args.window_step,
                                                 wavelet=args.wavelet_fun,
                                                 level=args.wavelet_level,
                                                 delete_nan=False,
                                                 num_worker=0)
        val_data_loader = get_dual_data_loader(val_data, val_label, batch_size=args.batch_size,
                                               window_size=args.window_size
                                               , window_step=args.window_step,
                                               wavelet=args.wavelet_fun,
                                               level=args.wavelet_level,
                                               delete_nan=False,
                                               shuffle=False, num_worker=0)
        test_data_loader = get_dual_data_loader(test_data, test_label, batch_size=args.batch_size,
                                                window_size=args.window_size
                                                , window_step=args.window_step,
                                                wavelet=args.wavelet_fun,
                                                level=args.wavelet_level,
                                                delete_nan=False,
                                                shuffle=False, num_worker=0)

        num_window = train_data_loader.dataset.num_window
        model = DualLSTMClassifier(num_rois=num_brain_areas, window_size=args.window_size,
                                   num_bands=args.wavelet_level + 1, out_dim=args.classes, num_window=num_window,
                                   embedding_dim=args.embedding_dim)

        initialize_weights(model)


        checkpoint_callback = ModelCheckpoint(
            monitor='val_accuracy',  
            dirpath='results/MCI2/my_save_folder/checkpoints/DSMFNet_test/fold-{}'.format(k),  
            filename='model-{epoch:02d}-{val_accuracy:.2f}',  
            save_top_k=1,  
            mode='max',  
        )


        early_stop_callback = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0,  
            patience=args.max_epoch,
            verbose=True,
            mode='max'
        )

        if args.only_test:
            fold_path = "results/MCI2/my_save_folder/checkpoints/DSMFNet_test/fold-{}".format(k)
            ckpt_path = glob.glob(os.path.join(fold_path, "*.ckpt"))[0]
            MyTrainer = DualTrainer.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                model=model, metrics=["accuracy", "recall", "specificity", "auroc", "f1score"])
            score = MyTrainer.test(test_loader=test_data_loader)
            acc = score[0]["test_accuracy"]
            results = np.append(results, acc)
            sen = np.append(sen, score[0]["test_recall"])
            spe = np.append(spe, score[0]["test_specificity"])
            auc = np.append(auc, score[0]["test_auroc"])
            f1score = np.append(f1score, score[0]["test_f1score"])

        else:
            MyTrainer = DualTrainer(model=model, num_classes=args.classes, lr=args.lr,
                                    weight_decay=1e-4,
                                    accelerator="gpu",
                                    metrics=["accuracy"], label_smoothing=args.label_smoothing, args=args)

            MyTrainer.fit(train_loader=train_data_loader, val_loader=val_data_loader, min_epochs=0,
                          max_epochs=args.max_epoch,
                          precision="32",
                          default_root_dir='results/MCI2/my_save_folder/log_dirs_train/DSMFNet_test/fold-{}'.format(k),
                          callbacks=[checkpoint_callback, early_stop_callback],
                          detect_anomaly=False,
                          fast_dev_run=fast_dev_run, overfit_batches=0, deterministic="warn")
            torch.cuda.empty_cache()
            best_model_path = checkpoint_callback.best_model_path
            print(best_model_path)

            MyTrainer = DualTrainer.load_from_checkpoint(checkpoint_path=best_model_path, model=model)

            score = MyTrainer.test(test_loader=test_data_loader,
                                   default_root_dir='results/MCI2/my_save_folder/log_dirs_test/DSMFNet_test/fold-{}'.format(k))
            acc = score[0]["test_accuracy"]
            results = np.append(results, acc)
        with open(save_txt_name, "a") as f:
            log = "fold{} {:.2%}\n".format(k, acc)
            f.write(log)
    with open(save_txt_name, "a") as f:
        f.write("average acc {:.2%}\n----------------------\n".format(results.mean()))
