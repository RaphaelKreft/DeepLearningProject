"""
This file contains training logic for the Contrastive Learning Paradigm.

Usage:
    python train_hybrid.py --config <path-to-json-config-file> [options]

Example config files fo our project:
    - configs/DLPROJ_pretrain_Hybrid_CNN_Attention_Sleep-EDF-2018.json
    - configs/DLPROJ_pretrain_Hybrid_CNN_Sleep-EDF-2018.json
    - configs/DLPROJ_pretrain_Hybrid_Transformer_Sleep-EDF-2018.json
"""

import json
import argparse
import warnings

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.main_model_dlproj import MainModelDLProject
from utils import *
from loss import *
from loader import EEGDataLoader


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold

        self.cfg = config
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        self.dset_cfg = config['dataset']

        # assert that the correct training mode is set: 'pretrain_mp'. This makes sure the models and dataset show correct behavior
        if not 'mode' in self.tp_cfg.keys() and self.tp_cfg['mode'] == 'pretrain-hybrid':
            raise ValueError(
                'Running train_hybrid.py, only mode pretrain-hybrid is supported and must be declared in the config file')

        self.device = get_device(preference="cuda")
        print('[INFO] Config name: {}'.format(config['name']))
        print('[INFO] Device: {}'.format(str(self.device)))

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        # create tensorboard writer
        self.writer = SummaryWriter(log_dir=os.path.join("logs", config['name'], f"fold-{fold}"))

        # load selected loss with its parameters (if these parameters are given) - only used if model has no internal loss calculation
        assert 'loss_mp' in self.tp_cfg.keys() and 'loss_crl' in self.tp_cfg.keys()
        assert self.tp_cfg['loss_mp'] in SUPPORTED_LOSS_FUNCTIONS and self.tp_cfg['loss_crl'] in SUPPORTED_LOSS_FUNCTIONS
        # 4. change criterion to classification loss if training classifier
        self.criterion_classifier = LOSS_MAP["cross_entropy"]()
        self.criterion_mp = LOSS_MAP[self.tp_cfg['loss_mp']](
                **(self.tp_cfg['loss_params_mp'] if 'loss_params_mp' in self.tp_cfg.keys() else {}))
        self.criterion_crl = LOSS_MAP[self.tp_cfg['loss_crl']](
                **(self.tp_cfg['loss_params_crl'] if 'loss_params_crl' in self.tp_cfg.keys() else {}))
        self.alpha_crl = float(self.tp_cfg['alpha_crl'])

        # check if masking is performed internally, in that case throw a warning if masking is also activated in dataset
        if "masking" in self.dset_cfg.keys() and not self.dset_cfg["masking"]:
            raise ValueError("MAsking needs to be activated for hybrid training")

        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)

        # use switch mode to setup rest - initially we activate masking
        self.dset_masking_activated = ("masking" in self.dset_cfg.keys() and self.dset_cfg["masking"])
        self.switch_mode(self.tp_cfg['mode'], set_masking=self.dset_masking_activated)

    def switch_mode(self, mode, set_masking=False):
        """
        Switch mode switches internal mode of main model and resets optimizer, dataloaders and early stopping
        """
        # 0. set variables
        self.train_iter = 0
        self.cfg['training_params']['mode'] = mode
        self.tp_cfg['mode'] = mode
        self.dset_cfg['masking'] = set_masking
        self.cfg['dataset']['masking'] = set_masking
        self.dset_masking_activated = ("masking" in self.dset_cfg.keys() and self.dset_cfg["masking"])
        # 1. Switch model mode (Main Model does setup and freezing parts(bb and classifier) internally)
        self.model.module.switch_mode(mode)
        self.model.to(self.device)
        if not mode in ['gen-embeddings', 'classification']:
            self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.tp_cfg['lr'],
                                        weight_decay=self.tp_cfg['weight_decay'])
        # 2. Reset early stopping
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path,
                                            ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])
        # 3. reinitialize datasets/loaders for new mode
        self.loader_dict = self.build_dataloader()
        print('[INFO] Overall Number of trainable parameters in MainModel: ',
              sum(p.numel() for p in self.model.parameters() if p.requires_grad))


    def reload_best_model_weights(self):
        # reload best model weights from checkpoint
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)), strict=False)
        self.model.to(self.device)


    def build_model(self):
        model = MainModelDLProject(self.cfg)
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        return model

    def build_dataloader(self):
        dataloader_args = {'batch_size': self.tp_cfg['batch_size'],
                           # default data loader args, using 4 workers per GPU, also have 2 batches prefetched by default
                           'shuffle': True,
                           'num_workers': 4 * len(self.args.gpu.split(",")),
                           # self.args.gpu.split defaults to 1 even when arg not given
                           'pin_memory': True}
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, **dataloader_args)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, **dataloader_args)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                 num_workers=4 #* len(self.args.gpu.split(","))
                                 ,pin_memory=True)
        print('[INFO] Dataloader prepared')
        print('[INFO] Batch-Size: {}'.format(self.tp_cfg['batch_size']))
        print('[INFO] Train-Batches: {}, Val-Batches: {}, Test-Batches: {}'.format(len(train_loader), len(val_loader), len(test_loader)))
        return {'train': train_loader, 'val': val_loader, 'test': test_loader}


    def train_one_epoch(self, epoch, classifier=False):
        """
        We modify this method from train_crl.py to be able to deal with only raw single eeg epochs and not the two-view
        augmented approach.

        @params:
            classifier:bool     Specifies whether the training is for the classifier or the backbone. Difference is that in case of classifier training
                                only one view epoch is loaded and CE loss is used.
        """
        self.model.module.activate_train_mode()
        train_loss = 0
        train_mode = 'classifier' if self.tp_cfg['mode'] == 'train-classifier' else 'backbone'


        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            # inputs for backbone train: [{inputs, masked_inp, mask}, input_a, input_b]
            # inputs for classifier training: single view batches.
            loss = 0
            labels = labels.view(-1).to(self.device)  # No effect in our case!

            if not classifier:
                masked_inp_dict = inputs[0]
                masked_input = masked_inp_dict["masked_inputs"].to(self.device)
                mask = masked_inp_dict["mask"].to(self.device)
                original_inputs = masked_inp_dict["inputs"].to(self.device)
                augmented_a, augmented_b = inputs[1].to(self.device), inputs[2].to(self.device)
                inputs = torch.cat([masked_input, augmented_a, augmented_b], dim=0)

            inputs = inputs.to(self.device)
            outputs = self.model(inputs) # list of outputs

            if not classifier:
                reconstruction, _, _ = torch.split(outputs[0], [labels.size(0), labels.size(0), labels.size(0)], dim=0)
                _, latent_1, latent_2 = torch.split(outputs[1], [labels.size(0), labels.size(0), labels.size(0)], dim=0)
                latent_outputs = torch.cat([latent_1.unsqueeze(1), latent_2.unsqueeze(1)], dim=1)
                mp_loss = self.criterion_mp(original_inputs, outputs=reconstruction, reduction='mean', mask=mask, labels=None)
                crl_loss = self.criterion_crl(None, outputs=latent_outputs, mask=None, labels=None)
                loss += mp_loss + self.alpha_crl * crl_loss
                self.writer.add_scalar(f"train/loss-mp", mp_loss.item(), self.train_iter)
                self.writer.add_scalar(f"train/loss-crl", crl_loss.item(), self.train_iter)
            else:
                # for classification we only expect logit output
                outputs = outputs[0]
                loss += self.criterion_classifier(inputs, outputs, reduction='mean', mask=None, labels=labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar(f"train/total-loss-{train_mode}", loss.item(), self.train_iter)
            train_loss += loss.item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']),
                         'Lr: %.4e | Loss: %.6f' % (get_lr(self.optimizer), train_loss / (i + 1)))

            # perform validation every X iterations
            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                print(f'[INFO] Starting evaluation...')
                if classifier:
                    val_loss, _, _ = self.benchmark_classifier(mode='val') 
                else:
                    val_loss = self.evaluate(mode='val')
                self.early_stopping(None, val_loss, self.model)
                self.model.module.activate_train_mode()
                if self.early_stopping.early_stop:
                    print("[INFO] Early stopping...")
                    break

        # Log average training loss of an epoch to TensorBoard
        avg_train_loss = train_loss / len(self.loader_dict['train'])
        self.writer.add_scalar(f"train/epoch-avg-loss-{train_mode}", avg_train_loss, epoch)
        print(f"\n[INFO] Epoch {epoch}, Epochal Avg - Training Loss: {avg_train_loss:.6f}")

    @torch.no_grad()
    def benchmark_classifier(self, mode='test'):
        """
        Evaluates Whole model including classifier on the test set
        """
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])

            for j in range(len(outputs)):
                loss += F.cross_entropy(outputs[j], labels)
                outputs_sum += outputs[j]

            eval_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()

            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs_sum.cpu().numpy()])

            progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (eval_loss / (i + 1), 100. * correct / total, correct, total))

        avg_eval_loss = eval_loss / len(self.loader_dict[mode])
        print(f"[INFO] {mode.capitalize()} Eval-Loss: {avg_eval_loss:.4f}")
        self.writer.add_scalar(f"{mode.capitalize()}/loss-avg-classifier", avg_eval_loss, self.train_iter)
        # Compute additional metrics and log to TensorBoard for classifier
        self.log_metrics_to_tensorboard(y_true, y_pred)
        self.writer.flush()
        return eval_loss, y_true, y_pred


    @torch.no_grad()
    def evaluate(self, mode):
        """
        Runs the validation during model training on the val set.
        """
        self.model.eval()
        eval_loss, eval_mp_loss, eval_crl_loss = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            # inputs loaded one view, construct val data for hybrid training here

            loss = 0
            labels = labels.view(-1).to(self.device)
            # Prepare Inputs
            masked_inp_dict = inputs
            masked_input = masked_inp_dict["masked_inputs"].to(self.device)
            mask = masked_inp_dict["mask"].to(self.device)
            original_inputs = masked_inp_dict["inputs"].to(self.device)
            inputs = torch.cat([masked_input, original_inputs], dim=0)

            inputs = inputs.to(self.device)
            outputs = self.model(inputs) # list of outputs
            # Unpack outputs and calculate loss
            reconstruction, _ = torch.split(outputs[0], [labels.size(0), labels.size(0)], dim=0)
            _, latent_1 = torch.split(outputs[1], [labels.size(0), labels.size(0)], dim=0)
            latent_outputs = latent_1.unsqueeze(1).repeat(1, 2, 1)
            mp_loss = self.criterion_mp(original_inputs, outputs=reconstruction, reduction='mean', mask=mask, labels=None)
            crl_loss = self.criterion_crl(None, outputs=latent_outputs, mask=None, labels=None)
            loss += mp_loss + self.alpha_crl * crl_loss

            # calculate loss based on predictions, gt and whether a mask is given or not
            eval_loss += loss.item()
            eval_mp_loss += mp_loss.item()
            eval_crl_loss += crl_loss.item()

            progress_bar(i, len(self.loader_dict[mode]),
                         'Lr: %.4e | Loss: %.6f' % (get_lr(self.optimizer), eval_loss / (i + 1)))

        avg_eval_loss = eval_loss / len(self.loader_dict[mode])
        avg_eval_mp_loss = eval_mp_loss / len(self.loader_dict[mode])
        avg_eval_crl_loss = eval_crl_loss / len(self.loader_dict[mode])
        print(f"[INFO] {mode.capitalize()} Eval-Loss: {avg_eval_loss:.4f}")
        self.writer.add_scalar(f"{mode.capitalize()}/loss-avg-backbone", avg_eval_loss, self.train_iter)
        self.writer.add_scalar(f"{mode.capitalize()}/loss-avg-mp", avg_eval_mp_loss, self.train_iter)
        self.writer.add_scalar(f"{mode.capitalize()}/loss-avg-crl", avg_eval_crl_loss, self.train_iter)
        return eval_loss

    def log_metrics_to_tensorboard(self, y_true, y_pred):
        y_pred_argmax = np.argmax(y_pred, 1)
        result_dict = skmet.classification_report(y_true, y_pred_argmax, digits=3, output_dict=True)

        # Extract relevant metrics
        accuracy = round(result_dict['accuracy']*100, 1)
        macro_f1 = round(result_dict['macro avg']['f1-score']*100, 1)
        kappa = round(skmet.cohen_kappa_score(y_true, y_pred_argmax), 3)

        # Log to TensorBoard
        self.writer.add_scalar(f"Metrics/Accuracy", accuracy, self.train_iter)
        self.writer.add_scalar(f"Metrics/Macro_F1", macro_f1, self.train_iter)
        self.writer.add_scalar(f"Metrics/Cohen_Kappa", kappa, self.train_iter)



    def generate_and_store_embeddings(self):
        self.model.eval()
        embeddings = []
        labels_list = []
        for i, (inputs, labels) in enumerate(self.loader_dict['test']):
            inputs = inputs.to(self.device)
            embedding = self.model(inputs)[0]
            embeddings.append(embedding)
            labels_list.append(labels)
        embedding_torch = torch.cat(embeddings, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        embeddings_path = os.path.join(self.ckpt_path, 'embeddings.pt')
        print("[INFO] Storing embeddings to {}".format(embeddings_path))
        torch.save({"embeddings": embedding_torch, "labels": all_labels}, embeddings_path)

    def train_classifier(self):
        self.model.train()
        for epoch in range(self.tp_cfg['classifier_epochs']):
            print('\n[INFO] ClassifierTrain, Epoch: {}'.format(epoch))
            self.train_one_epoch(epoch, classifier=True)
            self.writer.flush()
            if self.early_stopping.early_stop:
                break


    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch)
            self.writer.flush()
            if self.early_stopping.early_stop:
                break


def main():
    """
    Will train the backbone, generate embeddings, train a mlp classifier and benchmark it
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')

    # for our use-case we only need one split (no cross validation needed)
    trainer = OneFoldTrainer(args, 1, config)
    trainer.run()

    # Generate embeddings for later benchmark of latent space - store to
    print("[INFO] Generate and store embeddings...")
    trainer.switch_mode('gen-embeddings', set_masking=False)
    trainer.reload_best_model_weights()
    trainer.generate_and_store_embeddings()

    #  Train classifier with frozen backbone
    print("[INFO] Training the classifier...")
    trainer.switch_mode('train-classifier', set_masking=False)
    trainer.train_classifier()

    # Perform classification
    print("[INFO] Run classification benchmarks...")
    trainer.switch_mode('classification', set_masking=False)
    trainer.reload_best_model_weights()
    _, y_pred, y_true = trainer.benchmark_classifier()
    summarize_result(config, 1, y_pred, y_true)
    # close tensorboard-writer
    trainer.writer.close()


def test(train_bb=False, gen_embed=False, train_classifier=False, benchmark_classifier=False):
    """
    Takes the transformer model and performs tests as wished.
    """
    sample_cfg = {
        "name": "test",
        "_comment": "Pretraining Encoder backbone using MaskedPrediction. Run train_mp.py script with this config. Projection Head is not used with MP training that's why its omitted",

        "dataset": {
            "name": "Sleep-EDF-2018",
            "eeg_channel": "Fpz-Cz",
            "num_splits": 10,
            "seq_len": 1,
            "target_idx": 0,
            "root_dir": "./",
            "masking": True,
            "masking_type": "fixed_proportion_random",
            "masking_ratio": 0.35
        },

        "backbone": {
            "_comment": "This model uses internal masking on the latent frames and not full signal restoration, also using internal loss calculation. Thats why mask params and loss params need to be defined here",
            "name": "Transformer",
            "fs": 100,
            "second": 30,
            "time_window": 5,
            "time_step": 0.5,
            "encoder_embed_dim": 128,
            "encoder_heads": 8,
            "encoder_depths": 6,
            "decoder_embed_dim": 128,
            "decoder_heads": 4,
            "decoder_depths": 8,
            "projection_hidden": [1024, 512],
            "use_sig_backbone": False,
            "input_size": 3000,
            "num_patches": 1,
            "use_cls": False
        },

        "classifier": {
            "_comment": "Classifier used to finetune it and benmchmark -> linear Evaluation",
            "name": "DLProjMLP",
            "input_dim": 128,
            "hidden_dim": 256,
            "dropout": 0.5,
            "num_classes": 5
        },

        "training_params": {
            "_comment": "All default sleepyco settings despite 'mode'. 'pretrain-mp' is passed to dataloader to use the base EEG epochs, not two-view as in 'pretrain'",
            "mode": "pretrain-hybrid",
            "loss_crl": "NTXent",
            "loss_mp": "l2",
            "alpha_crl": 0,
            "max_epochs": 2,
            "batch_size": 128,
            "lr": 0.0005,
            "weight_decay": 0.0001,
            "temperature": 0.07,
            "val_period": 325,
            "early_stopping": {
                "mode": "min",
                "patience": 8,
            },
            "classifier_epochs": 3
        }
    }
    class Args:
        def __init__(self, gpu):
            self.gpu = gpu

    trainer = OneFoldTrainer(Args("0"), 1, sample_cfg)

    if train_bb:
        print("[INFO] Run Backbone Training...")
        trainer.run()
    else:
        # save initialized weights for case of testing
        trainer.early_stopping.save_checkpoint(-np.inf, trainer.model)
    if gen_embed:
        print("[INFO] Generate and store embeddings...")
        trainer.switch_mode('gen-embeddings', set_masking=False)
        trainer.reload_best_model_weights()
        trainer.generate_and_store_embeddings()
    if train_classifier:
        #  Train classifier with frozen backbone
        print("[INFO] Training the classifier...")
        trainer.switch_mode('train-classifier', set_masking=False)
        trainer.train_classifier()
    if benchmark_classifier:
        # Perform classification
        print("[INFO] Run classification benchmarks...")
        trainer.switch_mode('classification', set_masking=False)
        trainer.reload_best_model_weights()
        _, y_pred, y_true = trainer.benchmark_classifier()
        summarize_result(sample_cfg, 1, y_pred, y_true)


if __name__ == "__main__":
    # Uncomment test for testing
    #test(train_bb=True, gen_embed=True, train_classifier=True, benchmark_classifier=True)
    main()
