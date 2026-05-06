import os
import math
from math import isnan
import re
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
from lora import get_lora_params
import models


class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
    
    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        # ── Device setup ───────────────────────────────────────────────────────
        self.device = torch.device('cuda' if (torch.cuda.is_available() and cuda) else 'cpu')
        print("\n" + "="*60)
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU : {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  WARNING: CUDA not available — training on CPU will be very slow.")
            print("  In Colab: Runtime → Change runtime type → GPU")
        print("="*60)

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)

        # Print model parameters
        print("\n" + "="*60)
        print("Model Parameters:")
        print("="*60)
        for name, param in self.model.named_parameters():
            # Bert freezing customizations 
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= 8:
                        param.requires_grad = False
            elif self.train_config.data == "ur_funny":
                if "bertmodel" in name and "lora" not in name:
                    param.requires_grad = False
            
            # Freeze non-LoRA parameters if using LoRA-only training
            if self.train_config.use_lora and self.train_config.train_only_lora:
                if 'lora_' not in name:
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            
            print(f"\t{name}: {param.shape} - requires_grad={param.requires_grad}")

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.train_config.use_bert:
            if self.train_config.pretrained_emb is not None:
                self.model.embed.weight.data = self.train_config.pretrained_emb
            self.model.embed.requires_grad = False
        
        self.model.to(self.device)

        if self.is_train:
            # Use LoRA-specific parameters if LoRA is enabled
            if self.train_config.use_lora and self.train_config.train_only_lora:
                lora_params = get_lora_params(self.model)
                self.optimizer = self.train_config.optimizer(
                    lora_params,
                    lr=self.train_config.learning_rate
                )
                print(f"\n[LoRA Mode] Training only {len(lora_params)} LoRA parameters")
            else:
                self.optimizer = self.train_config.optimizer(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.train_config.learning_rate
                )
                print(f"\n[Standard Mode] Training all {sum(1 for p in self.model.parameters() if p.requires_grad)} parameters")

        print("="*60 + "\n")


    @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1

        # Select loss criterion based on dataset
        if self.train_config.data == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else:  # mosi and mosei are regression datasets
            self.criterion = criterion = nn.MSELoss(reduction="mean")

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        
        best_valid_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        train_losses = []
        valid_losses = []
        
        for e in range(self.train_config.n_epoch):
            self.model.train()

            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss_recon = []
            train_loss_sp = []
            train_loss = []
            y_pred = []
            y_true = []
            
            for batch in self.train_data_loader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                batch_size = t.size(0)
                t    = t.to(self.device)
                v    = v.to(self.device)
                a    = a.to(self.device)
                y    = y.to(self.device)
                l    = l.to(self.device)
                bert_sent      = bert_sent.to(self.device)
                bert_sent_type = bert_sent_type.to(self.device)
                bert_sent_mask = bert_sent_mask.to(self.device)

                is_train = self.model.training

                self.model.is_test = self.train_config.is_test
                self.model.train_method = self.train_config.train_method
                self.model.train_changed_modal = self.train_config.train_changed_modal
                self.model.train_changed_pct = self.train_config.train_changed_pct
                self.model.test_method = self.train_config.test_method
                self.model.test_changed_modal = self.train_config.test_changed_modal
                self.model.test_changed_pct = self.train_config.test_changed_pct

                y_tilde = self.model(is_train, t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                
                if self.train_config.data == "ur_funny":
                    y = y.squeeze()

                cls_loss = criterion(y_tilde, y)
                diff_loss = self.get_diff_loss()
                domain_loss = self.get_domain_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()

                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())
                
                if self.train_config.use_cmd_sim:
                    similarity_loss = cmd_loss
                else:
                    similarity_loss = domain_loss
                
                loss = cls_loss + \
                    self.train_config.diff_weight * diff_loss + \
                    self.train_config.sim_weight * similarity_loss + \
                    self.train_config.recon_weight * recon_loss

                loss.backward()
                
                torch.nn.utils.clip_grad_value_(
                    [param for param in self.model.parameters() if param.requires_grad],
                    self.train_config.clip
                )
                self.optimizer.step()

                train_loss_cls.append(cls_loss.item())
                train_loss_diff.append(diff_loss.item())
                train_loss_recon.append(recon_loss.item())
                train_loss.append(loss.item())
                train_loss_sim.append(similarity_loss.item())
                

            train_losses.append(train_loss)
            print(f"Epoch {e+1}/{self.train_config.n_epoch}")
            print(f"Training loss: {round(np.mean(train_loss), 4)}")

            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()

            accuracy = self.calc_metrics(y_true, y_pred, "train")
            print(f"Training accuracy: {round(accuracy, 4)}")

            valid_loss, valid_acc = self.eval(mode="dev")
            
            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")

            if self.train_config.train_method == "missing":
                save_mode = f'0'
            elif self.train_config.train_method == "g_noise":
                save_mode = f'N'
            elif self.train_config.train_method == "hybird":
                save_mode = f'H'
            else:
                save_mode = f'0'
            
            save_dir = f'checkpoints/{self.train_config.data}/best_{int(self.train_config.train_changed_pct*100)}%{self.train_config.train_changed_modal[0].upper()}={save_mode}'
            if self.train_config.use_lora:
                save_dir += '_LoRA'
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save best model
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'{save_dir}/best_model.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print(f"Patience exceeded. Training stopped at epoch {e+1}")
                    break
            
            # Learning rate scheduling
            lr_scheduler.step()

        # Merge LoRA weights after training if requested
        if self.train_config.use_lora and self.train_config.merge_lora_at_end:
            print("\nMerging LoRA weights into main model...")
            self.model.merge_lora_weights()
            print("LoRA weights merged successfully!")

        # Evaluate on test set
        print("\n" + "="*60)
        print("Loading best model and evaluating on test set...")
        print("="*60)
        
        if not os.path.exists(save_dir):
            save_dir = f'checkpoints/{self.train_config.data}/best_{int(self.train_config.train_changed_pct*100)}%{self.train_config.train_changed_modal[0].upper()}=0'
            if self.train_config.use_lora:
                save_dir += '_LoRA'
        
        self.model.load_state_dict(torch.load(f'{save_dir}/best_model.std', map_location=self.device))
        self.model.to(self.device)
        self.eval(mode="test", to_print=True)

    @time_desc_decorator('Evaluation')
    def eval(self, mode="dev", to_print=False):
        """Evaluate model on validation or test set"""
        self.model.eval()
        
        if mode == "dev":
            dataloader = self.dev_data_loader
        else:
            dataloader = self.test_data_loader

        eval_loss = []
        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                t    = t.to(self.device)
                v    = v.to(self.device)
                a    = a.to(self.device)
                y    = y.to(self.device)
                l    = l.to(self.device)
                bert_sent      = bert_sent.to(self.device)
                bert_sent_type = bert_sent_type.to(self.device)
                bert_sent_mask = bert_sent_mask.to(self.device)

                is_train = self.model.training

                self.model.is_test = self.train_config.is_test
                self.model.train_method = self.train_config.train_method
                self.model.train_changed_modal = self.train_config.train_changed_modal
                self.model.train_changed_pct = self.train_config.train_changed_pct
                self.model.test_method = self.train_config.test_method
                self.model.test_changed_modal = self.train_config.test_changed_modal
                self.model.test_changed_pct = self.train_config.test_changed_pct

                y_tilde = self.model(is_train, t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()
                
                cls_loss = self.criterion(y_tilde, y)
                loss = cls_loss

                eval_loss.append(loss.item())
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)

        return eval_loss, accuracy

    def multiclass_acc(self, preds, truths):
        """Compute multiclass accuracy"""
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """Calculate evaluation metrics"""
    
        if self.train_config.data == "ur_funny":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if to_print:
                print("Confusion Matrix (pos/neg) :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg) :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
            
            return accuracy_score(test_truth, test_preds)

        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)
            
            f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
            
            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            if to_print:
                print("mae: ", mae)
                print("corr: ", corr)
                print("mult_acc: ", mult_a7)
                print("Classification Report (pos/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
            
            # non-neg - neg
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)

            if to_print:
                print("Classification Report (non-neg/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            
            return accuracy_score(binary_truth, binary_preds)

    def get_domain_loss(self):
        """Compute domain adversarial loss"""
        if self.train_config.use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_t = self.model.domain_label_t
        domain_pred_v = self.model.domain_label_v
        domain_pred_a = self.model.domain_label_a

        # True domain labels
        domain_true_t = torch.LongTensor([0]*domain_pred_t.size(0)).to(self.device)
        domain_true_v = torch.LongTensor([1]*domain_pred_v.size(0)).to(self.device)
        domain_true_a = torch.LongTensor([2]*domain_pred_a.size(0)).to(self.device)

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self):
        """Compute Correlation-Max Discrepancy loss"""
        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
        loss = loss / 3.0

        return loss

    def get_diff_loss(self):
        """Compute difference loss between private and shared representations"""
        shared_t = self.model.utt_shared_t
        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss
    
    def get_recon_loss(self):
        """Compute reconstruction loss"""
        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss / 3.0
        return loss