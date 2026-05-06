import numpy as np
import random
import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from utils import to_gpu, ReverseLayerF
from lora import LoRALinear, add_lora_to_linear, merge_lora_weights, get_lora_params


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)


# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between

        if self.config.use_bert:
            # Initializing a BERT bert-base-uncased style configuration
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        else:
            self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        # ==========================================
        # mapping modalities to same sized space
        # ==========================================
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            if config.use_lora:
                self.project_t.add_module('project_t', LoRALinear(
                    in_features=768, 
                    out_features=config.hidden_size,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout
                ))
            else:
                self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            if config.use_lora:
                self.project_t.add_module('project_t', LoRALinear(
                    in_features=hidden_sizes[0]*2,
                    out_features=config.hidden_size,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout
                ))
            else:
                self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*2, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        if config.use_lora:
            self.project_v.add_module('project_v', LoRALinear(
                in_features=hidden_sizes[1]*2,
                out_features=config.hidden_size,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*2, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        if config.use_lora:
            self.project_a.add_module('project_a', LoRALinear(
                in_features=hidden_sizes[2]*2,
                out_features=config.hidden_size,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*2, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        # ==========================================
        # private encoders
        # ==========================================
        self.private_t = nn.Sequential()
        if config.use_lora:
            self.private_t.add_module('private_t_1', LoRALinear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        if config.use_lora:
            self.private_v.add_module('private_v_1', LoRALinear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        if config.use_lora:
            self.private_a.add_module('private_a_3', LoRALinear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        
        # ==========================================
        # shared encoder
        # ==========================================
        self.shared = nn.Sequential()
        if config.use_lora:
            self.shared.add_module('shared_1', LoRALinear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        # ==========================================
        # reconstruct
        # ==========================================
        self.recon_t = nn.Sequential()
        if config.use_lora:
            self.recon_t.add_module('recon_t_1', LoRALinear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        
        self.recon_v = nn.Sequential()
        if config.use_lora:
            self.recon_v.add_module('recon_v_1', LoRALinear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        
        self.recon_a = nn.Sequential()
        if config.use_lora:
            self.recon_a.add_module('recon_a_1', LoRALinear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        # ==========================================
        # shared space adversarial discriminator
        # ==========================================
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            if config.use_lora:
                self.discriminator.add_module('discriminator_layer_1', LoRALinear(
                    in_features=config.hidden_size,
                    out_features=config.hidden_size,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout
                ))
            else:
                self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            if config.use_lora:
                self.discriminator.add_module('discriminator_layer_2', LoRALinear(
                    in_features=config.hidden_size,
                    out_features=len(hidden_sizes),
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout
                ))
            else:
                self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        # ==========================================
        # shared-private collaborative discriminator
        # ==========================================
        self.sp_discriminator = nn.Sequential()
        if config.use_lora:
            self.sp_discriminator.add_module('sp_discriminator_layer_1', LoRALinear(
                in_features=config.hidden_size,
                out_features=4,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))

        # ==========================================
        # fusion module
        # ==========================================
        self.fusion = nn.Sequential()
        if config.use_lora:
            self.fusion.add_module('fusion_layer_1', LoRALinear(
                in_features=self.config.hidden_size*6,
                out_features=self.config.hidden_size*3,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        if config.use_lora:
            self.fusion.add_module('fusion_layer_3', LoRALinear(
                in_features=self.config.hidden_size*3,
                out_features=output_size,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            ))
        else:
            self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features=output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        """Extract features from sequence using bidirectional RNN"""
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        padded_h1 = torch.nn.functional.dropout(padded_h1, p=self.config.dropout, training=self.training)

        if self.config.rnncell == "lstm":
            packed_h2, (final_h2, _) = rnn2(pack_padded_sequence(padded_h1, lengths))
        else:
            packed_h2, final_h2 = rnn2(pack_padded_sequence(padded_h1, lengths))

        padded_h2, _ = pad_packed_sequence(packed_h2)
        padded_h2 = torch.nn.functional.dropout(padded_h2, p=self.config.dropout, training=self.training)

        return padded_h2

    def alignment(self, is_train, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        """Main forward pass with alignment"""
        
        batch_size = lengths.size(0)
        
        # Get BERT embeddings for text
        if self.config.use_bert:
            bert_output = self.bertmodel(input_ids=bert_sent, token_type_ids=bert_sent_type, attention_mask=bert_sent_mask)
            utterance_text = bert_output[1]  # Use [CLS] token representation
            utterance_text = utterance_text.unsqueeze(1).expand(-1, lengths.max().item(), -1).transpose(0, 1)
        else:
            utterance_text = self.embed(sentences)
            utterance_text = self.extract_features(utterance_text, lengths, self.trnn1, self.trnn2, self.tlayer_norm)

        # Get visual embeddings
        utterance_video = self.extract_features(video, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        
        # Get acoustic embeddings
        utterance_audio = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)

        # Take mean across sequence dimension
        utterance_text = utterance_text.mean(dim=0)
        utterance_video = utterance_video.mean(dim=0)
        utterance_audio = utterance_audio.mean(dim=0)

        # Apply modality perturbation during training
        if is_train and self.config.train_method != 'none':
            utterance_text, utterance_video, utterance_audio = self._apply_train_perturbations(
                utterance_text, utterance_video, utterance_audio
            )

        # Apply test perturbations if needed
        if self.config.is_test:
            utterance_text, utterance_video, utterance_audio = self._apply_test_perturbations(
                utterance_text, utterance_video, utterance_audio
            )

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        # Domain adversarial loss (if not using CMD)
        if not self.config.use_cmd_sim:
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        # Shared-private discriminator
        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator((self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0)
        
        # Reconstruction
        self.reconstruct()
        
        # Fusion with transformer
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, 
                        self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        o = self.fusion(h)
        return o
    
    def _apply_train_perturbations(self, text, video, audio):
        """Apply perturbations during training"""
        utterance = text if self.config.train_changed_modal == 'language' else (
            video if self.config.train_changed_modal == 'video' else audio
        )
        
        if self.config.train_method == 'missing':
            for i in range(len(utterance)):
                if torch.rand(1) < self.config.train_changed_pct:
                    utterance[i] = utterance[i] * 0
        elif self.config.train_method == 'g_noise':
            noise = torch.normal(0, 1, utterance.shape).to(utterance.device)
            sample_num = int(len(utterance) * self.config.train_changed_pct)
            sample_list = np.random.choice(len(utterance), sample_num, replace=False)
            utterance[sample_list] = utterance[sample_list] * noise[sample_list]
        
        if self.config.train_changed_modal == 'language':
            text = utterance
        elif self.config.train_changed_modal == 'video':
            video = utterance
        else:
            audio = utterance
            
        return text, video, audio
    
    def _apply_test_perturbations(self, text, video, audio):
        """Apply perturbations during testing"""
        utterance = text if self.config.test_changed_modal == 'language' else (
            video if self.config.test_changed_modal == 'video' else audio
        )
        
        if self.config.test_method == 'missing':
            for i in range(len(utterance)):
                if torch.rand(1) < self.config.test_changed_pct:
                    utterance[i] = utterance[i] * 0
        elif self.config.test_method == 'g_noise':
            noise = torch.normal(0, 1, utterance.shape).to(utterance.device)
            sample_num = int(len(utterance) * self.config.test_changed_pct)
            sample_list = np.random.choice(len(utterance), sample_num, replace=False)
            utterance[sample_list] = utterance[sample_list] * noise[sample_list]
        
        if self.config.test_changed_modal == 'language':
            text = utterance
        elif self.config.test_changed_modal == 'video':
            video = utterance
        else:
            audio = utterance
            
        return text, video, audio
    
    def reconstruct(self):
        """Reconstruct original representations from shared+private"""
        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_v, utterance_a):
        """Decompose into shared and private representations"""
        # Project to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, is_train, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        """Forward pass"""
        o = self.alignment(is_train, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o

    def get_lora_params(self):
        """Get all LoRA parameters for optimization"""
        return get_lora_params(self)

    def freeze_non_lora(self):
        """Freeze all parameters except LoRA parameters"""
        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

    def merge_lora_weights(self):
        """Merge LoRA weights into main weights"""
        merge_lora_weights(self)