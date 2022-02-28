import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *
from utils.tools import get_mask_from_lengths

class StyleFormant(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(StyleFormant, self).__init__()
        self.model_config = model_config

        self.mel_style_encoder = MelStyleEncoder(preprocess_config, model_config)
        self.encoder = PhonemeEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.formant_generator = Generator(model_config)
        self.excitation_generator = Generator(model_config, query_projection=True)
        self.decoder = Decoder(preprocess_config, model_config)

        # not in FPF
        self.phoneme_linear = nn.Linear(
            model_config["transformer"]["encoder_hidden"],
            model_config["transformer"]["encoder_hidden"]
        )
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        )

        self.D_t = PhonemeDiscriminator(preprocess_config, model_config)
        self.D_s = StyleDiscriminator(preprocess_config, model_config)
        
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json") ,"r") as f:
            n_speaker = len(json.load(f))
        
        self.style_prototype = nn.Embedding(
            n_speaker,
            model_config["melencoder"]["encoder_hidden"]
        )

    # Generate Mel Spectrogram
    def G(
            self, style_vector,
            texts, src_masks,  
            mel_masks, max_mel_len,
            p_targets=None,
            d_targets=None,
            p_control=1.0,
            d_control=1.0,
        ):

        """ Encoder """
        output = self.encoder(texts, style_vector, src_masks)
        # output = self.phoneme_linear(output)




        """ Variance Adaptor """
        # max_mel_len is about ref audio
        (
            h, p, \
            p_predictions, log_d_predictions, d_rounded, \
            mel_lens, mel_masks
        ) = self.variance_adaptor(
            output, \
            style_vector,\
            src_masks, mel_masks, max_mel_len, \
            p_targets, d_targets, \
            p_control, d_control
        )

        formant_hidden = self.formant_generator(h, mel_masks)
    
        excitation_hidden = self.excitation_generator(p, mel_masks, hidden_query=h)


        """ Decoder """
        mel_iters, mel_masks = self.decoder(style_vector, formant_hidden, excitation_hidden, mel_masks)
        #mel_iters = self.mel_linear(mel_iters)



        return (
            mel_iters, \
            p_predictions, log_d_predictions, d_rounded, \
            mel_lens, mel_masks
        )        
    

    # StyleSpeech에서는 speakers 안쓰는 듯 _로 되어있었음
    def forward(
        self, speakers, texts, src_lens, max_src_len, \
        mels=None, mel_lens=None, max_mel_len=None, \
        p_targets=None,
        d_targets=None,
        p_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        style_vector = self.mel_style_encoder(mels, mel_masks)


        ## FPF
        output = self.encoder(texts, style_vector, src_masks)


        (
            mel_iters,
            p_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.G(
            style_vector,
            texts, src_masks,  
            mel_masks, max_mel_len,
            p_targets,
            d_targets,
            p_control,
            d_control
        )

        return (
            mel_iters,
            p_predictions, log_d_predictions, d_rounded,
            src_masks, mel_masks, src_lens, mel_lens
        )


    
    def meta_learner_1(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets=None,
        d_targets=None,
        raw_quary_texts=None,
        quary_texts=None,
        quary_src_lens=None,
        max_quary_src_len=None,
        quary_d_targets=None,
        p_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        quary_mel_lens = quary_d_targets.sum(dim=-1)
        max_quary_mel_len = max(quary_mel_lens).item()
        quary_src_masks = get_mask_from_lengths(quary_src_lens, max_quary_src_len)
        quary_mel_masks = get_mask_from_lengths(quary_mel_lens, max_quary_mel_len)

        style_vector = self.mel_style_encoder(mels, mel_masks)

        (
            output,
            _,
            _,
            d_rounded_adv,
            mel_lens_adv,
            mel_masks_adv,
        ) = self.G(
            style_vector,
            quary_texts,
            quary_src_masks,
            quary_mel_masks,
            max_quary_mel_len,
            None,
            quary_d_targets,
            p_control,
            d_control,
        )


        D_s = self.D_s(self.style_prototype, speakers, output[0], mel_masks_adv)
        quary_texts = self.encoder.src_word_emb(quary_texts)
        D_t = self.D_t(self.variance_adaptor.upsample, quary_texts, output[0], max(mel_lens_adv).item(), mel_masks_adv, d_rounded_adv)

        (
            G,
            p_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.G(
            style_vector,
            texts,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            d_targets,
            p_control,
            d_control,
        )

        return (
            D_s,
            D_t,
            G,
            p_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

    def meta_learner_2(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets=None,
        d_targets=None,
        raw_quary_texts=None,
        quary_texts=None,
        quary_src_lens=None,
        max_quary_src_len=None,
        quary_d_targets=None,
        p_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        quary_mel_lens = quary_d_targets.sum(dim=-1)
        max_quary_mel_len = max(quary_mel_lens).item()
        quary_src_masks = get_mask_from_lengths(quary_src_lens, max_quary_src_len)
        quary_mel_masks = get_mask_from_lengths(quary_mel_lens, max_quary_mel_len)

        style_vector = self.mel_style_encoder(mels, mel_masks)

        (
            output,
            _,
            _,
            d_rounded_adv,
            mel_lens_adv,
            mel_masks_adv,
        ) = self.G(
            style_vector,
            quary_texts,
            quary_src_masks,
            quary_mel_masks,
            max_quary_mel_len,
            None,
            quary_d_targets,
            p_control,
            d_control,
        )

        texts = self.encoder.src_word_emb(texts)
        D_t_s = self.D_t(self.variance_adaptor.upsample, texts, mels, max_mel_len, mel_masks, d_targets)

        quary_texts = self.encoder.src_word_emb(quary_texts)
        D_t_q = self.D_t(self.variance_adaptor.upsample, quary_texts, output[0], 
                        max(mel_lens_adv).item(), mel_masks_adv, d_rounded_adv)

        D_s_s = self.D_s(self.style_prototype, speakers, mels, mel_masks)
        D_s_q = self.D_s(self.style_prototype, speakers, output[0], mel_masks_adv)

        # Get Style Logit
        w = style_vector.squeeze() # [B, H]
        style_logit = torch.matmul(w, self.style_prototype.weight.contiguous().transpose(0, 1)) # [B, K]

        return D_t_s, D_t_q, D_s_s, D_s_q, style_logit