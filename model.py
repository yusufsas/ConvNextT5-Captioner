import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from transformers import T5Tokenizer, T5ForConditionalGeneration
from timm import create_model

import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

class ConvnextT5CaptioningModel(nn.Module):
    def __init__(self, image_encoder_name="convnextv2_base", text_decoder_name="t5-base"):
        super().__init__()
        self.encoder = create_model(image_encoder_name, pretrained=True, num_classes=0, global_pool='avg')  # 'avg' pooling doğrudan [B, C] verir
        self.encoder_out_dim = self.encoder.num_features

        self.proj = nn.Sequential(
    # nn.LayerNorm(self.encoder_out_dim),   # [B, C] içindeki her örneği normalize eder
    nn.Linear(self.encoder_out_dim, 2048),
    nn.SiLU(),
    nn.Linear(2048, 768),
    # nn.LayerNorm(768)                     # Son çıktının da dengesini sağlar
      )

        self.decoder = T5ForConditionalGeneration.from_pretrained(text_decoder_name)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        feats = self.encoder(pixel_values)  # [B, encoder_out_dim]
        proj_feats = self.proj(feats)       # [B, 512]

        encoder_hidden_states = proj_feats.unsqueeze(1)  # [B, 1, 512]
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        output = self.decoder(
            encoder_outputs=encoder_outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return output


