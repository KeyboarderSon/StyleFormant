# class StyleDiscriminator(nn.Module):
#     """ Style Discriminator """

#     def __init__(self, preprocess_config, model_config):
#         super(StyleDiscriminator, self).__init__()
#         n_position = model_config["max_seq_len"] + 1
#         n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
#         d_melencoder = model_config["melencoder"]["encoder_hidden"]
#         n_spectral_layer = model_config["melencoder"]["spectral_layer"]
#         n_temporal_layer = model_config["melencoder"]["temporal_layer"]
#         n_slf_attn_layer = model_config["melencoder"]["slf_attn_layer"]
#         n_slf_attn_head = model_config["melencoder"]["slf_attn_head"]
#         d_k = d_v = (
#             model_config["melencoder"]["encoder_hidden"]
#             // model_config["melencoder"]["slf_attn_head"]
#         )
#         kernel_size = model_config["melencoder"]["conv_kernel_size"]

#         self.max_seq_len = model_config["max_seq_len"]

#         self.fc_1 = FCBlock(n_mel_channels, d_melencoder, spectral_norm=True)

#         self.spectral_stack = nn.ModuleList(
#             [
#                 FCBlock(
#                     d_melencoder, d_melencoder, activation=nn.LeakyReLU(), spectral_norm=True
#                 )
#                 for _ in range(n_spectral_layer)
#             ]
#         )

#         self.temporal_stack = nn.ModuleList(
#             [
#                 Conv1DBlock(
#                     d_melencoder, d_melencoder, kernel_size, activation=nn.LeakyReLU(), spectral_norm=True
#                 )
#                 for _ in range(n_temporal_layer)
#             ]
#         )

#         self.slf_attn_stack = nn.ModuleList(
#             [
#                 MultiHeadAttention(
#                     n_slf_attn_head, d_melencoder, d_k, d_v, layer_norm=True, spectral_norm=True
#                 )
#                 for _ in range(n_slf_attn_layer)
#             ]
#         )

#         self.fc_2 = FCBlock(d_melencoder, d_melencoder, spectral_norm=True)

#         self.V = FCBlock(d_melencoder, d_melencoder)
#         self.w_b_0 = FCBlock(1, 1, bias=True)

#     def forward(self, style_prototype, speakers, mel, mask):

#         max_len = mel.shape[1]
#         slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

#         x = self.fc_1(mel)

#         # Spectral Processing
#         for _, layer in enumerate(self.spectral_stack):
#             x = layer(x)

#         # Temporal Processing
#         for _, layer in enumerate(self.temporal_stack):
#             residual = x
#             x = layer(x)
#             x = residual + x

#         # Multi-head self-attention
#         for _, layer in enumerate(self.slf_attn_stack):
#             residual = x
#             x, _ = layer(
#                 x, x, x, mask=slf_attn_mask
#             )
#             x = residual + x

#         # Final Layer
#         x = self.fc_2(x) # [B, T, H]

#         # Temporal Average Pooling, h(x)
#         x = torch.mean(x, dim=1, keepdim=True) # [B, 1, H]

#         # Output Computation
#         s_i = style_prototype(speakers) # [B, H]
#         V = self.V(s_i).unsqueeze(2) # [B, H, 1]
#         o = torch.matmul(x, V).squeeze(2) # [B, 1]
#         o = self.w_b_0(o).squeeze() # [B,]

#         return o