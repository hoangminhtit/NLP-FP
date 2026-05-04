from config import config
import torch.nn as nn
import torch
from transformers import AutoTokenizer
from features_extraction import ImageEmbedding, QuesEmbedding
from sans import StackedAttentionNets
from xlstm_decoder import xLSTM

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)
vocab = tokenizer.get_vocab()

class DualGatedFusion(nn.Module):
    def __init__(self, d_model=768, dropout=0.1):
        super(DualGatedFusion, self).__init__()
        self.image_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)
        self.image_gate = nn.Linear(d_model * 2, d_model)
        self.text_gate = nn.Linear(d_model * 2, d_model)
        self.norm_img = nn.LayerNorm(d_model)
        self.norm_txt = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_tokens, question_token):
        # image_tokens: [B, N, D], question_token: [B, 1, D]
        question_expanded = question_token.expand(-1, image_tokens.size(1), -1)
        image_for_gate = torch.cat([image_tokens, question_expanded], dim=-1)
        text_for_gate = torch.cat([question_expanded, image_tokens], dim=-1)

        g_img = torch.sigmoid(self.image_gate(image_for_gate))
        g_txt = torch.sigmoid(self.text_gate(text_for_gate))

        fused_image_tokens = g_img * self.image_proj(image_tokens) + (1.0 - g_img) * self.text_proj(question_expanded)
        fused_text_tokens = g_txt * self.text_proj(question_expanded) + (1.0 - g_txt) * self.image_proj(image_tokens)

        fused_image_tokens = self.norm_img(self.dropout(fused_image_tokens))
        fused_text_token = self.norm_txt(self.dropout(fused_text_tokens.mean(dim=1, keepdim=True)))
        return fused_image_tokens, fused_text_token

class VQAModel(nn.Module):
    def __init__(self, vocab_size=len(vocab), output_size=768, d_model=768,
                 num_heads=8, hidden_size=768,num_att_layers=4, use_dual_gating=False):
        super(VQAModel, self).__init__()
        self.use_dual_gating = use_dual_gating
        self.image_model = ImageEmbedding(output_size=output_size).to(config.DEVICE)
        self.ques_model = QuesEmbedding(output_size=output_size).to(config.DEVICE)
        self.dual_gated_fusion = DualGatedFusion(d_model=d_model).to(config.DEVICE)
        self.san_model = nn.ModuleList(
                        [StackedAttentionNets(d=d_model, k=768) for _ in range(num_att_layers)]).to(config.DEVICE)

        self.decoder = xLSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_heads=num_heads,
            layers=['m']
        ).to(config.DEVICE)

        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size)
        )

    def _extract_image_tensor(self, image_outputs):
        """
        Normalize HuggingFace vision outputs to a tensor.
        """
        if torch.is_tensor(image_outputs):
            return image_outputs

        if hasattr(image_outputs, "last_hidden_state") and image_outputs.last_hidden_state is not None:
            return image_outputs.last_hidden_state

        if hasattr(image_outputs, "pooler_output") and image_outputs.pooler_output is not None:
            return image_outputs.pooler_output

        raise TypeError(
            f"Unsupported image output type from encoder: {type(image_outputs)}"
        )

    def forward(self, images, questions, max_len=config.MAX_LEN):
        image_outputs = self.image_model(images.to(config.DEVICE))
        image_embeddings = self._extract_image_tensor(image_outputs)
        batch_size = images.size(0)

        if image_embeddings.dim() == 3:
            # [batch, seq_len, hidden]
            image_embedds = image_embeddings
        else:
            # [batch, hidden] -> [batch, 1, hidden]
            image_embedds = image_embeddings.reshape(batch_size, 768, -1).permute(0, 2, 1)

        ques_embeddings = self.ques_model(questions)
        ques_embedds = ques_embeddings.unsqueeze(1)

        if self.use_dual_gating:
            image_embedds, ques_embedds = self.dual_gated_fusion(
                image_embedds.to(config.DEVICE),
                ques_embedds.to(config.DEVICE),
            )

        for att_layer in self.san_model:
            att_embedds = att_layer(image_embedds.to(config.DEVICE), ques_embedds.to(config.DEVICE))
        y = att_embedds.unsqueeze(1).expand(-1, max_len, -1)
        out, _ = self.decoder(y)
        output_logits = self.mlp(out)
        return output_logits # [32, 64, 50257]