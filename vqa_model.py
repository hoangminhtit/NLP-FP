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

class VQAModel(nn.Module):
    def __init__(self, vocab_size=len(vocab), output_size=768, d_model=768,
                 num_heads=8, hidden_size=768,num_att_layers=4):
        super(VQAModel, self).__init__()
        self.image_model = ImageEmbedding(output_size=output_size).to(config.DEVICE)
        self.ques_model = QuesEmbedding(output_size=output_size).to(config.DEVICE)
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

    def forward(self, images, questions, max_len=config.MAX_LEN):
        image_embeddings = self.image_model(images.to(config.DEVICE))
        batch_size = images.size(0)
        image_embedds = image_embeddings.reshape(batch_size, 768, -1).permute(0, 2, 1)
        ques_embeddings = self.ques_model(questions)
        ques_embedds = ques_embeddings.unsqueeze(1)
        for att_layer in self.san_model:
            att_embedds = att_layer(image_embedds.to(config.DEVICE), ques_embedds.to(config.DEVICE))
        y = att_embedds.unsqueeze(1).expand(-1, max_len, -1)
        out, _ = self.decoder(y)
        output_logits = self.mlp(out)
        return output_logits # [32, 64, 50257]