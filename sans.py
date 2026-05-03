from config import config
import torch.nn as nn
import torch.nn.functional as F
import torch
from features_extraction import ImageEmbedding, QuesEmbedding
from data_processing import build_dataloaders

class StackedAttentionNets(nn.Module):
    def __init__(self, d=768, k=768):
        super(StackedAttentionNets, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        self.dropout = nn.Dropout(0.3)
        self.ff_attention = nn.Linear(k, 1)
    def forward(self, vi, vq):
        hi = self.ff_image(vi)
        hq = self.ff_ques(vq)
        ha = F.gelu(hi + hq)
        ha = self.dropout(ha)
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha, dim=1)
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq.squeeze(1)
        return u
    
if __name__=="__main__":
    image_model = ImageEmbedding(output_size=config.d_model).to(config.DEVICE)
    ques_model = QuesEmbedding(output_size=config.d_model).to(config.DEVICE)

    train_loader, _, _ = build_dataloaders()
    for batch in train_loader:
        images, questions, answers = batch
        if torch.cuda.is_available():
            images = images.to(config.DEVICE)
            questions = questions

        with torch.no_grad():
            image_embeddings = image_model(images)
            ques_embeddings = ques_model(questions)
        break

    image_embeddings = image_embeddings.reshape(config.BATCH_SIZE, config.d_model, -1).permute(0, 2, 1)
    ques_embeddings = ques_embeddings.unsqueeze(1)

    san_model = StackedAttentionNets(d=config.d_model, k=768).to(config.DEVICE)
    img_text_att = san_model(image_embeddings.to(config.DEVICE), ques_embeddings.to(config.DEVICE))
    
    print(f"features combined with SANs size: {img_text_att.size()}")