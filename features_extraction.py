from config import config
from transformers import AutoTokenizer, SiglipProcessor, SiglipModel, GPTNeoModel
import torch
import torch.nn as nn
from data_processing import build_dataloaders
from xlstm_decoder import xLSTM

class ImageEmbedding(nn.Module):
    def __init__(self, output_size=config.d_model):
        super(ImageEmbedding, self).__init__()
        self.process = SiglipProcessor.from_pretrained(config.IMG_DIR)
        self.model = SiglipModel.from_pretrained(config.IMG_DIR)

    def forward(self, image):
        inputs = self.process(images=image, return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        return outputs

class QuesEmbedding(nn.Module):
    def __init__(self, input_size=config.d_model, output_size=config.d_model):
        super(QuesEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.text_model = GPTNeoModel.from_pretrained(config.TEXT_DIR)
        self.xlstm = xLSTM(
            input_size=input_size,
            hidden_size=output_size,
            num_heads=8,
            layers=['m']
        )


    def forward(self, ques):
        if isinstance(ques, tuple):
            ques = list(ques)
        elif isinstance(ques, str):
            ques = [ques]

        tokenized_input = self.tokenizer(
            ques,
            return_tensors='pt',
            padding='max_length',
            max_length=config.MAX_LEN,
            truncation=True
        )

        ques = self.text_model(**tokenized_input.to(config.DEVICE)).last_hidden_state

        output, _ = self.xlstm(ques)
        return torch.mean(output, dim=1)
    
class AnsEmbedding(nn.Module):
    def __init__(self, input_size=config.d_model):
        super(AnsEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text_model = GPTNeoModel.from_pretrained(config.TEXT_DIR)
    def forward(self, ans):
        if isinstance(ans, tuple):
            ans = list(ans)
        elif isinstance(ans, str):
            ans = [ans]
        tokenized_input = self.tokenizer(
            ans,
            return_tensors='pt',
            padding='max_length',
            max_length=config.MAX_LEN,
            truncation=True,
            return_attention_mask=False
        )
        tokenized_input = {k: v.to(config.DEVICE) for k, v in tokenized_input.items()}
        # CrossEntropyLoss needs class indices [batch, seq_len] as target.
        return tokenized_input['input_ids']
    
if __name__=="__main__":
    image_model = ImageEmbedding(output_size=config.d_model).to(config.DEVICE)
    ques_model = QuesEmbedding(output_size=config.d_model).to(config.DEVICE)
    ans_model = AnsEmbedding().to(config.DEVICE)

    train_loader, _, _ = build_dataloaders()
    for batch in train_loader:
        images, questions, answers = batch
        if torch.cuda.is_available():
            images = images.to(config.DEVICE)
            questions = questions
            answers = answers

        with torch.no_grad():
            image_embeddings = image_model(images)
            ques_embeddings = ques_model(questions)
            ans_tokens = ans_model(answers)
        break

    image_embeddings = image_embeddings.reshape(config.BATCH_SIZE, config.d_model, -1).permute(0, 2, 1)
    ques_embeddings = ques_embeddings.unsqueeze(1)
    
    print(f"image embeddings size: {image_embeddings.size()}")
    print(f"question embeddings size: {ques_embeddings.size()}")
    print(f"answer token size: {ans_tokens.size()}")