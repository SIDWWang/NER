import config
from model import BertNER
import torch
from transformers import BertTokenizer

def get_entity(sentence, config):


    id2label = config.id2label
    words = []

    tokenizer = BertTokenizer.from_pretrained(config.roberta_model, do_lower_case=True)
    for token in sentence:
        words.append(tokenizer.tokenize(token))
    words = ['[CLS]'] + [item for token in words for item in token]

    sent = tokenizer.convert_tokens_to_ids(words)
    sent_token = [0] + [1]*(len(sent)-1)
    sent_mask = [True] * len(sent)
    label_mask = [True] * (len(sent)-1)

    sent = torch.tensor([sent],dtype=torch.int64).to(config.device)
    sent_token = torch.tensor([sent_token],dtype=torch.int64).to(config.device)
    sent_mask = torch.tensor([sent_mask],dtype=torch.bool).to(config.device)
    label_mask = torch.tensor([label_mask],dtype=torch.bool).to(config.device)

    model = BertNER.from_pretrained(config.model_dir)
    model.to(config.device)

    batch_output = model((sent, sent_token),token_type_ids=None, attention_mask=sent_mask)[0]
    batch_output = model.crf.decode(batch_output, mask=label_mask)

    return [[id2label.get(idx) for idx in indices] for indices in batch_output]

if __name__ == "__main__":
    sentence = '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，'
    print(get_entity(sentence, config))