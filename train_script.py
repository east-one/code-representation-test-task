import torch
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

torch.manual_seed(42)

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

lib_to_id = {}
id_to_lib = {}
unique_labels = np.unique(data.lib)

for i, lib in enumerate(unique_labels):
    lib_to_id[lib] = i
    id_to_lib[i] = lib


def process_labels(labels, label_to_id):
    new_ids = [label_to_id[label] for label in labels]
    return torch.tensor(new_ids)


labels_data = data.lib.values

# train, val = train_test_split(data, test_size=0.3, stratify=labels_data)
train = data

texts_train = train.title.values
texts_test = test.title.values

labels_train = train.lib.values

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def preprocess(texts):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoding = tokenizer.encode_plus(text,
                      add_special_tokens=True,
                      max_length=32,
                      pad_to_max_length=True,
                      return_attention_mask=True,
                      return_tensors='pt'
                        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


train_ids, train_masks = preprocess(texts_train)

test_ids, test_masks = preprocess(texts_test)

train_labels = process_labels(labels_train, lib_to_id)


train_dataset = TensorDataset(train_ids, train_masks, train_labels)

test_dataset = TensorDataset(test_ids, test_masks)

batch_size = 64

train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size
        )

test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size
        )

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(lib_to_id),
    output_attentions=False,
    output_hidden_states=False,
)

optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  eps=1e-8
                )

epochs = 4

device = 'cuda:0' if torch.cuda.is_available else 'cpu'
model.to(device)

for epoch in range(epochs):
    total_loss = 0
    model.train()
    for text_ids, masks, labels in train_dataloader:
        text_ids = text_ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        loss, logits = model(text_ids,
                             attention_mask=masks,
                             labels=labels).values()
        loss.backward()
        optimizer.step()

predictions = []

for text_ids, masks in test_dataloader:
    text_ids = text_ids.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        logits = model(text_ids,
                       token_type_ids=None,
                       attention_mask=masks)['logits']

    preds = torch.argmax(logits, dim=1)
    predictions.append(preds)

predictions = torch.cat(predictions).cpu().numpy()

predictions_text = np.array([id_to_lib[pred] for pred in predictions])
sub_np = np.stack([test.id, predictions_text]).T
submission = pd.DataFrame(data=sub_np, columns=['id', 'lib'])
submission.to_csv('submission.csv', index=False)
