import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

label2id = {
    'cs':    0,
    'econ':  1,
    'eess':  2,
    'math':  3,
    'q-bio': 4,
    'q-fin': 5,
    'stat':  6,
    'phys':  7,
}

id2label = {v: k for k, v in label2id.items()}

class MultilabelClassificationHead(nn.Module):
  def __init__(self, emb_dim, num_labels, dropout_p):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(emb_dim, 512),
        nn.LayerNorm(normalized_shape=512),
        nn.GELU(),
        nn.Dropout(dropout_p),
        nn.Linear(512, 512),
        nn.LayerNorm(normalized_shape=512),
        nn.GELU(),
        nn.Dropout(dropout_p),
        nn.Linear(512, 512),
        nn.LayerNorm(normalized_shape=512),
        nn.GELU(),
        nn.Dropout(dropout_p),
        nn.Linear(512, num_labels)
    )
    self.criterion = nn.BCEWithLogitsLoss()

  def forward(self, x):
    return self.model(x)


class Prediction:
  def __init__(self, emb_dim, num_labels, dropout_p, path_to_model, threshold=0.5, id2label=id2label):
    self.id2label = id2label

    self.model = MultilabelClassificationHead(emb_dim=emb_dim, num_labels=num_labels, dropout_p=dropout_p)
    self.model.load_state_dict(torch.load(path_to_model, weights_only=True))
    self.model.eval()
    self.model.to('cpu')

    self.encoder = SentenceTransformer("malteos/scincl")
    self.encoder.to('cpu')
    self.threshold = threshold

  @torch.no_grad
  def get_labels(self, title: str, abstract: str):
    title_abstract = title + ' [SEP] ' + abstract
    embedding = self.encoder.encode(title_abstract, convert_to_tensor=True)
    output = self.model(embedding).sigmoid().detach().cpu().numpy()
    labels_mask = np.argwhere(output >= self.threshold).flatten().tolist()

    labels = [self.id2label[label] for label in labels_mask]
    return labels