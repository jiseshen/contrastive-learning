import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def contrastive_eval(model, loader_a, loader_b, batch_size):
  model.eval()
  all_emb_a = []
  all_emb_b = []
  with torch.no_grad():  
    for batch in dataloader_a:
      emb = model.emb_a(batch)
      all_emb_a.append(embeddings)

    for batch in dataloader_b:
      emb = model.emb_b(batch)
      all_emb_b.append(embeddings)

  all_emb_a = torch.cat(all_embeddings_a, dim=0)
  all_emb_b = torch.cat(all_embeddings_b, dim=0)
  retrieval_metric = {k: topk_accuracy(all_emb_a, all_emb_b, k) for k in [1, 5, 10]}
  return retrieval_metric
  

def topk_accuracy(emb_a, emb_b, k=10):
  _, topk_indices_a = torch.topk(similarity_matrix, k, dim=1)
  _, topk_indices_b = torch.topk(similarity_matrix, k ,dim=0)
  correct_a = 0
  correct_b = 0
  for i in range(emb_a.size(0)):
    _, topk_indices = torch.topk(F.cosine_similarity(emb_a[i], emb_b), k)
    if i in topk_indices.tolist():
      correct_a += 1
  for i in range(sim_matrix.size(1)):
    _, topk_indices = torch.topk(F.cosine_similarity(emb_b[i], emb_a), k)
    if i in topk_indices.tolist():
      correct_b += 1
  return correct_a / sim_matrix.size(0), correct_b / sim_matrix.size(1)
