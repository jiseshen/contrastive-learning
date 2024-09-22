import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def contrastive_eval(model, loader_a, loader_b, batch_size):
  model.eval()
  all_embeddings_a = []
  all_embeddings_b = []
  with torch.no_grad():  
    for batch in dataloader_a:
      embeddings = model(batch)
      all_embeddings_a.append(embeddings)

    for batch in dataloader_b:
      embeddings = model(batch)
      all_embeddings_b.append(embeddings)

  all_embeddings_a = torch.cat(all_embeddings_a, dim=0)
  all_embeddings_b = torch.cat(all_embeddings_b, dim=0)
  sim_matrix = F.cosine_similarity(all_embeddings_a, all_embeddings_b, dim=-1)
  retrieval_metric = {k: topk_accuracy(sim_matrix, k) for k in [1, 5, 10]}
  return retrieval_metric
  

def topk_accuracy(sim_matrix, k=10):
  _, topk_indices_a = torch.topk(similarity_matrix, k, dim=1)
  _, topk_indices_b = torch.topk(similarity_matrix, k ,dim=0)
  correct_a = 0
  correct_b = 0
  for i in range(sim_matrix.size(0)):
    if i in topk_indices[i].tolist():
      correct_a += 1
  for i in range(sim_matrix.size(1)):
    if i in topk_indices[:,i].tolist():
      correct_b += 1
  return correct_a / sim_matrix.size(0), correct_b / sim_matrix.size(1)
