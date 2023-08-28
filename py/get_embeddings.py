import os
from time import sleep
from typing import List, Union
import numpy as np
import json
import torch
from lean_dojo import LeanGitRepo, trace
# from lean_dojo.data_extraction.lean import info_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5EncoderModel

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean3-retriever-byt5-small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def encode(model : T5EncoderModel, s: Union[str, List[str]]) -> torch.Tensor:
    """Encode texts into feature vectors."""
    if isinstance(s, str):
        s = [s]
        should_squeeze = True
    else:
        should_squeeze = False
    tokenized_s = tokenizer(s, return_tensors="pt", padding=True).to(device)
    hidden_state = model(tokenized_s.input_ids).last_hidden_state
    lens = tokenized_s.attention_mask.sum(dim=1)
    features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
    del tokenized_s
    if should_squeeze:
      features = features.squeeze()
    return features


def main(sample_size=10_000, seed=69_420):
  np.random.seed(seed)
  # TODO: figure the below out...
  repo = LeanGitRepo("https://github.com/Lev-Stambler/mathlib-clustering", "6069641a14213014f6e92fad3280e3a5f524497c")
  # exit()
  # hmmm...
  # sleep(2)
  # repo = LeanGitRepo("https://github.com/yangky11/lean4-example", "7d711f6da4584ecb7d4f057715e1f72ba175c910")

  # repo = LeanGitRepo("https://github.com/yangky11/lean-example", "5a0360e49946815cb53132638ccdd46fb1859e2a")
  # repo.is_lean3 = True
  dst_dir_name = "./traced_boy"
  # Check if the repo dir exists
  # if os.path.exists(dst_dir_name):
  # 	# Remove the repo dir
  # 	os.system("rm -rf {}".format(dst_dir_name))
  file_path = f"data_store/embeddings_seed_{seed}_size_{sample_size}.json"
  if os.path.exists(file_path):
    j = json.load(open(file_path, "r"))
    n_compl = len(j)
  else: n_compl = 0
  repo = trace(repo, dst_dir=dst_dir_name)
  thms = repo.get_traced_theorems()

  print("Theorem numbs: {}. N proved {}. Sample size".format(len(thms), n_compl, sample_size))
  sample_idxs = np.random.choice(len(thms), sample_size, replace=False)
  model = T5EncoderModel.from_pretrained("kaiyuy/leandojo-lean3-retriever-byt5-small").to(device)
  chunk_size = 64
  rets = []
  for i in range(n_compl, sample_size, chunk_size):
    thm_chunk = [thms[j] for j in sample_idxs[i:min(i+chunk_size, len(thms))]]
    thm_chunk = [t.ast.text.split(":=")[0].split("\n|")[0].strip() for t in thm_chunk]
    embeddings = encode(model, thm_chunk)
    print(embeddings.shape)
    emb_list = [(thm_chunk[i], embs) for i, embs in enumerate(embeddings.to('cpu').tolist())]
    del embeddings
    torch.cuda.empty_cache()

    rets += emb_list
    json.dump(rets, open(file_path, "w"))
  # TODO: write to JSON

if __name__ == "__main__":
  main()