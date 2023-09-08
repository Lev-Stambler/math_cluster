import os, sys
sys.path.append(os.getcwd())

USE_FAKE = False
if not USE_FAKE:
	sys.path.append(os.path.join(os.getcwd(), "exllama"))
	import exllama_lang

import json
import cluster
from langchain.llms.base import LLM
from langchain.llms import fake
from typing import List, Tuple
import numbers
import numpy as np
import asyncio
import copy
import langchain
import numpy.typing as npt
import custom_types

STOP_DEFAULT_TOKENS = ["### Instruction", "\n"]

def get_theorems_in_group(embeddings: custom_types.Embeddings, labels: npt.NDArray, group_idx: int, max_size=None, random=True):
	s = [embeddings[i][0] for i in np.where(labels == group_idx)[0]]
	if max_size is None or len(s) <= max_size: 
		return s
	if not random:
		return s[:max_size]
	c = np.random.choice(np.arange(len(s)), size=max_size, replace=False)
	return [s[i] for i in c]


# From https://github.com/hichamjanati/pyldpc, but modified to make a square matrix
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def parity_check_matrix(n_code, d_v, d_c, seed=None):
    """
    Build a regular Parity-Check Matrix H following Callager's algorithm.

    Parameters
    ----------
    n_code: int, Length of the codewords.
    d_v: int, Number of parity-check equations including a certain bit.
        Must be greater or equal to 2.
    d_c: int, Number of bits in the same parity-check equation. d_c Must be
        greater or equal to d_v and must divide n.
    seed: int, seed of the random generator.

    Returns
    -------
    H: array (n_equations, n_code). LDPC regular matrix H.
        Where n_equations = d_v * n / d_c, the total number of parity-check
        equations.

    """
    rng = check_random_state(seed)

    if d_v <= 1:
        raise ValueError("""d_v must be at least 2.""")

    # if d_c <= d_v:
    #     raise ValueError("""d_c must be greater than d_v.""")

    if n_code % d_c:
        raise ValueError("""d_c must divide n for a regular LDPC matrix H.""")

    n_equations = (n_code * d_v) // d_c

    block = np.zeros((n_equations // d_v, n_code), dtype=int)
    H = np.empty((n_equations, n_code))
    block_size = n_equations // d_v

    # Filling the first block with consecutive ones in each row of the block

    for i in range(block_size):
        for j in range(i * d_c, (i+1) * d_c):
            block[i, j] = 1
    H[:block_size] = block

    # reate remaining blocks by permutations of the first block's columns:
    for i in range(1, d_v):
        H[i * block_size: (i + 1) * block_size] = rng.permutation(block.T).T
    H = H.astype(int)
    return H

parity_check_matrix(16, 4, 4)

async def local_neighbor_with_descr_labels(thms_node: List[str], descr_node: str, thms_local: List[List[str]], descr_thms_local: List[str], llm: LLM):
    merged_non_prim = [f"Description: {descr_thms_local[i]}\n" + "\n".join(t) for i, t in enumerate(thms_local)] if descr_thms_local[0] != "" \
        else ["\n".join(t) for t in thms_local]
    joined_non_prim = "\n\n".join(merged_non_prim)

    joined_prim = (f"Description: {descr_node}" + "\n" if descr_node != "" else "") + "\n".join(thms_node)
		# TODO: Fix up so that we can use any format of promtp
    prompt = f"""### Instruction:

You will be given a set of non-primary theorems and a set of primary theorems{ " as well as descriptions for both" if descr_node[0] != "" else ""}. Can you briefly discuss the main focus of the primary theorems and how it differs from the remaining theorems?
Assume that when the descriptions are given for the non-primary theorems, that they do not reference the set of primary theorems at all.

Non-primary theorems: "{joined_non_prim}"

Primary theorems: "{joined_prim}"

### Response:
"""
    r = await llm.agenerate([prompt], stop=STOP_DEFAULT_TOKENS)
    return r.generations[0][0].text
   


class RunParams:
	n_clusters: int
	seed: int
	n_rounds: int
	model_name: str
	max_sample_size: int
	descr: str
	cluster_cluster_deg: int

	def __init__(self, n_clusters: int, seed: int, n_rounds: int, model_name: str, max_sample_size: int, cluster_cluster_deg=3, descr: str="default"):
		self.n_clusters = n_clusters
		self.seed = seed
		self.n_rounds = n_rounds
		self.model_name = model_name
		self.max_sample_size = max_sample_size
		self.descr = descr
		self.cluster_cluster_deg = cluster_cluster_deg
	
	def to_dict(self):
		return self.__dict__

class RunData:
	# The outer list for each round. The middle list is the list of messages per node, the inner list is the specific messages to its neighbor
	rounds: List[
		List[List[str]]
	] = []
	parity_check_matrix: npt.NDArray = None
	params: RunParams
	completed_rounds = 0
	cluster_labels: npt.NDArray

	def __init__(self, cluster_labels: npt.NDArray, parity_check_matrix: npt.NDArray, params: RunParams) -> None:
		self.cluster_labels = cluster_labels
		self.parity_check_matrix = parity_check_matrix
		self.params = params

	def to_dict(self):
		return {
			"rounds": self.rounds,
			"parity_check_matrix": self.parity_check_matrix.tolist(),
			"params": self.params.to_dict(),
			"completed_rounds": self.completed_rounds,
			"cluster_labels": self.cluster_labels.tolist()
		}
	
	def from_dict(d: dict):
		r = RunData(d["cluster_labels"], np.array(d["parity_check_matrix"]), RunParams(**d["params"]))
		r.rounds = d["rounds"]
		r.completed_rounds = d["completed_rounds"]
		return r


def get_data_file_name(params: RunParams):
	return f"data_store/llm_bp_clustersize_{params.n_clusters}__seed_{params.seed}_{params.model_name}__descr_{params.descr}.json"

def save_dict(params: RunParams, d: RunData):
	json.dump(d.to_dict(), open(get_data_file_name(params), "w"))

async def llm_bp(embeddings: custom_types.Embeddings, llm: LLM, data: RunData):
	"""
		Run bp-ish.... TODO: document

		Because K-clustering initializes the clusters randomly, we can assume that we each cluster is distinct from each other (i.e. cluster 1 and 2 are not correlated any differently than cluster 1 and 69)
		Thus, we say that if i < params.n_clusters / 2, then i is a data bit, and if i >= params.n_clusters / 2, then i is a parity check bit
  """
	assert data.parity_check_matrix is not None, "Must have a parity check matrix"

	H_cluster: npt.NDArray = data.parity_check_matrix
	params = data.params

	# For simplicity, we will use an adjacency matrix for now. Later we can flatten this data-structure to make it cheaper
	if data.rounds is not None and len(data.rounds) > 0:
		primary_focuses_msgs_last = data.rounds[-1]
	else:
		data.rounds = []
		primary_focuses_msgs_last = [["" for _ in range(params.n_clusters)] for _ in range(params.n_clusters)]

	async def pc_to_bit(i):
		assert i >= params.n_clusters / 2, "Must be a parity check bit"
		# H_ind = np.where(check_inds == i)[0][0]
		pc_ind = i - int(params.n_clusters / 2)
		neighbors = np.where(H_cluster[pc_ind, :] == 1)[0]
		# cluster_neighbor_inds = bit_inds[neighbors]
		p = ["" for _ in range(params.n_clusters)]

		for neighbor_ind in range(params.cluster_cluster_deg):
			neighbors_without_neighbor = np.delete(neighbors, neighbor_ind)
			
			ret = await local_neighbor_with_descr_labels(get_theorems_in_group(embeddings, data.cluster_labels, i, max_size=params.max_sample_size), primary_focuses_msgs_last[i],
																[get_theorems_in_group(embeddings, data.cluster_labels, j, max_size=params.max_sample_size)
																for j in neighbors_without_neighbor], [primary_focuses_msgs_last[j][i] for j in neighbors_without_neighbor], llm=llm)
			
			# primary_focuses_msgs[i][cluster_neighbor_inds[neighbor_ind]] = ret
			p[neighbors[neighbor_ind]] = ret
		return (i, p)
	
	async def bit_to_pc(i):
		assert i < params.n_clusters / 2, "Must be a data bit"
		# We offset by n_clusters / 2 because we want to start at the parity check bits
		offset = int(params.n_clusters / 2) 
		neighbors = offset + np.where(H_cluster[:, i] == 1)[0]
		# print("Neighbors", neighbors, H_cluster.shape)
		p = ["" for _ in range(params.n_clusters)]

		for neighbor_ind in range(params.cluster_cluster_deg):
			neighbors_without_neighbor = np.delete(neighbors, neighbor_ind)
			
			ret = await local_neighbor_with_descr_labels(get_theorems_in_group(embeddings, data.cluster_labels, i, max_size=params.max_sample_size), primary_focuses_msgs_last[i],
																[get_theorems_in_group(embeddings, data.cluster_labels, j, max_size=params.max_sample_size)
																for j in neighbors_without_neighbor], [primary_focuses_msgs_last[j][i] for j in neighbors_without_neighbor], llm=llm)
			p[neighbors[neighbor_ind]] = ret
		return (i, p)

	for round_numb in range(data.completed_rounds, params.n_rounds):
		print(f"Starting BP Round {round_numb + 1} out of {params.n_rounds}")
		SKIP = 1
		tmp = []
		for i in range(0, params.n_clusters, SKIP):
			tasks = []
			for skip in range(min(SKIP, params.n_clusters - i)):
				# Then we have a bit
				if i + skip < params.n_clusters / 2:
					tasks.append(bit_to_pc(i + skip))
				else:
					tasks.append(pc_to_bit(i + skip))
				print("Appended cluster", i + skip)
			rets = await asyncio.gather(*tasks)
			rets_str = "\n\n".join(["\n".join(list(filter(lambda x: x != "", r[1]))) for r in rets])
			print(f"\nReturns for BP round {round_numb + 1} out of {params.n_rounds} and cluster {i} to {i + SKIP - 1} (inclusive): {rets_str}\n")
			tmp = tmp + (rets)

			

		tmp.sort(key=lambda x: x[0])
		# sorted = np.array(tmp)[np.argsort(np.array([a[0] for a in tmp]))]
		primary_focuses_msgs =  [a[1] for a in tmp]
		print(primary_focuses_msgs)

		data.rounds.append(primary_focuses_msgs)
		data.completed_rounds += 1

		save_dict(params, data)
		primary_focuses_msgs_last = copy.deepcopy(primary_focuses_msgs)
	return data


async def run_bp_labeling(n_clusters: int, params: RunParams, thm_embs: custom_types.Embeddings, llm: LLM):
	"""
		Runs BP on the given theorems, returning the labels for each theorem
	"""
	assert n_clusters % 2 == 0, "Must have an even number of clusters"
	_, labels, _unique_label_set = cluster.cluster(thm_embs, n_clusters) # Cluster with the number of dimensions equal to the number of embeddings
	H = parity_check_matrix(int(n_clusters / 2), params.cluster_cluster_deg, params.cluster_cluster_deg)
	data = RunData(cluster_labels=labels, parity_check_matrix=H, params=params)
	await llm_bp(thm_embs, llm, data)

async def run_from_file(thm_embs: custom_types.Embeddings, file_path: str, llm: LLM, n_rounds = None):
	"""
		Runs BP on the given theorems, returning the labels for each theorem
	"""
	_data = json.load(open(file_path, "r"))
	data = RunData.from_dict(_data)
	# params = RunParams(**_data["params"])
	if n_rounds is not None:
		params.n_rounds = n_rounds
	# data = {}
	# data["params"] = params
	# data["cluster_labels"] = np.array(_data["cluster_labels"])
	# data["parity_check_matrix"] = np.array(_data["parity_check_matrix"])
	# data = RunData(**data)
	# data.completed_rounds = _data["completed_rounds"]

	await llm_bp(thm_embs, llm, data)

if __name__ == "__main__":
	# await run_bp_labeling(16, thm_embs, llm)
	loop = asyncio.get_event_loop()
	file_path = f"data_store/embeddings_seed_69420_size_10000.json"
	embeddings: List[Tuple[str, List[float]]] = json.load(open(file_path, "r"))
	# thm_embs = 
	n_clusters = 24
	bp_rounds = 2
	params = RunParams(n_clusters=n_clusters, seed=69_420, n_rounds=bp_rounds, model_name="exlamma-luban-13b-4bit" if not USE_FAKE else "FAKE", max_sample_size=20, cluster_cluster_deg=3)
	if USE_FAKE:
		llm = fake.FakeListLLM(responses=["hello " * 30] * 1_000)
	else:
		llm = exllama_lang.ExLLamaLLM(model_dir="../../Luban-13B-GPTQ", max_response_tokens=1_000, max_seq_len=4_096, temperature=0.1, beams=3, beam_length=10)
	if False:
		loop.run_until_complete(run_bp_labeling(24, embeddings, llm))
	if True:
		new_n_rounds = 5
		loop.run_until_complete(run_from_file(embeddings, get_data_file_name(params), llm, n_rounds=new_n_rounds))
