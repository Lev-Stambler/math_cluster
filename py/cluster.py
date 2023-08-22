import os
from lean_dojo import LeanGitRepo, trace

def main():
	repo = LeanGitRepo("https://github.com/yangky11/lean-example", "5a0360e49946815cb53132638ccdd46fb1859e2a")
	# repo.is_lean4 = True
	dst_dir_name = "traced/traced_lean-example"
	# Check if the repo dir exists
	if not os.path.exists(dst_dir_name):
		trace(repo, dst_dir=dst_dir_name)


if __name__ == "__main__":
	main()