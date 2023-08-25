import Lake
open Lake DSL

package «leancorr» {
  -- add any package configuration options here
}

lean_lib «leanex»  {
  -- add package configuration options here
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «LeanExp» {
}
