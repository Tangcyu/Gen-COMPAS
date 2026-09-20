# Trp-cage demo: trajectory preparation

Before running the `RiteWeight` stage, separately save protein-only, hydrogen-free trajectories using the VMD atom selection `protein and noh`. Export `unbias.A.dcd` and `unbias.B.dcd` as `Unbiased.noh.A.dcd` and `Unbiased.noh.B.dcd`, respectively, in their corresponding simulation directories. The demo configuration expects `Unbiased.noh.[AB].dcd` with `Dataset/trp_noh.psf`; atom counts and ordering must match this topology.

One approach is to use a VMD Tcl script to extract and write the selected atoms, running independent VMD processes in parallel across trajectories. Preserve frame order and count so that each trajectory remains aligned with its `unbias.[AB].colvars.traj` file.

Alternatively, adapt RiteWeight's trajectory-reading setup to read `unbias.[AB].dcd` directly, updating the input pattern, full-system topology, and atom selections consistently. Reading the full trajectories makes cache construction much slower, so preparing the reduced trajectories in advance is recommended.

## Acknowledgments

We thank Dr. Li Qiushi for helping identify this issue.
