"""Exercise the real training/sampling checkpoint boundary."""
import pytest

torch = pytest.importorskip("torch")
md = pytest.importorskip("mdtraj")

from common.diffusion_sample import setup_model_and_diffusion
from common.diffusion_train import _load_warm_start_weights
from utils.model import DiffusionModel


@pytest.mark.parametrize("legacy", [False, True])
def test_sampling_requires_trained_segment_parameters(tmp_path, legacy):
    topology = md.Topology()
    for _ in range(2):
        chain = topology.add_chain()
        residue = topology.add_residue("MOL", chain)
        topology.add_atom("C", md.element.carbon, residue)
    top_path = tmp_path / "topology.pdb"
    md.Trajectory(torch.zeros(1, 2, 3).numpy(), topology).save_pdb(top_path)
    model_cfg = dict(node_feature_dim=8, time_embedding_dim=8, hidden_dim=8,
                     num_schnet_layers=1, num_gat_layers=1,
                     residue_attn_heads=2, k_neighbors=1,
                     num_segment_layers=1, segment_distance_rbf=4)
    model = DiffusionModel(2, topology, ["C", "C"], **model_cfg)
    state = model.state_dict()
    if legacy:
        state = {k: v for k, v in state.items() if not k.startswith("segment_")}
    checkpoint = tmp_path / "model.pt"
    torch.save(state, checkpoint)
    torch.save(torch.zeros(3), tmp_path / "coord_mean.pt")
    torch.save(torch.ones(3), tmp_path / "coord_std.pt")
    config = dict(inference=dict(checkpoint=str(checkpoint)),
                  data=dict(topology_path=str(top_path)), model=model_cfg,
                  diffusion=dict(timesteps=10, beta_schedule="linear"))
    if legacy:
        # Old learned weights remain usable for training the additional layers.
        _load_warm_start_weights(model, str(checkpoint), torch.device("cpu"))
        with pytest.raises(ValueError, match="Warm-start training"):
            setup_model_and_diffusion(config, torch.device("cpu"))
    else:
        setup_model_and_diffusion(config, torch.device("cpu"))
