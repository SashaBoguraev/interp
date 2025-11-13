import torch
from pyvene.models.intervention_utils import _do_intervention_by_swap
from pyvene.models.interventions import TrainableIntervention, DistributedRepresentationIntervention
import normflows as nf

from interventions.layers import StretchLayer, RealNVPLayer, RevNetLayer, UnconstrainedLayer
import os


class UnconstrainedIntervention(TrainableIntervention, DistributedRepresentationIntervention):
    """Unconstrained linear intervention."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        unconstrained_layer = UnconstrainedLayer(self.embed_dim)
        self.unconstrained_layer = unconstrained_layer


    def forward(self, base, source, subspaces=None, **kwargs):
        transformed_base = self.unconstrained_layer(base)
        transformed_source = self.unconstrained_layer(source)
        # interchange
        # clone to avoid in-place modifications that can break autograd
        _tb = transformed_base.clone()
        _ts = transformed_source.clone()
        transformed_base = _do_intervention_by_swap(
            _tb,
            _ts,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )
        # inverse base
        output = torch.linalg.pinv(self.unconstrained_layer.weight) @ transformed_base.T
        return output.T.to(base.dtype)


    def __str__(self):
        return f"UnconstrainedIntervention()"


class StretchedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the scaled space."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        stretch_layer = StretchLayer(self.embed_dim)
        self.stretch_layer = stretch_layer


    def forward(self, base, source, subspaces=None, **kwargs):
        stretched_base = self.stretch_layer(base)
        stretched_source = self.stretch_layer(source)
        # interchange
        # clone to avoid in-place modifications that can break autograd
        _sb = stretched_base.clone()
        _ss = stretched_source.clone()
        stretched_base = _do_intervention_by_swap(
            _sb,
            _ss,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )
        # inverse base
        output = stretched_base / self.stretch_layer.weight
        return output.to(base.dtype)



    def __str__(self):
        return f"StretchedSpaceIntervention()"



class RealNVPIntervention(TrainableIntervention, DistributedRepresentationIntervention):
    """
        Intervention using RealNVP for the scaled space.
    """
    
    def __init__(self, n_layers=4, latent_dim=None, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.realNVPlayer = RealNVPLayer(n_layers, latent_dim, hidden_dim)

    def forward(self, base, source, subspaces=None, **kwargs):
        flowed_base = self.realNVPlayer(base)
        flowed_source = self.realNVPlayer(source)
        flowed_base = _do_intervention_by_swap(
            flowed_base,
            flowed_source,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )
        # inverse base
        output = self.realNVPlayer.realnvp.inverse(flowed_base)
        return output.to(base.dtype)

    def __str__(self):
        return f"RealNVPIntervention()"
    


class RevNetIntervention(TrainableIntervention, DistributedRepresentationIntervention):
    """
        Intervention using RevNet for the scaled space.
    """
    
    def __init__(self, n_layers=None, latent_dim=None, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.revNetLayer = RevNetLayer(n_layers, latent_dim, hidden_dim)

    def forward(self, base, source, subspaces=None, **kwargs):
        revnet_base = self.revNetLayer(base)
        revnet_source = self.revNetLayer(source)
        # Optional debug hook: if REVNET_DEBUG=1 in the environment register a
        # backward hook on the revnet representation tensor so we can observe
        # the gradient that flows back to the representation during training.
        try:
            if os.environ.get("REVNET_DEBUG", "0") == "1":
                def _show_grad(g):
                    try:
                        nz = int((g.abs() > 0).sum().item()) if g.numel() > 0 else 0
                        print(f"REVNET_BASE grad: max={g.abs().max().item():.6e} mean={g.abs().mean().item():.6e} nonzero={nz} dtype={g.dtype} device={g.device}")
                    except Exception as e:
                        print("REVNET debug hook error:", e)

                # register_hook returns a removable handle but we don't store it here
                revnet_base.register_hook(_show_grad)
        except Exception:
            # don't fail forward if debugging hook can't be registered
            pass
        # interchange
        # clone to avoid in-place modifications that can break autograd
        _rb = revnet_base.clone()
        _rs = revnet_source.clone()
        revnet_base = _do_intervention_by_swap(
            _rb,
            _rs,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )
        # inverse base
        output = self.revNetLayer.revnet.inverse(revnet_base)
        return output.to(base.dtype)

    def __str__(self):
        return f"RevNetIntervention()"