import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations = None
        self._gradients   = None

        # Pasang hooks
        self._hook_forward  = target_layer.register_forward_hook(self._save_activation)
        self._hook_backward = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Simpan activation waktu forward pass."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Simpan gradient waktu backward pass."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        task: str = 'brugada',
        lead_idx: int = 0,
    ) -> np.ndarray:

        self.model.eval()
        x = x.requires_grad_(True)

        # Forward pass
        if edge_index is not None:
            outputs = self.model(x, edge_index, edge_weight)
        else:
            outputs = self.model(x)

        score = outputs[task]  # shape: [1, 1]

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Grad-CAM: bobot tiap channel = rata-rata gradient
        # activations: [1, C, T], gradients: [1, C, T]
        weights = self._gradients.mean(dim=-1, keepdim=True)  # [1, C, 1]
        cam     = (weights * self._activations).sum(dim=1)     # [1, T]
        cam     = torch.relu(cam).squeeze(0)                   # [T]

        # Upsample ke panjang sinyal asli
        T_original = x.shape[-1]
        cam = cam.unsqueeze(0).unsqueeze(0)                    # [1, 1, T_cam]
        cam = torch.nn.functional.interpolate(
            cam, size=T_original, mode='linear', align_corners=False
        ).squeeze()                                            # [T_original]

        # Normalize ke [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Hapus hooks setelah selesai dipakai."""
        self._hook_forward.remove()
        self._hook_backward.remove()


class GNNAttentionMapper:
    # Nama standar 12 lead ECG
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def __init__(self, model: nn.Module):
        if not hasattr(model, 'get_lead_importance'):
            raise AttributeError(
                "Model harus punya method get_lead_importance(). "
                "Pastikan pakai SpatialGNN, bukan ResNetBaseline."
            )
        self.model = model

    def extract(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> np.ndarray:

        self.model.eval()
        with torch.no_grad():
            importance = self.model.get_lead_importance(x, edge_index, edge_weight)
        return importance.squeeze(0).cpu().numpy()  # shape: (12,)

    def plot_lead_bar(
        self,
        importance: np.ndarray,
        title: str = 'Lead Importance (GNN)',
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> plt.Axes:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        colors = ['#d44' if imp > np.percentile(importance, 75) else 'steelblue'
                  for imp in importance]

        bars = ax.bar(self.LEAD_NAMES, importance, color=colors, edgecolor='white', width=0.6)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance Score')
        ax.set_ylim(0, importance.max() * 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotasi nilai
        for bar, val in zip(bars, importance):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # Highlight V1–V3 (lead khas Brugada)
        for i, name in enumerate(self.LEAD_NAMES):
            if name in ['V1', 'V2', 'V3']:
                ax.get_xticklabels()[i].set_color('#d44')
                ax.get_xticklabels()[i].set_fontweight('bold')

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'[saved] {save_path}')

        return ax

    def plot_lead_graph(
        self,
        importance: np.ndarray,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        title: str = 'Lead Connectivity Graph',
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        # Posisi anatomis (x, y) untuk 12 lead
        pos = {
            'I':   (-1.5,  1.5), 'II':  ( 0.0,  1.5), 'III': ( 1.5,  1.5),
            'aVR': (-1.5,  0.0), 'aVL': ( 0.0,  0.0), 'aVF': ( 1.5,  0.0),
            'V1':  (-2.5, -1.5), 'V2':  (-1.5, -1.5), 'V3':  (-0.5, -1.5),
            'V4':  ( 0.5, -1.5), 'V5':  ( 1.5, -1.5), 'V6':  ( 2.5, -1.5),
        }

        xs = [pos[n][0] for n in self.LEAD_NAMES]
        ys = [pos[n][1] for n in self.LEAD_NAMES]

        # Gambar edges
        ei = edge_index.cpu().numpy()
        ew = edge_weight.cpu().numpy() if edge_weight is not None else np.ones(ei.shape[1])
        ew_norm = (ew - ew.min()) / (ew.max() - ew.min() + 1e-8)

        for i in range(ei.shape[1]):
            src, dst = ei[0, i], ei[1, i]
            ax.plot(
                [xs[src], xs[dst]], [ys[src], ys[dst]],
                color='gray', linewidth=ew_norm[i] * 2.5 + 0.3,
                alpha=0.4, zorder=1
            )

        # Gambar nodes
        node_sizes  = importance * 3000 + 300
        node_colors = importance

        cmap = plt.cm.YlOrRd
        sc = ax.scatter(xs, ys, s=node_sizes, c=node_colors,
                        cmap=cmap, vmin=0, vmax=importance.max(),
                        zorder=2, edgecolors='white', linewidths=1.5)

        # Label lead
        for i, name in enumerate(self.LEAD_NAMES):
            color = '#d44' if name in ['V1', 'V2', 'V3'] else 'black'
            ax.text(xs[i], ys[i], name, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color, zorder=3)

        plt.colorbar(sc, ax=ax, label='Importance Score', shrink=0.6)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-2.5, 2.5)
        ax.axis('off')

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'[saved] {save_path}')

        return ax


def plot_gradcam_signal(
    signal: np.ndarray,
    heatmap: np.ndarray,
    lead_name: str = 'V1',
    label: int = 1,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))

    T = len(signal)
    t = np.arange(T)

    # Background heatmap (colored spans)
    cmap = plt.cm.YlOrRd
    for i in range(T - 1):
        ax.axvspan(t[i], t[i+1], alpha=heatmap[i] * 0.6,
                   color=cmap(heatmap[i]), linewidth=0)

    # Sinyal ECG
    ax.plot(t, signal, color='black', linewidth=0.9, zorder=2)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Grad-CAM intensity', shrink=0.8, pad=0.01)

    label_str = 'Brugada' if label == 1 else 'Normal'
    ax.set_title(f'Grad-CAM — Lead {lead_name} ({label_str})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude (mV)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[saved] {save_path}')

    return ax


def plot_xai_report(
    signal: np.ndarray,
    gradcam_heatmaps: dict,
    lead_importance: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    patient_id: str = '?',
    label: int = 1,
    save_path: str = 'xai_report.png',
):

    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.5, wspace=0.35)

    label_str = 'Brugada' if label == 1 else 'Normal'
    fig.suptitle(
        f'XAI Report — Patient {patient_id} | Prediction: {label_str}',
        fontsize=14, fontweight='bold', y=0.98
    )

    # ── Grad-CAM: V1, V2, V3 ──────────────────────────────────
    target_leads = ['V1', 'V2', 'V3']
    for row, lead in enumerate(target_leads):
        ax = fig.add_subplot(gs[row, :])
        lead_idx = LEAD_NAMES.index(lead)
        heatmap  = gradcam_heatmaps.get(lead, np.zeros(signal.shape[1]))
        plot_gradcam_signal(signal[lead_idx], heatmap, lead_name=lead, label=label, ax=ax)

    # ── GNN Attention: bar chart ───────────────────────────────
    ax_bar   = fig.add_subplot(gs[3:, 0])
    mapper   = GNNAttentionMapper.__new__(GNNAttentionMapper)
    mapper.LEAD_NAMES = LEAD_NAMES
    mapper.plot_lead_bar(lead_importance, title='Lead Importance (GNN)', ax=ax_bar)

    # ── GNN Attention: graph ───────────────────────────────────
    ax_graph = fig.add_subplot(gs[3:, 1])
    mapper.plot_lead_graph(lead_importance, edge_index, edge_weight,
                           title='Lead Connectivity', ax=ax_graph)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'[saved] {save_path}')
    plt.show()