import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, List

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
BRUGADA_LEADS = ['V1', 'V2', 'V3']
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._hook_forward = target_layer.register_forward_hook(self._save_activation)
        self._hook_backward = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        task: str = 'brugada',
    ) -> np.ndarray:
        self.model.eval()
        x = x.requires_grad_(True)

        if edge_index is not None:
            outputs = self.model(x, edge_index, edge_weight)
        else:
            outputs = self.model(x)

        score = outputs[task]
        self.model.zero_grad()
        score.backward()

        weights = self._gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self._activations).sum(dim=1)
        cam = torch.relu(cam).squeeze(0)

        T_original = x.shape[-1]
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = torch.nn.functional.interpolate(
            cam, size=T_original, mode='linear', align_corners=False
        ).squeeze()

        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        self._hook_forward.remove()
        self._hook_backward.remove()


class GNNAttentionMapper:
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
        return importance.squeeze(0).cpu().numpy()

    def get_top_leads(self, importance: np.ndarray, k: int = 3) -> List[str]:
        top_indices = np.argsort(importance)[::-1][:k]
        return [LEAD_NAMES[i] for i in top_indices]

    def plot_lead_bar(
        self,
        importance: np.ndarray,
        highlight_leads: Optional[List[str]] = None,
        title: str = 'Lead Importance (GNN)',
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        if highlight_leads is None:
            highlight_leads = self.get_top_leads(importance, k=3)

        colors = ['#d44' if name in highlight_leads else 'steelblue' for name in LEAD_NAMES]
        bars = ax.bar(LEAD_NAMES, importance, color=colors, edgecolor='white', width=0.6)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance Score')
        ax.set_ylim(0, importance.max() * 1.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, val in zip(bars, importance):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        top_model = self.get_top_leads(importance, k=3)
        ax.text(0.01, 0.97, f"Top model: {', '.join(top_model)}",
                transform=ax.transAxes, fontsize=8, va='top', color='#d44')
        ax.text(0.01, 0.90, f"Klinis Brugada: {', '.join(BRUGADA_LEADS)}",
                transform=ax.transAxes, fontsize=8, va='top', color='gray')

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return ax

    def plot_lead_graph(
        self,
        importance: np.ndarray,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        highlight_leads: Optional[List[str]] = None,
        title: str = 'Lead Connectivity Graph',
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        if highlight_leads is None:
            highlight_leads = self.get_top_leads(importance, k=3)

        pos = {
            'I':   (-1.5,  1.5), 'II':  ( 0.0,  1.5), 'III': ( 1.5,  1.5),
            'aVR': (-1.5,  0.0), 'aVL': ( 0.0,  0.0), 'aVF': ( 1.5,  0.0),
            'V1':  (-2.5, -1.5), 'V2':  (-1.5, -1.5), 'V3':  (-0.5, -1.5),
            'V4':  ( 0.5, -1.5), 'V5':  ( 1.5, -1.5), 'V6':  ( 2.5, -1.5),
        }

        xs = [pos[n][0] for n in LEAD_NAMES]
        ys = [pos[n][1] for n in LEAD_NAMES]

        ei = edge_index.cpu().numpy()
        ew = edge_weight.cpu().numpy() if edge_weight is not None else np.ones(ei.shape[1])
        ew_norm = (ew - ew.min()) / (ew.max() - ew.min() + 1e-8)

        drawn_edges = set()
        for i in range(ei.shape[1]):
            src, dst = ei[0, i], ei[1, i]
            ax.plot([xs[src], xs[dst]], [ys[src], ys[dst]],
                    color='gray', linewidth=ew_norm[i] * 2.5 + 0.3, alpha=0.4, zorder=1)
            
            edge_pair = tuple(sorted((src, dst)))
            if edge_pair not in drawn_edges and edge_weight is not None:
                mid_x = (xs[src] + xs[dst]) / 2
                mid_y = (ys[src] + ys[dst]) / 2
                ax.text(mid_x, mid_y, f"{ew[i]:.2f}", 
                        ha='center', va='center', fontsize=7, color='blue', 
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
                        zorder=2)
                drawn_edges.add(edge_pair)
                        

        sc = ax.scatter(xs, ys, s=importance * 3000 + 300, c=importance,
                        cmap=plt.cm.YlOrRd, vmin=0, vmax=importance.max(),
                        zorder=2, edgecolors='white', linewidths=1.5)

        for i, name in enumerate(LEAD_NAMES):
            color = '#d44' if name in highlight_leads else 'black'
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
    cmap = plt.cm.YlOrRd

    for i in range(T - 1):
        ax.axvspan(t[i], t[i + 1], alpha=heatmap[i] * 0.6,
                   color=cmap(heatmap[i]), linewidth=0)

    ax.plot(t, signal, color='black', linewidth=0.9, zorder=2)

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

    return ax


def _get_target_layer(model: nn.Module, model_type: str) -> Optional[nn.Module]:
    if model_type == 'spatial_gnn':
        return model.temporal_encoder[-1]
    elif model_type == 'resnet_baseline':
        return model.blocks[-1]
    return None


def plot_xai_report(
    model: nn.Module,
    model_type: str,
    x: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
    edge_weight: Optional[torch.Tensor] = None,
    task: str = 'brugada',
    top_k: int = 3,
    patient_id: str = '?',
    label: int = 1,
    save_path: str = 'xai_report.png',
):
    signal = x.squeeze(0).cpu().detach().numpy()
    has_gnn = hasattr(model, 'get_lead_importance')

    if has_gnn and edge_index is not None:
        mapper = GNNAttentionMapper(model)
        importance = mapper.extract(x, edge_index, edge_weight)
        top_leads = mapper.get_top_leads(importance, k=top_k)
        if hasattr(model, 'get_learned_edges'):
            edge_index, edge_weight = model.get_learned_edges(x, edge_index, edge_weight)
    else:
        importance = None
        top_leads = BRUGADA_LEADS[:top_k]

    target_layer = _get_target_layer(model, model_type)
    heatmaps = {}
    if target_layer is not None:
        cam = GradCAM(model, target_layer)
        for lead in top_leads:
            heatmaps[lead] = cam.generate(x, edge_index, edge_weight, task=task)
        cam.remove_hooks()

    n_rows = top_k + (2 if has_gnn else 0)
    fig = plt.figure(figsize=(16, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.5, wspace=0.35)
    label_str = 'Brugada' if label == 1 else 'Normal'

    fig.suptitle(
        f'XAI Report — Patient {patient_id} | Model: {model_type} | Label: {label_str}',
        fontsize=13, fontweight='bold', y=0.99
    )

    for row, lead in enumerate(top_leads):
        ax = fig.add_subplot(gs[row, :])
        lead_idx = LEAD_NAMES.index(lead)
        heatmap = heatmaps.get(lead, np.zeros(signal.shape[1]))
        plot_gradcam_signal(signal[lead_idx], heatmap,
                            lead_name=lead, label=label, ax=ax)

    if has_gnn and importance is not None:
        ax_bar = fig.add_subplot(gs[top_k:, 0])
        mapper.plot_lead_bar(importance, highlight_leads=top_leads,
                             title='Lead Importance (GNN)', ax=ax_bar)

        ax_graph = fig.add_subplot(gs[top_k:, 1])
        mapper.plot_lead_graph(importance, edge_index, edge_weight,
                               highlight_leads=top_leads,
                               title='Lead Connectivity', ax=ax_graph)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'[saved] {save_path}')
    plt.show()

def generate_xai_from_dataset(model, dataset, config, args):
    model.eval()
    device = next(model.parameters()).device
    model_type = config['model']['type']
    primary_task = 'brugada'
    
    idx = None
    
    # Safely extract patient IDs and labels whether it's a Subset or full Dataset
    if isinstance(dataset, torch.utils.data.Subset):
        patient_ids = [str(dataset.dataset.patient_ids[i]) for i in dataset.indices]
        if hasattr(dataset.dataset, 'labels') and primary_task in dataset.dataset.labels:
            labels_list = [dataset.dataset.labels[primary_task][i] for i in dataset.indices]
        else:
            labels_list = None
    else:
        patient_ids = [str(pid) for pid in dataset.patient_ids]
        if hasattr(dataset, 'labels') and primary_task in dataset.labels:
            labels_list = dataset.labels[primary_task]
        else:
            labels_list = None
            
    if args.patient_id:
        try:
            idx = patient_ids.index(str(args.patient_id))
            print(f"Found requested Patient {args.patient_id} at index {idx}")
        except ValueError:
            print(f"Patient {args.patient_id} not found in this split.")
            
    if idx is None:
        print("Searching for a Brugada-positive patient (label=1) in the dataset...")
        brugada_indices = []
        
        if labels_list is not None:
            brugada_indices = [i for i, lbl in enumerate(labels_list) if int(lbl) == 1]
        else:
            # Fallback for when labels are not pre-cached
            for i in range(len(dataset)):
                if int(dataset[i]['labels'][primary_task].item()) == 1:
                    brugada_indices.append(i)
                    
        if brugada_indices:
            idx = random.choice(brugada_indices)
            print(f"Selected Brugada patient: {patient_ids[idx]} (Index {idx})")
        else:
            print("Warning: No Brugada patients found in this split! Picking completely random.")
            idx = random.randint(0, len(dataset) - 1)

    sample = dataset[idx]
    patient_id = sample['patient_id']
    x = sample['signal'].unsqueeze(0).to(device)
    label = int(sample['labels'][primary_task].item())
    
    edge_index = sample.get('edge_index', None)
    edge_weight = sample.get('edge_weight', None)
    
    if edge_index is not None:
        edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
        
    out = args.output or f"experiments/xai_{config['experiment_name']}_{patient_id}.png"
    
    plot_xai_report(
        model=model,
        model_type=model_type,
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        task=primary_task,
        top_k=args.top_k,
        patient_id=patient_id,
        label=label,
        save_path=out
    )