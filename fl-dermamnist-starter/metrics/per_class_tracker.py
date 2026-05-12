from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ModuleNotFoundError:  # plotting heatmaps still works with matplotlib fallback
    sns = None


class PerClassTracker:
    def __init__(self, num_classes, class_names, save_dir):
        self.num_classes = num_classes
        self.class_names = class_names
        self.save_dir = Path(save_dir)
        self.data = []

    def log_round(self, round_num, metrics_dict):
        record = {'round': round_num}
        for name in self.class_names:
            short = name[:20].replace(' ', '_')
            record[f'f1_{short}'] = metrics_dict.get(f'f1_{short}', np.nan)
            record[f'recall_{short}'] = metrics_dict.get(f'recall_{short}', np.nan)
        record['balanced_accuracy'] = metrics_dict.get('balanced_accuracy', np.nan)
        record['worst_class_f1'] = metrics_dict.get('worst_class_f1', np.nan)
        self.data.append(record)

    def save(self, path=None):
        path = Path(path) if path else self.save_dir / 'per_class_history.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.data)
        df.to_csv(path, index=False)
        return df

    def plot_per_class_f1_curves(self, save_path):
        df = pd.DataFrame(self.data)
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in self.class_names:
            short = name[:20].replace(' ', '_')
            col = f'f1_{short}'
            if col in df:
                style = '--' if ('dermatofibroma' in name.lower() or 'vascular' in name.lower()) else '-'
                lw = 2.5 if style == '--' else 1.5
                ax.plot(df['round'], df[col], linestyle=style, linewidth=lw, label=name)
        ax.set_xlabel('Round')
        ax.set_ylabel('F1')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return fig

    def plot_per_class_recall_curves(self, save_path):
        df = pd.DataFrame(self.data)
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in self.class_names:
            short = name[:20].replace(' ', '_')
            col = f'recall_{short}'
            if col in df:
                style = '--' if ('dermatofibroma' in name.lower() or 'vascular' in name.lower()) else '-'
                lw = 2.5 if style == '--' else 1.5
                marker = '*' if 'melanoma' in name.lower() else None
                ax.plot(df['round'], df[col], linestyle=style, linewidth=lw, marker=marker, markevery=max(len(df) // 6, 1), label=name)
        ax.set_xlabel('Round')
        ax.set_ylabel('Recall')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return fig

    def plot_convergence_heatmap(self, metric='f1', save_path=None):
        df = pd.DataFrame(self.data)
        value_rows = []
        labels = []
        for name in self.class_names:
            short = name[:20].replace(' ', '_')
            col = f'{metric}_{short}'
            if col in df:
                value_rows.append(df[col].astype(float).values)
                labels.append(name)
        arr = np.asarray(value_rows, dtype=float)
        step = max(arr.shape[1] // 20, 1) if arr.size else 1
        fig, ax = plt.subplots(figsize=(8, 4))
        if sns is not None:
            sns.heatmap(arr[:, ::step], cmap='viridis', vmin=0, vmax=1, ax=ax,
                        xticklabels=df['round'].values[::step] if 'round' in df else True,
                        yticklabels=labels)
        else:
            im = ax.imshow(arr[:, ::step], aspect='auto', vmin=0, vmax=1, cmap='viridis')
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xticks(np.arange(0, arr[:, ::step].shape[1]))
            ax.set_xticklabels(df['round'].values[::step] if 'round' in df else np.arange(arr[:, ::step].shape[1]))
            fig.colorbar(im, ax=ax)
        ax.set_xlabel('Round')
        ax.set_ylabel('Class')
        ax.set_title(f'Per-class {metric.upper()} convergence')
        fig.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return fig

    def compute_convergence_round(self, threshold=0.3, metric='f1'):
        df = pd.DataFrame(self.data)
        out = {}
        for name in self.class_names:
            short = name[:20].replace(' ', '_')
            col = f'{metric}_{short}'
            if col not in df:
                out[name] = None
                continue
            reached = df.loc[df[col].astype(float) >= threshold, 'round']
            out[name] = int(reached.iloc[0]) if not reached.empty else None
        return out
