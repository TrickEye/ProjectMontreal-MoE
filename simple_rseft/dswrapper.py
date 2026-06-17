import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchedMoEGate(nn.Module):
    """把 MoEGate.weight (nn.Parameter) 换成 nn.Linear，使 PEFT 可以识别"""
    def __init__(self, gate):
        super().__init__()
        # 复制所有配置属性
        for attr in ["top_k", "n_routed_experts", "scoring_func", "alpha",
                     "seq_aux", "norm_topk_prob", "gating_dim", "config"]:
            setattr(self, attr, getattr(gate, attr))

        # 关键：把 nn.Parameter 换成 nn.Linear
        n_experts, hidden = gate.weight.shape
        self.router_linear = nn.Linear(hidden, n_experts, bias=False)
        self.router_linear.weight.data.copy_(gate.weight.data)  # 迁移原始权重

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states_2d = hidden_states.view(-1, h)

        logits = self.router_linear(hidden_states_2d)   # 用 nn.Linear 替代 F.linear
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # aux_loss（load balancing）
        if self.training and self.alpha > 0.0:
            topk_idx_for_aux = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_view = scores.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states_2d.device)
                ce.scatter_add_(1, topk_idx_for_aux,
                    torch.ones(bsz, seq_len * self.top_k, device=hidden_states_2d.device)
                ).div_(seq_len * self.top_k / self.n_routed_experts)
                aux_loss = (ce * scores_view.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (scores.mean(0) * fi).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss