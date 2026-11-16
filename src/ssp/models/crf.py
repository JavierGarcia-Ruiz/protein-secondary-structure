"""Linear-Chain Conditional Random Field (CRF) layer in pure PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """Linear-chain Conditional Random Field.

    Computes negative log-likelihood loss and Viterbi decoding over variable-length sequences.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # transitions[i, j] is the score of transitioning from tag j to tag i
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the negative log-likelihood: -log P(tags | emissions).

        Args:
            emissions: (B, L, num_tags)
            tags: (B, L) with tag indices
            mask: (B, L) boolean mask indicating non-padding residues
        """
        # Gold score: score of ground truth sequence
        gold_score = self._score_sentence(emissions, tags, mask)
        # Partition function Z: sum over all possible sequences
        forward_score = self._compute_log_partition(emissions, mask)
        return torch.mean(forward_score - gold_score)

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # emissions: (B, L, K)
        B, L, K = emissions.shape
        # Initialize forward variables with start transitions
        # alpha_0 = start_transitions + emissions[:, 0]
        alpha = self.start_transitions.view(1, K) + emissions[:, 0]  # (B, K)

        for t in range(1, L):
            emit_score = emissions[:, t].unsqueeze(2)  # (B, K, 1)
            trans_score = self.transitions.unsqueeze(0)  # (1, K, K)
            # broadcast alpha: (B, 1, K)
            # next_tag_var = alpha + trans_score + emit_score -> (B, K, K)
            next_tag_var = alpha.unsqueeze(1) + trans_score + emit_score
            next_alpha = torch.logsumexp(next_tag_var, dim=2)  # (B, K)
            
            # Mask padding
            m_t = mask[:, t].unsqueeze(1)  # (B, 1)
            alpha = torch.where(m_t, next_alpha, alpha)

        # Add end transitions
        terminal_vars = alpha + self.end_transitions.view(1, K)
        return torch.logsumexp(terminal_vars, dim=1)  # (B,)

    def _score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[torch.arange(B), 0, tags[:, 0]]

        for t in range(1, L):
            m_t = mask[:, t]
            prev_t = tags[:, t - 1]
            curr_t = tags[:, t]
            
            trans_score = self.transitions[curr_t, prev_t]
            emit_score = emissions[torch.arange(B), t, curr_t]
            step_score = trans_score + emit_score
            score = score + torch.where(m_t, step_score, torch.zeros_like(score))

        # Add end transitions at the actual last valid position
        last_valid_indices = mask.sum(dim=1).long() - 1
        last_valid_indices = torch.clamp(last_valid_indices, min=0)
        last_tags = tags[torch.arange(B), last_valid_indices]
        score = score + self.end_transitions[last_tags]
        return score

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding to find the highest-scoring sequence of tags."""
        B, L, K = emissions.shape
        viterbi_vars = self.start_transitions.view(1, K) + emissions[:, 0]
        backpointers: list[torch.Tensor] = []

        for t in range(1, L):
            m_t = mask[:, t]
            # (B, K, 1) + (1, K, K) + (B, 1, K)
            broadcast_vit = viterbi_vars.unsqueeze(1)  # (B, 1, K)
            broadcast_trans = self.transitions.unsqueeze(0)  # (1, K, K)
            next_tag_var = broadcast_vit + broadcast_trans
            max_vars, bptrs = torch.max(next_tag_var, dim=2)  # (B, K)
            
            next_viterbi = max_vars + emissions[:, t]
            viterbi_vars = torch.where(m_t.unsqueeze(1), next_viterbi, viterbi_vars)
            backpointers.append(bptrs)

        # Transition to stop_tag
        terminal_vars = viterbi_vars + self.end_transitions.view(1, K)
        best_last_tags = torch.argmax(terminal_vars, dim=1).tolist()

        # Follow backpointers
        best_paths: list[list[int]] = []
        lengths = mask.sum(dim=1).long().tolist()

        for b in range(B):
            seq_len = lengths[b]
            best_tag = best_last_tags[b]
            best_path = [best_tag]
            for t in range(seq_len - 2, -1, -1):
                best_tag = int(backpointers[t][b, best_tag].item())
                best_path.append(best_tag)
            best_path.reverse()
            best_paths.append(best_path)

        return best_paths

