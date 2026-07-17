import torch


class NTXentLoss(torch.nn.Module):
    targets = {}

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        """x: [batch_size, n_features]"""
        xcs = torch.nn.functional.cosine_similarity(x[..., None, :, :], x[..., :, None, :], dim=-1)
        xcs[..., torch.arange(xcs.shape[-1]), torch.arange(xcs.shape[-1])] = float("-inf")
        labels = self.make_labels(xcs.shape[-1], xcs.device)
        if xcs.ndim==3:
            labels = labels.expand(xcs.shape[0], -1)
            xcs = xcs.view(-1, xcs.shape[-1])
            labels = labels.reshape(-1)

        return torch.nn.functional.cross_entropy(xcs / self.temperature, labels)

    def make_labels(self, n: int, device="cpu") -> torch.Tensor:
        if n not in self.targets:
            target = torch.arange(n, device=device)
            target[0::2] += 1
            target[1::2] -= 1
            self.targets[n] = target
        return self.targets[n].to(device=device)


def get_cls_criterion(class_weights, n_outputs, device):
    if n_outputs == 1:
        if class_weights is not None:
            class_weights = (class_weights[1] / class_weights[0]).to(device)
        return torch.nn.BCEWithLogitsLoss(weight=class_weights)
    else:
        return torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

def get_criterion(objective: str, class_weights=None, n_out=None, device=None, temperature=0.1):
    if objective == "classification":
        return get_cls_criterion(class_weights, n_out, device)
    if objective == "contrastive":
        return NTXentLoss(temperature=temperature)

    raise ValueError(f"Unsupported objective: {objective}")
