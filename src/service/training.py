from dataclasses import dataclass
from typing import Iterable, List, Tuple

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # Optional dependencies for fine-tuning the weighting model
    torch = None
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    AutoModelForSequenceClassification = object  # type: ignore
    AutoTokenizer = object  # type: ignore

try:
    from peft import LoraConfig, get_peft_model  # type: ignore
except ImportError:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


@dataclass
class TripletExample:
    anchor: str
    positive: str
    negative: str


class TripletDataset(Dataset):  # type: ignore[misc]
    def __init__(self, tokenizer, examples: Iterable[TripletExample], max_length: int = 256) -> None:
        if torch is None:
            raise ImportError('torch and transformers are required to use TripletDataset')
        self.tokenizer = tokenizer
        self.examples: List[TripletExample] = list(examples)
        self.max_length = max_length

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.examples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        example = self.examples[idx]
        anchor_inputs = self.tokenizer(
            example.anchor,
            example.positive,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        negative_inputs = self.tokenizer(
            example.anchor,
            example.negative,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        anchor_batch = {k: v.squeeze(0) for k, v in anchor_inputs.items()}
        negative_batch = {k: v.squeeze(0) for k, v in negative_inputs.items()}
        return anchor_batch, negative_batch


class WeightingModelTrainer:
    def __init__(
        self,
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        lr: float = 2e-5,
        use_dora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Tuple[str, ...] = ('query', 'key', 'value'),
    ) -> None:
        if torch is None:
            raise ImportError('Install torch and transformers to use WeightingModelTrainer')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        if use_dora:
            if LoraConfig is None or get_peft_model is None:
                raise ImportError('peft is required when use_dora=True')
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=list(target_modules),
                lora_dropout=lora_dropout,
                bias='none',
                task_type='SEQ_CLS',
                use_dora=True,
            )
            self.model = get_peft_model(base_model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MarginRankingLoss(margin=0.2)

    def build_dataloader(self, examples: Iterable[TripletExample], batch_size: int = 4, max_length: int = 256) -> DataLoader:
        dataset = TripletDataset(self.tokenizer, examples, max_length=max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, dataloader: DataLoader, epochs: int = 1, device: str = 'cpu') -> None:
        self.model.to(device)
        self.model.train()
        target = torch.ones(1, device=device)
        for _ in range(epochs):
            for positive_inputs, negative_inputs in dataloader:
                positive_inputs = {k: v.to(device) for k, v in positive_inputs.items()}
                negative_inputs = {k: v.to(device) for k, v in negative_inputs.items()}

                positive_scores = self.model(**positive_inputs).logits
                negative_scores = self.model(**negative_inputs).logits
                loss = self.loss_fn(positive_scores, negative_scores, target.expand_as(positive_scores))

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model.from_pretrained(path)
        self.tokenizer.from_pretrained(path)
