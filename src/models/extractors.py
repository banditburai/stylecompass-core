from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from contextlib import contextmanager
from abc import ABC, abstractmethod
import faiss
from einops import rearrange
from src.utils import MetricLogger
from src.utils.embeddings import EmbeddingIO

@dataclass(frozen=True)
class ExtractorConfig:
    """Unified configuration for feature extraction"""
    model_type: Literal["dino", "moco", "vgg", "clip", "csd"] = "csd"
    feature_type: Literal["normal", "gram"] = "normal"
    layer: int = 1
    eval_embed: Literal["head", "backbone"] = "head"
    use_cuda: bool = True
    use_fp16: bool = False
    multiscale: bool = False
    gram_dims: Optional[int] = None
    embed_dir: Path = Path("./embeddings")
    device: torch.device = torch.device("cuda")
    pca_dim: Optional[int] = None
    arch: str = "vit"
    multilayer: Optional[List[int]] = None

@contextmanager
def cuda_autocast_context(enabled: bool):
    """Context manager for automatic mixed precision"""
    if enabled:
        with torch.cuda.amp.autocast():
            yield
    else:
        yield

class FeatureExtractorStrategy(ABC):
    @abstractmethod
    def extract(self, model: nn.Module, samples: torch.Tensor, config: ExtractorConfig) -> torch.Tensor:
        pass

class DinoExtractor(FeatureExtractorStrategy):
    def extract(self, model: nn.Module, samples: torch.Tensor, config: ExtractorConfig) -> torch.Tensor:
        if config.layer > 1:
            return model.module.get_intermediate_layers(samples, config.layer)[0][:, 0, :].clone()
        elif config.layer == -1:
            allfeats = model.module.get_intermediate_layers(samples, len(model.module.blocks))
            feats = [allfeats[i - 1][:, 0, :] for i in config.multilayer]
            bdim, _ = feats[0].shape
            return torch.stack(feats, dim=1).reshape((bdim, -1)).clone()
        return model(samples).clone()

class MocoExtractor(FeatureExtractorStrategy):
    def extract(self, model: nn.Module, samples: torch.Tensor, config: ExtractorConfig) -> torch.Tensor:
        feats = model.module.forward_features(samples)
        return feats[:, 0, :].clone()

class VGGExtractor(FeatureExtractorStrategy):
    def extract(self, model: nn.Module, samples: torch.Tensor, config: ExtractorConfig) -> torch.Tensor:
        if config.feature_type == "gram":
            temp = model.module.features(samples)
            temp = temp.view(temp.size(0), temp.size(1), -1)
            feats = torch.einsum('bji,bki->bjk', temp, temp)
            feats = feats.div(temp.shape[1])
            return rearrange(feats, 'b c d -> b (c d)')
        return model.module.features(samples).clone()

class ClipExtractor(FeatureExtractorStrategy):
    def extract(self, model: nn.Module, samples: torch.Tensor, config: ExtractorConfig) -> torch.Tensor:
        allfeats = model.module.visual.get_intermediate_layers(samples.type(model.module.dtype))
        allfeats.reverse()
        
        if config.feature_type == "gram":
            temp = allfeats[config.layer - 1]
            temp = nn.functional.normalize(temp, dim=2)
            feats = torch.einsum('bij,bik->bjk', temp, temp)
            feats = feats.div(temp.shape[1])
            return rearrange(feats, 'b c d -> b (c d)')
            
        if config.arch == 'resnet50':
            if config.layer == -1:
                raise Exception('Layer=-1 not allowed with clip resnet')
            elif config.layer == 1:
                return allfeats[0].clone()
            else:
                assert len(allfeats) >= config.layer
                return rearrange(allfeats[config.layer - 1], 'b c h w -> b c', 'mean').clone()
        else:
            if config.layer == -1:
                feats = [allfeats[i - 1][:, 0, :] for i in config.multilayer]
                bdim, _ = feats[0].shape
                return torch.stack(feats, dim=1).reshape((bdim, -1)).clone()
            else:
                assert len(allfeats) >= config.layer
                return allfeats[config.layer - 1][:, 0, :].clone()

class CSDExtractor(FeatureExtractorStrategy):
    def extract(self, model: nn.Module, samples: torch.Tensor, config: ExtractorConfig) -> torch.Tensor:
        bb_feats, _, style_feats = model(samples)
        return style_feats if config.eval_embed == "head" else bb_feats

class FeatureExtractor:
    def __init__(self, config: ExtractorConfig):
        self.config = config
        self._strategy = self._get_strategy()
        self.pca_model: Optional[faiss.PCAMatrix] = None
        self.embedding_io = EmbeddingIO()
        self.metric_logger = MetricLogger(delimiter="  ")

    def _get_strategy(self) -> FeatureExtractorStrategy:
        strategies: Dict[str, FeatureExtractorStrategy] = {
            "dino": DinoExtractor(),
            "moco": MocoExtractor(),
            "vgg": VGGExtractor(),
            "clip": ClipExtractor(),
            "csd": CSDExtractor()
        }
        return strategies.get(self.config.model_type, DinoExtractor())

    @torch.no_grad()
    def extract_features(self, model: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, Optional[faiss.PCAMatrix]]:
        model.eval()
        features = None

        for samples, index in self.metric_logger.log_every(loader, 100):
            features = self._extract_with_amp(model, samples, index, features, loader)

        if features is not None and self.config.feature_type == "gram":
            return self._apply_pca(features)
        return features, None

    def _extract_with_amp(self, model: nn.Module, samples: torch.Tensor, 
                         index: torch.Tensor, features: Optional[torch.Tensor], 
                         loader: DataLoader) -> torch.Tensor:
        if self.config.use_fp16:
            with torch.cuda.amp.autocast():
                return self._extract_and_update(model, samples, index, features, loader)
        return self._extract_and_update(model, samples, index, features, loader)

    def _extract_and_update(self, model: nn.Module, samples: torch.Tensor,
                           index: torch.Tensor, features: Optional[torch.Tensor],
                           loader: DataLoader) -> torch.Tensor:
        if self.config.use_cuda:
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)

        feats = self._strategy.extract(model, samples, self.config)
        feats = nn.functional.normalize(feats, dim=1, p=2).to(torch.float16)

        if dist.is_initialized():
            return self._update_distributed(feats, index, features, loader)
        return self._update_single(feats, index, features, loader)

    def _update_single(self, feats: torch.Tensor, index: torch.Tensor, 
                      features: Optional[torch.Tensor], loader: DataLoader) -> torch.Tensor:
        if features is None:
            features = torch.zeros(len(loader.dataset), feats.shape[-1], dtype=feats.dtype)
            if self.config.use_cuda:
                features = features.cuda(non_blocking=True)
        
        if self.config.use_cuda:
            features.index_copy_(0, index, feats)
        else:
            features.index_copy_(0, index.cpu(), feats.cpu())
        return features

    def _update_distributed(self, feats: torch.Tensor, index: torch.Tensor, 
                          features: Optional[torch.Tensor], loader: DataLoader) -> torch.Tensor:
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = dist.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        feats_all = torch.empty(
            dist.get_world_size(), feats.size(0), feats.size(1),
            dtype=feats.dtype, device=feats.device
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = dist.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        if dist.get_rank() == 0:
            if features is None:
                features = torch.zeros(len(loader.dataset), feats.shape[-1], dtype=feats.dtype)
                if self.config.use_cuda:
                    features = features.cuda(non_blocking=True)
            features.index_copy_(0, index_all, torch.cat(output_l))

        return features

    def _apply_pca(self, features: torch.Tensor) -> Tuple[torch.Tensor, Optional[faiss.PCAMatrix]]:
        if self.pca_model is None:
            features_np = features.cpu().numpy()
            self.pca_model = faiss.PCAMatrix(features_np.shape[-1], self.config.gram_dims)
            self.pca_model.train(features_np)
            features_np = self.pca_model.apply_py(features_np)
            return torch.from_numpy(features_np).to(features.device), self.pca_model
        
        features_np = features.cpu().numpy()
        features_np = self.pca_model.apply_py(features_np)
        return torch.from_numpy(features_np).to(features.device), None

    def save_features(self, features: torch.Tensor, filenames: List[str], 
                     save_dir: Path, split: str = 'database') -> None:
        self.embedding_io.save_embeddings(features, filenames, save_dir, split)