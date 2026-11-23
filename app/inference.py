"""LMP predictor interface and safe startup loader for trained XLM-R artifacts."""
from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
from .config import Settings
from .scoring import SCORE_MAX, SCORE_MIN
MANIFEST = "lmp_manifest.json"
class InferenceError(RuntimeError): pass
class Predictor(Protocol):
    version: str
    backbone: str
    def predict(self, originals: list[str], simplified: list[str], languages: list[str]) -> Sequence[float]: ...
@dataclass(frozen=True)
class ModelStatus:
    available: bool
    reason: str | None
    message: str
    version: str | None
    backbone: str

def validate_scores(scores, n: int) -> list[float]:
    try: out=[float(s) for s in scores]
    except (TypeError,ValueError) as e: raise InferenceError("predictor returned non-numeric scores") from e
    if len(out)!=n: raise InferenceError(f"predictor returned {len(out)} scores for {n} inputs")
    if any(not math.isfinite(s) or not SCORE_MIN <= s <= SCORE_MAX for s in out): raise InferenceError("predictor returned a score outside [1, 10]")
    return out

class XLMRPredictor:
    def __init__(self, directory: Path, manifest: dict):
        import torch
        from transformers import AutoTokenizer
        from .models.xlmr import XLMRPairRegressor
        self.version=manifest["version"]; self.backbone=manifest["backbone"]
        self.max_length=int(manifest.get("max_length",256)); self.device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer=AutoTokenizer.from_pretrained(directory / manifest.get("tokenizer_dir","tokenizer"))
        self.model=XLMRPairRegressor(self.backbone, pretrained=False, config_path=directory / manifest.get("tokenizer_dir","tokenizer"))
        state=torch.load(directory / manifest["weights_file"], map_location="cpu", weights_only=True)
        self.model.load_state_dict(state); self.model.to(self.device).eval()
        self.torch=torch
    def predict(self, originals, simplified, languages):
        if not (len(originals)==len(simplified)==len(languages)): raise InferenceError("input lengths do not match")
        scores=[]
        with self.torch.no_grad():
            for start in range(0,len(originals),16):
                enc=self.tokenizer(originals[start:start+16], simplified[start:start+16], truncation=True,
                                   max_length=self.max_length, padding=True, return_tensors="pt")
                enc={k:v.to(self.device) for k,v in enc.items()}
                scores.extend(self.model(**enc).cpu().tolist())
        return validate_scores(scores,len(originals))

def load_predictor(settings: Settings) -> tuple[Predictor | None, ModelStatus]:
    def unavailable(reason,message,version=None): return None,ModelStatus(False,reason,message,version,settings.backbone)
    d=Path(settings.model_dir); mp=d/MANIFEST
    if not mp.is_file(): return unavailable("weights_missing","No trained LMP model is installed; LMP scores are not available.")
    try:
        manifest=json.loads(mp.read_text()); version=manifest["version"]; weights=manifest["weights_file"]
        if not isinstance(version,str) or not isinstance(weights,str) or Path(weights).name!=weights: raise ValueError
        tokenizer_dir=manifest.get("tokenizer_dir","tokenizer")
        if not isinstance(tokenizer_dir,str) or Path(tokenizer_dir).name!=tokenizer_dir: raise ValueError
        backbone=manifest["backbone"]
        if not isinstance(backbone,str): raise ValueError
    except Exception: return unavailable("manifest_invalid","The model manifest is missing required fields or is unreadable.")
    if not (d/weights).is_file() or not (d/tokenizer_dir).is_dir(): return unavailable("weights_missing","The model manifest exists but model weights or tokenizer files are missing.",version)
    try:
        predictor=XLMRPredictor(d,manifest)
        return predictor,ModelStatus(True,None,"Trained XLM-R LMP model loaded.",version,predictor.backbone)
    except ImportError: return unavailable("dependencies_missing","Install the ML dependencies (torch and transformers) to load this model.",version)
    except Exception: return unavailable("load_error","The trained model artifacts could not be loaded; inspect server logs.",version)
