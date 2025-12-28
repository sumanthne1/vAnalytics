"""OSNet-based player re-identification extractor."""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# Suppress torchreid warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torchreid
from torchvision import transforms

logger = logging.getLogger(__name__)


class ReIDExtractor:
    """Extract 512-D appearance embeddings using OSNet for player re-identification."""

    def __init__(
        self,
        model_name: str = "osnet_ain_x1_0",
        device: Optional[str] = None,
        bbox_padding: float = 0.1,  # 10% padding around bbox
        use_flip_average: bool = True,  # Average original + flipped for robustness
    ):
        """
        Initialize the ReID extractor.

        Args:
            model_name: OSNet model variant. Options:
                - 'osnet_ain_x1_0' (default, best accuracy)
                - 'osnet_x1_0' (slightly faster)
                - 'osnet_x0_75' (smaller, faster)
            device: Device to run on ('mps', 'cuda', 'cpu'). Auto-detected if None.
            bbox_padding: Fraction to expand bbox (0.1 = 10% padding on each side).
            use_flip_average: If True, average embeddings from original + horizontally flipped crops.
        """
        self.bbox_padding = bbox_padding
        self.use_flip_average = use_flip_average
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name

        logger.info(f"Loading ReID model: {model_name} on {device}")

        # Build OSNet model with pretrained weights
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Standard ReID preprocessing (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard ReID input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info(f"ReID model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def _apply_padding(
        self, bbox: Tuple[float, float, float, float], frame_h: int, frame_w: int
    ) -> Tuple[int, int, int, int]:
        """Apply padding to bbox and clamp to frame boundaries."""
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        # Add padding
        pad_w = box_w * self.bbox_padding
        pad_h = box_h * self.bbox_padding

        x1 = int(x1 - pad_w)
        y1 = int(y1 - pad_h)
        x2 = int(x2 + pad_w)
        y2 = int(y2 + pad_h)

        # Clamp to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        return x1, y1, x2, y2

    def extract(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Extract 512-D embedding from a player crop.

        Args:
            frame: Full frame as BGR numpy array (H, W, 3)
            bbox: Bounding box as (x1, y1, x2, y2)

        Returns:
            L2-normalized 512-D embedding vector
        """
        h, w = frame.shape[:2]

        # Apply padding to bbox
        x1, y1, x2, y2 = self._apply_padding(bbox, h, w)

        # Extract crop
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            # Return zero embedding for invalid crops
            return np.zeros(512, dtype=np.float32)

        # Convert BGR to RGB for torchvision
        crop_rgb = crop[:, :, ::-1].copy()

        # Preprocess
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

            # Flip-average for robustness
            if self.use_flip_average:
                crop_flipped = np.fliplr(crop_rgb).copy()
                tensor_flipped = self.transform(crop_flipped).unsqueeze(0).to(self.device)
                embedding_flipped = self.model(tensor_flipped)
                embedding = (embedding + embedding_flipped) / 2

        # L2 normalize
        embedding = embedding.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding

    def extract_batch(
        self,
        frame: np.ndarray,
        bboxes: List[Tuple[float, float, float, float]],
    ) -> List[np.ndarray]:
        """
        Extract embeddings for multiple players efficiently (batch processing).

        Args:
            frame: Full frame as BGR numpy array
            bboxes: List of bounding boxes

        Returns:
            List of 512-D embeddings
        """
        if not bboxes:
            return []

        crops = []
        crops_flipped = []
        valid_indices = []

        h, w = frame.shape[:2]

        for i, bbox in enumerate(bboxes):
            # Apply padding
            x1, y1, x2, y2 = self._apply_padding(bbox, h, w)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            # Convert BGR to RGB
            crop_rgb = crop[:, :, ::-1].copy()
            crops.append(self.transform(crop_rgb))

            # Prepare flipped version if enabled
            if self.use_flip_average:
                crop_flipped = np.fliplr(crop_rgb).copy()
                crops_flipped.append(self.transform(crop_flipped))

            valid_indices.append(i)

        if not crops:
            return [np.zeros(512, dtype=np.float32) for _ in bboxes]

        # Batch inference
        batch = torch.stack(crops).to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch)

            # Flip-average for robustness
            if self.use_flip_average and crops_flipped:
                batch_flipped = torch.stack(crops_flipped).to(self.device)
                embeddings_flipped = self.model(batch_flipped)
                embeddings = (embeddings + embeddings_flipped) / 2

        embeddings = embeddings.cpu().numpy().astype(np.float32)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        embeddings = embeddings / norms

        # Build result with zeros for invalid crops
        result = [np.zeros(512, dtype=np.float32) for _ in bboxes]
        for idx, emb in zip(valid_indices, embeddings):
            result[idx] = emb

        return result

    def match(
        self,
        embedding: np.ndarray,
        references: Dict[int, np.ndarray],
        threshold: float = 0.5,
    ) -> Tuple[Optional[int], float]:
        """
        Match an embedding to reference embeddings.

        Args:
            embedding: 512-D query embedding
            references: Dict mapping player_id to 512-D embedding
            threshold: Minimum cosine similarity to consider a match

        Returns:
            (best_match_id, similarity) or (None, 0.0) if no match above threshold
        """
        if not references:
            return None, 0.0

        best_id = None
        best_sim = 0.0

        for ref_id, ref_emb in references.items():
            # Cosine similarity (embeddings are already L2 normalized)
            sim = float(np.dot(embedding, ref_emb))

            if sim > best_sim:
                best_sim = sim
                best_id = ref_id

        if best_sim >= threshold:
            return best_id, best_sim

        return None, 0.0

    def match_all(
        self,
        embeddings: List[np.ndarray],
        references: Dict[int, np.ndarray],
        threshold: float = 0.5,
    ) -> List[Tuple[Optional[int], float]]:
        """
        Match multiple embeddings to references using Hungarian algorithm
        to prevent double-matching.

        Args:
            embeddings: List of 512-D query embeddings
            references: Dict mapping player_id to 512-D embedding
            threshold: Minimum similarity threshold

        Returns:
            List of (matched_id, similarity) tuples, one per input embedding
        """
        if not embeddings or not references:
            return [(None, 0.0) for _ in embeddings]

        ref_ids = list(references.keys())
        ref_embs = np.array([references[rid] for rid in ref_ids])
        query_embs = np.array(embeddings)

        # Compute similarity matrix
        sim_matrix = query_embs @ ref_embs.T  # (num_queries, num_refs)

        # Greedy matching (simple approach - assign best available)
        results = [(None, 0.0) for _ in embeddings]
        used_refs = set()

        # Sort by best similarity first
        matches = []
        for i in range(len(embeddings)):
            for j in range(len(ref_ids)):
                matches.append((sim_matrix[i, j], i, j))
        matches.sort(reverse=True)

        used_queries = set()
        for sim, q_idx, r_idx in matches:
            if q_idx in used_queries or r_idx in used_refs:
                continue
            if sim >= threshold:
                results[q_idx] = (ref_ids[r_idx], float(sim))
                used_queries.add(q_idx)
                used_refs.add(r_idx)

        return results
