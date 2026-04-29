"""
MPS-Optimized Face Swapping Pipeline
Designed to preserve maximum resolution and detail for the enhancement stage
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Optional, List
import insightface
from core.device_manager import get_device_manager


class MPS_FaceSwapper:
    """
    High-quality face swapping optimized for Apple Silicon.
    Preserves original image resolution and minimizes artifacts.
    """
    
    def __init__(self):
        """Initialize face swapper with optimal settings for MPS."""
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        
        # Use insightface with CPU provider for MPS compatibility
        # (GPU providers may not work reliably on Apple Silicon)
        try:
            # Initialize swapper with CPU context for stability
            providers = [('CPUExecutionProvider',)]
            
            self.swapper = insightface.model_zoo.get_model(
                'inswapper_128.onnx',
                providers=providers
            )
            
            self.face_analyser = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=providers,
                root=None
            )
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
            
            print("✓ MPS_FaceSwapper initialized with CPU providers (ONNX Runtime)")
            
        except Exception as e:
            print(f"⚠️  Error initializing face swapper: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[insightface.utils.Face]:
        """
        Detect all faces in image using buffalo model.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            List of Face objects with embeddings
        """
        try:
            faces = self.face_analyser.get(image)
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def swap_face(self, source_image: np.ndarray, target_image: np.ndarray,
                  source_faces: List[insightface.utils.Face],
                  target_faces: List[insightface.utils.Face],
                  source_face_index: int = 0,
                  target_face_index: int = 0) -> Tuple[np.ndarray, bool]:
        """
        Swap face from source image to target image.
        Preserves original resolution.
        
        Args:
            source_image: Source image (BGR, face source)
            target_image: Target image (BGR, where face will be swapped)
            source_faces: Detected faces in source image
            target_faces: Detected faces in target image
            source_face_index: Which source face to use (0 = first)
            target_face_index: Which target face to replace (0 = first)
        
        Returns:
            Tuple of (swapped_image, success)
        """
        try:
            if not source_faces or not target_faces:
                return target_image, False
            
            source_face = source_faces[min(source_face_index, len(source_faces)-1)]
            target_face = target_faces[min(target_face_index, len(target_faces)-1)]
            
            # Perform swap
            swapped = self.swapper.get(target_image, target_face, source_face, 
                                       paste_back=True)
            
            return swapped, True
            
        except Exception as e:
            print(f"Face swap failed: {e}")
            return target_image, False
    
    def extract_face_region(self, image: np.ndarray, 
                           face: insightface.utils.Face) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """
        Extract face region with bounding box and mask.
        
        Args:
            image: Original image (BGR)
            face: Detected face object
        
        Returns:
            Tuple of (face_region, mask, bbox)
        """
        try:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Extract face region
            face_region = image[y1:y2, x1:x2].copy()
            
            # Create face mask from landmarks
            mask = self._create_face_mask(face_region, face)
            
            # Return face, mask, and bbox in (x, y, w, h) format
            return face_region, mask, (x1, y1, x2-x1, y2-y1)
            
        except Exception as e:
            print(f"Error extracting face region: {e}")
            return None, None, None
    
    def _create_face_mask(self, face_image: np.ndarray, 
                         face: insightface.utils.Face) -> np.ndarray:
        """
        Create face mask from detected landmarks.
        Mask is white (255) for face, black (0) for background.
        """
        try:
            h, w = face_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Get landmarks and translate to face region coordinates
            if hasattr(face, 'landmark_3d') and face.landmark_3d is not None:
                landmarks = face.landmark_3d
            else:
                landmarks = face.kps if hasattr(face, 'kps') else None
            
            if landmarks is not None:
                # Translate landmarks to face region
                landmarks_local = landmarks.copy()
                landmarks_local[:, 0] -= face.bbox[0]
                landmarks_local[:, 1] -= face.bbox[1]
                
                # Create convex hull from landmarks
                hull = cv2.convexHull(landmarks_local.astype(np.int32))
                
                # Fill the mask
                cv2.drawContours(mask, [hull], 0, 255, -1)
                
                # Expand mask slightly for better blending
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                # Fallback: use entire face region
                mask[:, :] = 255
            
            return mask
            
        except Exception as e:
            print(f"Mask creation error: {e}")
            # Return full white mask as fallback
            return np.ones((face_image.shape[0], face_image.shape[1]), dtype=np.uint8) * 255
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'swapper'):
            del self.swapper
        if hasattr(self, 'face_analyser'):
            del self.face_analyser
