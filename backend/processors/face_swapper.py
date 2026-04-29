import os
import insightface
from insightface.model_zoo import get_model
import numpy as np

class FaceSwapper:
    """
    Inference Stage:
    1. Load 'inswapper_128' model using InsightFace model_zoo.
    2. Input: source_face (from source image), target_face (from target image).
    3. Output: Swapped face region blended into the target image.
    """
    def __init__(self, model_path=None, providers=None):
        import os
        if model_path is None:
            model_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')

        if providers is None:
            # CoreML can fail to compile/save models on some macOS setups.
            providers = ['CPUExecutionProvider']
        
        # Specify providers to avoid issues with GPU on non-supported hardware
        self.model = get_model(model_path, providers=providers)

    def swap(self, target_img, target_face, source_face):
        """
        Performs the face swap.
        target_img: np.ndarray (the full target image)
        target_face: The face object detected in the target image
        source_face: The face object detected in the source image
        """
        # inswapper_128.get takes (image, target_face, source_face, paste_back=True)
        return self.model.get(target_img, target_face, source_face, paste_back=True)
