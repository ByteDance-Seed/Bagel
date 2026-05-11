# Optional runtime dependencies. All imports are best-effort.

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

try:
    from transformers import AutoProcessor, Owlv2ForObjectDetection
except Exception:
    AutoProcessor = None
    Owlv2ForObjectDetection = None

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except Exception:
    VGGT = None
    load_and_preprocess_images = None
    pose_encoding_to_extri_intri = None
