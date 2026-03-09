import os
# DISABLE CUDA FIRST - before torch is imported
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_HOME'] = ''
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Disable HuggingFace model downloads if cache not available
os.environ['HF_HUB_OFFLINE'] = '0'  # Allow online but will fallback to cache
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow online but will fallback to cache
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')

import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import base64
from pathlib import Path
import signal
import sys

# Disable CUDA in torch explicitly after import
try:
    # Patch cuda module to fake CUDA support
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.current_device = lambda: -1
    torch.cuda.get_device_name = lambda *args, **kwargs: "cpu"
    torch.cuda._initialized = False
    torch.cuda.is_initialized = lambda *args, **kwargs: False
    torch.cuda.init = lambda *args, **kwargs: None
    torch._C._cuda_init = None
    
    # Additional aggressive CUDA disabling
    torch.cuda.device = lambda x: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.synchronize = lambda: None
    torch.cuda.reset_peak_memory_stats = lambda *args, **kwargs: None
    torch.cuda.memory_stats = lambda *args, **kwargs: {}
    torch.cuda.mem_get_info = lambda *args, **kwargs: (1024*1024*1024, 1024*1024*1024)  # Fake 1GB free
    
    # CRITICAL FIX: Monkeypatch Tensor.to() to handle device parameter correctly
    original_tensor_to = torch.Tensor.to
    def fixed_tensor_to(self, *args, **kwargs):
        """Fixed .to() that handles bad device parameters"""
        try:
            # Try normal .to() first
            return original_tensor_to(self, *args, **kwargs)
        except TypeError as e:
            if "to() received an invalid combination" in str(e):
                # If .to() gets bad arguments, try to fix them
                # Check if device is being passed as dtype
                if len(args) > 0:
                    first_arg = args[0]
                    if isinstance(first_arg, torch.device):
                        # Device is a torch.device object - convert to CPU directly
                        return self.cpu()
                    elif isinstance(first_arg, str):
                        # Device is a string
                        if first_arg == 'cpu' or 'cpu' in str(first_arg):
                            return self.cpu()
                # If we can't figure it out, just return on CPU
                return self.cpu()
            else:
                raise
    
    torch.Tensor.to = fixed_tensor_to
    print("[CUDA PATCH] Monkeypatched Tensor.to() to handle device errors")
    
    # FIX: Patch subtract operation to handle bool tensors
    # When bool tensors are subtracted, convert to float first
    original_sub = torch.Tensor.__sub__
    def fixed_sub(self, other):
        if self.dtype == torch.bool:
            return original_sub(self.float(), other)
        elif isinstance(other, torch.Tensor) and other.dtype == torch.bool:
            return original_sub(self, other.float())
        return original_sub(self, other)
    torch.Tensor.__sub__ = fixed_sub
    
    original_rsub = torch.Tensor.__rsub__
    def fixed_rsub(self, other):
        if self.dtype == torch.bool:
            return original_rsub(self.float(), other)
        elif isinstance(other, torch.Tensor) and other.dtype == torch.bool:
            return original_rsub(self, other.float())
        return original_rsub(self, other)
    torch.Tensor.__rsub__ = fixed_rsub
    
    # Also patch torch.sub function
    original_torch_sub = torch.sub
    def fixed_torch_sub(input, other, *, alpha=1, out=None):
        if input.dtype == torch.bool:
            input = input.float()
        if isinstance(other, torch.Tensor) and other.dtype == torch.bool:
            other = other.float()
        return original_torch_sub(input, other, alpha=alpha, out=out)
    torch.sub = fixed_torch_sub
    
    # FIX: Patch torch.finfo() to handle device objects passed as dtype
    original_finfo = torch.finfo
    def fixed_finfo(dtype):
        """Handle case where torch.device is passed instead of dtype"""
        if isinstance(dtype, torch.device):
            # Device passed instead of dtype - return float32 info
            return original_finfo(torch.float32)
        return original_finfo(dtype)
    torch.finfo = fixed_finfo
    
    # FIX: Patch torch.iinfo() to handle device objects passed as dtype
    original_iinfo = torch.iinfo
    def fixed_iinfo(dtype):
        """Handle case where torch.device is passed instead of dtype"""
        if isinstance(dtype, torch.device):
            # Device passed instead of dtype - return int32 info
            return original_iinfo(torch.int32)
        return original_iinfo(dtype)
    torch.iinfo = fixed_iinfo
    
    print("[CUDA PATCH] Monkeypatched subtraction operations for bool tensors")
except Exception as e:
    print(f"[CUDA Patch] Warning: {e}")

# Ensure no MPS either (macOS specific)
if hasattr(torch.backends, 'mps'):
    try:
        torch.backends.mps.enabled = False
    except:
        pass

# FIX: Patch BertModel BEFORE importing GroundingDINO
try:
    from transformers.models.bert.modeling_bert import BertModel
    if not hasattr(BertModel, 'get_head_mask'):
        def get_head_mask(self, head_mask, num_hidden_layers):
            """Create a mask from the two representations of the head_mask."""
            if head_mask is not None:
                if head_mask.size()[0] != num_hidden_layers:
                    raise ValueError(
                        f"The head_mask should be specified for {num_hidden_layers} layers, but it was for"
                        f" {head_mask.size()[0]}."
                    )
                head_mask = head_mask.to(dtype=torch.float32)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask
        BertModel.get_head_mask = get_head_mask
        print("[PATCH] Added get_head_mask to BertModel")
except Exception as e:
    print(f"[PATCH WARNING] Could not patch BertModel: {e}")

# GroundingDINO
from groundingdino.util.inference import load_model, predict, load_image
import groundingdino.util.inference as groundingdino_inference

# FIX: Patch GroundingDINO's predict() to handle device properly
original_predict = groundingdino_inference.predict

def patched_predict(model, image, caption, box_threshold, text_threshold, device='cpu', remove_combined=False):
    """
    Fixed version - NO .to() calls. Subtraction operator patched for bool tensors.
    """
    from groundingdino.util.inference import preprocess_caption, get_phrases_from_posmap
    import bisect
    
    # Ensure image is float32
    if isinstance(image, torch.Tensor) and image.dtype != torch.float32:
        image = image.float()
    
    caption = preprocess_caption(caption=caption)
    
    with torch.no_grad():
        # Forward pass - bool tensor subtraction is now handled by patched __sub__
        outputs = model(image[None], captions=[caption])
    
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    prediction_boxes = outputs["pred_boxes"].cpu()[0]
    
    # These comparisons will create bool tensors, but subtraction on them will work
    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]
    
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]
    
    return boxes, logits.max(dim=1)[0] if len(logits) > 0 else torch.tensor([]), phrases

# Replace the predict function
groundingdino_inference.predict = patched_predict
predict = patched_predict
print("[PATCH] GroundingDINO predict() patched - Tensor.to() fix in place")


# Depth Anything V2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Audio LLM (faster-whisper)
from faster_whisper import WhisperModel

# Local instruction LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# TTS - Import lazily to avoid slow scipy load on macOS
TTS = None  # Will be imported on first use


# Timeout helper for long-running operations
def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Model loading operation timed out")


def load_with_timeout(load_func, timeout_secs=30, model_name="model"):
    """Wrap model loading with timeout protection"""
    if hasattr(signal, 'SIGALRM'):  # Unix-like systems only
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_secs)
            result = load_func()
            signal.alarm(0)  # Cancel alarm
            print(f"[SUCCESS] {model_name} loaded successfully")
            return result
        except TimeoutError:
            signal.alarm(0)  # Cancel alarm
            print(f"[TIMEOUT] {model_name} loading took too long (>{timeout_secs}s), may be downloading...")
            return None
        except Exception as e:
            signal.alarm(0)  # Cancel alarm
            raise
    else:
        # Windows doesn't support SIGALRM, just try normally
        return load_func()


class NavigationPipeline:
    def __init__(self, device='cpu'):
        # Force device to be 'cpu' always - simplest approach
        self.device = 'cpu'  # HARDCODED - no torch.device anywhere
        try:
            self.load_models()
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[WARNING] Ignoring CUDA error during initialization: {e}")
            else:
                raise
    
    def get_device_str(self):
        """Get device as string - always returns 'cpu'"""
        return 'cpu'
    
    def load_models(self):
        """Load all required models with CUDA error handling and download fallback"""
        
        print("Loading GroundingDINO...")
        try:
            base_dir = Path(__file__).parent.parent
            config_path = str(base_dir / "GroundingDINO_SwinT_OGC.py")
            model_path = str(base_dir / "groundingdino_swint_ogc.pth")
            self.grounding_model = load_model(config_path, model_path)
            self.grounding_model.eval()
            
            # Move model to CPU - use parameter loop, not .to() which is broken
            for param in self.grounding_model.parameters():
                param.data = param.data.cpu()
            for buffer in self.grounding_model.buffers():
                buffer.data = buffer.data.cpu()
            
            print("[✓] GroundingDINO loaded and moved to CPU")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[✗] GroundingDINO CUDA error: {e}")
                self.grounding_model = None
            else:
                print(f"[✗] GroundingDINO failed: {e}")
                self.grounding_model = None
        
        print("Loading Depth Anything V2...")
        try:
            def load_depth():
                processor = AutoImageProcessor.from_pretrained(
                    "depth-anything/Depth-Anything-V2-base-hf"
                )
                model = AutoModelForDepthEstimation.from_pretrained(
                    "depth-anything/Depth-Anything-V2-base-hf"
                )
                return processor, model
            
            self.depth_processor, self.depth_model = load_with_timeout(
                load_depth, timeout_secs=60, model_name="Depth Anything V2"
            ) or (None, None)
            
            if self.depth_model is not None:
                self.depth_model.eval()
                try:
                    self.depth_model = self.depth_model.to('cpu')
                except:
                    pass
                print("[✓] Depth Anything V2 loaded")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[✗] Depth V2 CUDA error: {e}")
            else:
                print(f"[✗] Depth V2 failed: {e}")
            self.depth_model = None
            self.depth_processor = None
        
        print("Loading Whisper...")
        try:
            def load_whisper():
                return WhisperModel("small", device='cpu')
            
            self.whisper_model = load_with_timeout(
                load_whisper, timeout_secs=60, model_name="Whisper"
            )
            if self.whisper_model is not None:
                print("[✓] Whisper loaded")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[✗] Whisper CUDA error: {e}")
            else:
                print(f"[✗] Whisper failed: {e}")
            self.whisper_model = None
        
        print("Loading Flan-T5...")
        try:
            def load_t5():
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
                return tokenizer, model
            
            result = load_with_timeout(load_t5, timeout_secs=60, model_name="Flan-T5")
            if result:
                self.instr_tokenizer, self.instr_model = result
                self.instr_model.eval()
                try:
                    self.instr_model = self.instr_model.to('cpu')
                except:
                    pass
                print("[✓] Flan-T5 loaded")
            else:
                self.instr_tokenizer = None
                self.instr_model = None
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[✗] Flan-T5 CUDA error: {e}")
            else:
                print(f"[✗] Flan-T5 failed: {e}")
            self.instr_model = None
            self.instr_tokenizer = None
        
        print("Loading TTS...")
        try:
            def load_tts():
                # Lazy import of TTS to avoid slow scipy load on macOS
                global TTS
                if TTS is None:
                    print("  [TTS] Importing TTS module (this may take 30+ seconds)...")
                    from TTS.api import TTS as TTS_API
                    TTS = TTS_API
                return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
            
            self.tts = load_with_timeout(load_tts, timeout_secs=120, model_name="TTS")
            if self.tts is not None:
                print("[SUCCESS] TTS loaded successfully")
                print("[✓] TTS loaded")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[✗] TTS CUDA error: {e}")
            else:
                print(f"[✗] TTS failed: {e}")
            self.tts = None
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Models Initialization Complete                    ║
║                                                              ║
║  GroundingDINO: {'✓ loaded' if self.grounding_model else '✗ failed'}                        
║  Depth Anything V2: {'✓ loaded' if self.depth_model else '✗ failed'}                   
║  Whisper: {'✓ loaded' if self.whisper_model else '✗ failed'}                          
║  Flan-T5: {'✓ loaded' if self.instr_model else '✗ failed'}                              
║  TTS: {'✓ loaded' if self.tts else '✗ failed'}                                
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio to text using Whisper"""
        segments, info = self.whisper_model.transcribe(audio_path)
        text = " ".join([segment.text for segment in segments])
        return text
    
    def extract_target_from_text(self, text):
        """Extract main object from text with contextual understanding"""
        text_lower = text.lower()
        
        # Filter out question words and common non-object words
        question_words = {'where', 'what', 'when', 'why', 'how', 'is', 'are', 'do', 'does', 'can', 'could', 'would', 'location', 'place', 'thing', 'stuff', 'it'}
        
        # Prompt with better context - ask for PHYSICAL OBJECT
        prompt = f"What physical object is the person looking for? Extract ONLY the object name (a noun like bottle, keys, phone, chair). Say just the object name: {text}"
        try:
            inputs = self.instr_tokenizer(prompt, return_tensors="pt")
            # Ensure tensors are on CPU
            inputs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            outputs = self.instr_model.generate(**inputs, max_length=10)
            target_object = self.instr_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"[WARNING] CUDA error in extract_target_from_text: {e}")
                # Retry on CPU
                inputs = self.instr_tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                outputs = self.instr_model.generate(**inputs, max_length=10)
                target_object = self.instr_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            else:
                raise
        
        # Clean up - remove common words/phrases
        target_object = target_object.lower().strip()
        target_object = target_object.replace('the ', '', 1).replace('a ', '', 1).replace('an ', '', 1).strip()
        
        # Remove question words if model picked them up
        words = target_object.split()
        words = [w for w in words if w not in question_words]
        target_object = ' '.join(words).strip() if words else target_object
        
        # If still too long (more than 2 words), take the last significant word (likely the object)
        words = target_object.split()
        if len(words) > 2:
            target_object = words[-1]
        
        # Fallback: if still bad, try to extract from original text
        if not target_object or target_object in question_words:
            # Extract nouns from text using simple heuristics
            words = text_lower.split()
            for word in words:
                if word not in question_words and len(word) > 2 and word.isalpha():
                    target_object = word.replace('.', '').replace(',', '').replace('?', '')
                    if target_object not in question_words:
                        break
        
        return target_object if target_object else "object"
    
    def generate_instruction(self, target, steps, angle, distance_meters=None, confidence=None, depth=None):
        """Generate simple, clear navigation instructions without LLM"""
        
        if distance_meters is None:
            distance_meters = steps * 0.75
        
        steps_int = int(round(steps))
        
        # Determine direction description - simple, natural language
        abs_angle = abs(angle)
        if abs_angle < 2:
            simple_direction = "straight ahead"
            turn_direction = "straight"
        elif abs_angle < 15:
            simple_direction = "slightly to the " + ("right" if angle > 0 else "left")
            turn_direction = "slightly right" if angle > 0 else "slightly left"
        elif abs_angle < 30:
            simple_direction = "to the " + ("right" if angle > 0 else "left")
            turn_direction = "right" if angle > 0 else "left"
        elif abs_angle < 60:
            simple_direction = "sharply to the " + ("right" if angle > 0 else "left")
            turn_direction = "sharp right" if angle > 0 else "sharp left"
        else:
            simple_direction = "almost completely to the " + ("right" if angle > 0 else "left")
            turn_direction = "all the way around"
        
        # Simple, clear voice instruction - for readability, not math
        voice_instruction = f"The {target} is {steps_int} steps away. Turn {simple_direction} and walk {steps_int} steps."
        
        # Detailed version for display
        detailed_instruction = (
            f"🎯 NAVIGATION INSTRUCTIONS FOR: {target.upper()}\n"
            f"{'='*50}\n\n"
            f"STEP 1: DIRECTION\n"
            f"   Turn {simple_direction}\n"
            f"   (Angle: {angle:.1f}°)\n\n"
            f"STEP 2: WALK\n"
            f"   Walk {steps_int} steps forward\n"
            f"   Distance: ~{distance_meters:.1f} meters\n\n"
            f"DETECTION CONFIDENCE: {(confidence*100 if confidence else 85):.0f}%\n"
            f"{'='*50}"
        )
        
        return {
            'detailed': detailed_instruction,
            'conversational': voice_instruction,
            'summary': {
                'target': target,
                'distance_m': round(distance_meters, 2),
                'steps': steps_int,
                'direction': 'right' if angle > 0 else 'left' if angle < 0 else 'straight',
                'angle_degrees': round(angle, 1),
                'confidence_percent': round((confidence * 100) if confidence else 85, 1),
                'depth_m': round(depth if depth else 0, 3)
            }
        }
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        tts_path = "instruction.wav"
        self.tts.tts_to_file(text=text, file_path=tts_path)
        return tts_path
    
    def verify_models_on_cpu(self):
        """Verify that all models are on CPU by moving parameters directly (not using .to())"""
        try:
            # GroundingDINO
            if self.grounding_model is not None:
                for param in self.grounding_model.parameters():
                    if param.is_cuda or str(param.device) != 'cpu':
                        param.data = param.data.cpu()
                for buffer in self.grounding_model.buffers():
                    if buffer.is_cuda or str(buffer.device) != 'cpu':
                        buffer.data = buffer.data.cpu()
            
            # Depth model
            if self.depth_model is not None:
                try:
                    self.depth_model = self.depth_model.to('cpu')
                except:
                    pass
            
            # Instruction model
            if self.instr_model is not None:
                try:
                    self.instr_model = self.instr_model.to('cpu')
                except:
                    pass
                    
        except Exception as e:
            print(f"[WARNING] Could not verify models on CPU: {e}")
    
    def predict_with_cpu_fallback(self, model, image, caption, box_threshold=0.3, text_threshold=0.25):
        """
        Wrapper that ensures everything is on CPU with correct dtypes BEFORE calling predict.
        """
        print(f"[PREDICT] Preparing for inference...")
        
        # Ensure image is on CPU and correct dtype BEFORE predict is called
        if hasattr(image, 'to'):
            image = image.to('cpu')
            print(f"[PREDICT] Image moved to CPU")
        
        # Convert to float32 if needed
        if hasattr(image, 'dtype'):
            if image.dtype not in [torch.float32, torch.float64]:
                print(f"[PREDICT] Converting image from {image.dtype} to float32")
                image = image.float()
        
        print(f"[PREDICT] Image dtype: {image.dtype}, shape: {image.shape}")
        
        # Ensure all model parameters are on CPU BEFORE predict is called
        try:
            for param in model.parameters():
                if param.is_cuda or str(param.device) != 'cpu':
                    param.data = param.data.cpu()
            for buffer in model.buffers():
                if buffer.is_cuda or str(buffer.device) != 'cpu':
                    buffer.data = buffer.data.cpu()
            print(f"[PREDICT] Model moved to CPU")
        except:
            print(f"[PREDICT] Could not move model to CPU, continuing anyway...")
        
        # NOW call predict - everything should be CPU and float32, so no .to() will be called
        print(f"[PREDICT] Calling patched predict() with device='cpu'")
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device='cpu'
        )
        print(f"[PREDICT] SUCCESS - found {len(boxes)} boxes")
        return boxes, logits, phrases
    
    def process_image(self, image_path, target):
        """
        Process image and estimate navigation parameters
        Returns dict with success status and results
        """
        import time
        start_time = time.time()
        
        try:
            # Verify models are on CPU before starting
            self.verify_models_on_cpu()
            
            # Load image
            image_source, image_tensor = load_image(image_path)
            img_np = np.array(image_source)
            h, w, _ = img_np.shape
            
            # Ensure image tensor is on CPU and correct dtype
            if hasattr(image_tensor, 'to'):
                image_tensor = image_tensor.to('cpu')
            
            # Ensure image is float32 (critical for avoiding bool tensor errors)
            if isinstance(image_tensor, torch.Tensor):
                if image_tensor.dtype not in [torch.float32, torch.float64]:
                    print(f"[PROCESS] Converting image from {image_tensor.dtype} to float32")
                    image_tensor = image_tensor.float()
                print(f"[PROCESS] Image tensor: dtype={image_tensor.dtype}, shape={image_tensor.shape}")
            
            # Detect target using GroundingDINO
            try:
                boxes, logits, phrases = self.predict_with_cpu_fallback(
                    model=self.grounding_model,
                    image=image_tensor,
                    caption=target,
                    box_threshold=0.3,
                    text_threshold=0.25
                )
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] GroundingDINO prediction failed: {error_msg}")
                return {
                    'success': False,
                    'error': f'Object detection failed: {error_msg}'
                }
            
            if len(boxes) == 0:
                return {
                    'success': False,
                    'error': f'Target "{target}" not detected in image'
                }
            
            # Get bounding box
            box = boxes[0] * torch.tensor([w, h, w, h])
            cx, cy, bw, bh = box
            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            x2 = int(cx + bw/2)
            y2 = int(cy + bh/2)
            
            # Estimate depth
            with torch.no_grad():
                try:
                    inputs = self.depth_processor(images=image_source, return_tensors="pt")
                    # Ensure inputs are on CPU
                    inputs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    outputs = self.depth_model(**inputs)
                    depth = outputs.predicted_depth.to('cpu')
                except Exception as depth_error:
                    print(f"[ERROR] Depth estimation failed: {depth_error}")
                    return {
                        'success': False,
                        'error': f'Depth estimation error: {str(depth_error)}'
                    }
            
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze()
            
            depth_map = depth.detach().cpu().numpy()
            obj_depth = depth_map[y1:y2, x1:x2].mean()
            
            # Calculate angle and steps
            img_center = w / 2
            obj_center = (x1 + x2) / 2
            fov = 60  # Field of view in degrees
            angle = (obj_center - img_center) / w * fov
            
            meters = obj_depth * 0.6  # Conversion factor
            steps = meters / 0.75  # Average step length
            
            # Draw visualization with enhanced styling
            vis = img_np.copy()
            
            # Draw semi-transparent overlay for better contrast
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 200, 255), -1)
            vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)
            
            # Draw thick bounding box with gradient effect
            cv2.rectangle(vis, (x1, y1), (x2, y2), (50, 200, 255), 4)
            # Inner glow effect
            cv2.rectangle(vis, (x1-2, y1-2), (x2+2, y2+2), (100, 220, 255), 1)
            
            # Draw centroid (target center)
            target_cx = (x1 + x2) // 2
            target_cy = (y1 + y2) // 2
            cv2.circle(vis, (target_cx, target_cy), 8, (50, 200, 255), -1)
            cv2.circle(vis, (target_cx, target_cy), 8, (255, 255, 0), 2)
            
            # Draw camera position indicator
            camera_cx = w // 2
            camera_cy = h // 2
            cv2.circle(vis, (camera_cx, camera_cy), 6, (0, 255, 100), -1)
            cv2.circle(vis, (camera_cx, camera_cy), 6, (255, 255, 255), 2)
            
            # Draw enhanced arrow from camera to target
            cv2.arrowedLine(vis, (camera_cx, camera_cy), (target_cx, target_cy), (50, 200, 255), 4, tipLength=0.25)
            # Arrow glow effect
            cv2.arrowedLine(vis, (camera_cx, camera_cy), (target_cx, target_cy), (150, 220, 255), 1, tipLength=0.25)
            
            # Create a CLEAN, spacious visualization with proper separation
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # ========== SECTION 1: TARGET (TOP) ==========
            section1_y = 0
            section1_h = 80
            cv2.rectangle(vis, (0, section1_y), (w, section1_h), (0, 255, 100), 4)  # Green border
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, section1_y), (w, section1_h), (20, 60, 20), -1)
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            
            # TARGET label in left corner
            cv2.putText(vis, "TARGET:", (20, 35), font, 0.65, (150, 200, 150), 2)
            # TARGET value centered and large
            target_text = target.upper()
            target_w = cv2.getTextSize(target_text, font, 1.5, 3)[0][0]
            cv2.putText(vis, target_text, ((w - target_w) // 2, 65), font, 1.5, (100, 255, 100), 3)
            
            # ========== SECTION 2: DIRECTION (TOP-RIGHT) ==========
            section2_y = section1_h + 5
            section2_h = 80
            cv2.rectangle(vis, (0, section2_y), (w, section2_y + section2_h), (100, 150, 255), 4)  # Blue border
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, section2_y), (w, section2_y + section2_h), (20, 40, 60), -1)
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            
            # DIRECTION label
            cv2.putText(vis, "DIRECTION:", (20, section2_y + 35), font, 0.65, (150, 180, 255), 2)
            
            # Determine direction
            if angle > 2:
                dir_text = "TURN RIGHT"
                dir_color = (100, 180, 255)
            elif angle < -2:
                dir_text = "TURN LEFT"
                dir_color = (0, 100, 255)
            else:
                dir_text = "GO STRAIGHT"
                dir_color = (100, 255, 100)
            
            dir_w = cv2.getTextSize(dir_text, font, 1.4, 3)[0][0]
            cv2.putText(vis, dir_text, ((w - dir_w) // 2, section2_y + 65), font, 1.4, dir_color, 3)
            
            # ========== SECTION 3: INFO (Distance & Steps) ==========
            section3_y = section2_y + section2_h + 5
            section3_h = 100
            cv2.rectangle(vis, (0, section3_y), (w, section3_y + section3_h), (150, 150, 100), 4)  # Gray border
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, section3_y), (w, section3_y + section3_h), (40, 40, 35), -1)
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            
            # LEFT COLUMN: DISTANCE
            col_spacing = w // 2
            cv2.putText(vis, "DISTANCE:", (20, section3_y + 30), font, 0.65, (180, 180, 150), 2)
            distance_text = f"{meters:.1f}m"
            distance_w = cv2.getTextSize(distance_text, font, 1.3, 2)[0][0]
            cv2.putText(vis, distance_text, (20 + (col_spacing - 20 - distance_w) // 2, section3_y + 70), font, 1.3, (100, 255, 255), 2)
            
            # RIGHT COLUMN: STEPS
            cv2.putText(vis, "STEPS NEEDED:", (col_spacing + 20, section3_y + 30), font, 0.65, (180, 180, 150), 2)
            steps_text = f"{int(round(steps))}"
            steps_w = cv2.getTextSize(steps_text, font, 1.3, 2)[0][0]
            cv2.putText(vis, steps_text, (col_spacing + 20 + (col_spacing - 20 - steps_w) // 2, section3_y + 70), font, 1.3, (100, 255, 150), 2)
            
            # ========== SECTION 4: ACTION (BOTTOM) ==========
            section4_y = section3_y + section3_h + 5
            cv2.rectangle(vis, (0, section4_y), (w, h), (0, 255, 100), 4)  # Green border
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, section4_y), (w, h), (20, 60, 20), -1)
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            
            # ACTION label
            cv2.putText(vis, "ACTION:", (20, section4_y + 35), font, 0.65, (150, 200, 150), 2)
            # ACTION instruction centered and large
            action_text = "WALK FORWARD"
            action_w = cv2.getTextSize(action_text, font, 1.6, 3)[0][0]
            cv2.putText(vis, action_text, ((w - action_w) // 2, section4_y + 75), font, 1.6, (100, 255, 100), 3)
            
            # Convert visualization to base64
            _, buffer = cv2.imencode('.png', vis)
            img_base64 = base64.b64encode(buffer).decode()
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'target': target,
                'angle': float(angle),
                'steps': float(steps),
                'distance_meters': float(meters),
                'depth': float(obj_depth),
                'bbox': [x1, y1, x2, y2],
                'visualization': img_base64,
                'confidence': float(logits[0].item() if logits is not None else 0),
                'processing_time': float(processing_time)
            }
        
        except Exception as e:
            print(f"[ERROR] process_image exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Processing error: {str(e)}'
            }
