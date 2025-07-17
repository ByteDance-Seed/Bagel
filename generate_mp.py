# -*- coding: utf-8 -*-

import os, argparse, copy, torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from data.data_utils import add_special_tokens
from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from PIL import Image
    
class BagelImageGeneratorAccelerate:
    def __init__(self, model_path, resolution=1024, width=None, height=None, max_latent_size=64, max_memory_per_gpu="20GB", vae_device='cuda:1'):
        self.resolution = resolution
        self.max_latent_size = max_latent_size
        self.max_memory_per_gpu = max_memory_per_gpu
        
        seed = 42
        self._set_seed(seed)
        
        print(f"Initializing model with Accelerate device mapping")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        
        self._load_configs(model_path)
        self._initialize_models_with_device_map(model_path)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)
        self.vae_model = self.vae_model.to(vae_device).eval()

        self.SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

        print("Model loaded successfully with Accelerate device mapping")
        self._print_device_map()

    def _set_seed(self, seed):
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
    
    def _load_configs(self, model_path):
        self.llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        self.llm_config.qk_norm = True
        self.llm_config.tie_word_embeddings = False
        self.llm_config.layer_module = "Qwen2MoTDecoderLayer"

        self.vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        self.vit_config.rope = False
        self.vit_config.num_hidden_layers = self.vit_config.num_hidden_layers - 1

        self.vae_model, self.vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    
    def _initialize_models_with_device_map(self, model_path):        
        # Create model config
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=self.llm_config, 
            vit_config=self.vit_config,
            vae_config=self.vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=self.max_latent_size,
        )
        
        # Initialize empty model to calculate memory requirements
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(self.llm_config)
            vit_model = SiglipVisionModel(self.vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(self.vit_config)
        
        # Get available memory for each GPU
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available")
        
        max_memory = {}
        for i in range(num_gpus):
            max_memory[i] = self.max_memory_per_gpu
        
        print(f"Memory allocation: {max_memory}")
        
        # Automatically infer device map
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["Qwen2MoTDecoderLayer", "SiglipEncoderLayer"],  # Don't split these modules
            dtype=torch.float16
        )
        
        print("Device map:", device_map)
        model_state_dict_path = os.path.join(model_path, "ema.safetensors")
        
        self.model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_state_dict_path,
            device_map=device_map,
            dtype=torch.float16,
            force_hooks=True,
            strict=False,
            offload_folder=None,  # 不使用CPU offload
            offload_state_dict=False,
        )
        
        self.model.eval()
        self.device_map = device_map
    
    def _print_device_map(self):
        print("\n" + "="*50)
        print("MODEL DEVICE MAPPING")
        print("="*50)
        
        for name, device in self.device_map.items():
            print(f"{name}: {device}")
        
        print("="*50 + "\n")
    
    def _get_primary_device(self):
        devices = list(self.device_map.values())
        device_counts = {}
        for device in devices:
            if isinstance(device, int):
                device = f"cuda:{device}"
            device_counts[device] = device_counts.get(device, 0) + 1
        
        primary_device = max(device_counts, key=device_counts.get)
        return primary_device
    
    def _move_inputs_to_device(self, inputs, target_device):
        if isinstance(inputs, dict):
            return {k: self._move_inputs_to_device(v, target_device) for k, v in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            return type(inputs)(self._move_inputs_to_device(item, target_device) for item in inputs)
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(target_device)
        else:
            return inputs
        
    def to_module_device(self, tensor, module):
        if tensor is None:
            return None
        device = None
        
        try:
            device = next(module.parameters()).device
        except StopIteration:
            try:
                device = next(module.buffers()).device
            except StopIteration:
                for child in module.modules():
                    try:
                        device = next(child.parameters()).device
                        break
                    except StopIteration:
                        continue
        if device is not None:
            return tensor.to(device, non_blocking=True)
        return tensor
    
    def generate_with_thinking(self, prompt, num_timesteps=50, cfg_scale=4.0, 
                              cfg_interval=[0.4, 1.0], cfg_renorm_min=0., 
                              timestep_shift=3.0, max_length=2048, simple_think=False):
        
        h, w = self.resolution, self.resolution
        primary_device = self._get_primary_device()
        
        past_key_values = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]
        
        # System prompt
        generation_input, newlens, new_rope = self.model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[self.SYSTEM_PROMPT],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._move_inputs_to_device(generation_input, primary_device)
        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)  
            
        # CFG preparation
        generation_input_cfg = self.model.prepare_vae_latent_cfg(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            image_sizes=[(h, w)], 
        )
        generation_input_cfg = self._move_inputs_to_device(generation_input_cfg, primary_device)
        
        # User prompt
        generation_input, newlens, new_rope = self.model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._move_inputs_to_device(generation_input, primary_device)
        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)  
            
        # Generate thinking process
        tmp_past_key_values = copy.deepcopy(past_key_values)
        tmp_generation_input = self.model.prepare_start_tokens(newlens, new_rope, self.new_token_ids)
        tmp_generation_input = self._move_inputs_to_device(tmp_generation_input, primary_device)
        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.model.generate_text(
                past_key_values=tmp_past_key_values,
                max_length=max_length,
                do_sample=True,
                temperature=0.3,
                end_token_id=self.new_token_ids['eos_token_id'],
                **tmp_generation_input,
            )
            output = self.tokenizer.decode(unpacked_latent[:,0])
            think_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]  
        
        print("="*30, "original think", "="*30)
        print(think_output) 
        
        if simple_think:
            think_output_list = think_output.split("</think>")
            if len(think_output_list) > 1 and think_output_list[1] != "":
                think_output = think_output_list[1].strip()
            print("="*30, "processed think", "="*30)
            print(think_output) 
        
        # Add thinking output to context
        generation_input, newlens, new_rope = self.model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[think_output],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._move_inputs_to_device(generation_input, primary_device)
        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)

        # Prepare for image generation
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            image_sizes=[(h, w)], 
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._move_inputs_to_device(generation_input, primary_device)

        # Generate image
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale, 
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type="global",
                cfg_text_past_key_values=None,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                **generation_input,
            )
        
        # Decode image
        latent0 = unpacked_latent[0]
        latent0 = latent0.reshape(1, h//16, w//16, 2, 2, 16)
        latent0 = torch.einsum("nhwpqc->nchpwq", latent0)
        latent0 = latent0.reshape(1, 16, h//8, w//8)
        latent0 = self.to_module_device(latent0,self.vae_model)
        image = self.vae_model.decode(latent0)
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        
        return tmpimage, think_output

    def generate(self, prompt, num_timesteps=50, cfg_scale=4.0, cfg_interval=[0.4, 1.0], cfg_renorm_min=0., timestep_shift=3.0):
        primary_device = self._get_primary_device()
        
        past_key_values = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        generation_input, newlens, new_rope = self.model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._move_inputs_to_device(generation_input, primary_device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)

        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            image_sizes=[(self.resolution, self.resolution)], 
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._move_inputs_to_device(generation_input, primary_device)

        cfg_past_key_values = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        cfg_newlens = [0]
        cfg_new_rope = [0]

        generation_input_cfg = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_newlens,
            curr_rope=cfg_new_rope, 
            image_sizes=[(self.resolution, self.resolution)],
        )
        generation_input_cfg = self._move_inputs_to_device(generation_input_cfg, primary_device)
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                unpacked_latent = self.model.generate_image(
                    past_key_values=past_key_values,
                    num_timesteps=num_timesteps,
                    cfg_text_scale=cfg_scale,
                    cfg_interval=cfg_interval,
                    cfg_renorm_min=cfg_renorm_min,
                    timestep_shift=timestep_shift,
                    cfg_text_past_key_values=cfg_past_key_values,
                    cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                    cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                    cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                    cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                    **generation_input,
                )

        latent = unpacked_latent[0]
        latent = latent.reshape(1, self.resolution//16, self.resolution//16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, self.resolution//8, self.resolution//8)
        latent = self.to_module_device(latent,self.vae_model)
        image = self.vae_model.decode(latent)
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)

        return tmpimage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_memory_per_gpu", type=str, default="20GB", help="Maximum memory per GPU")
    parser.add_argument("--output_path", type=str, default="generated_output.png")
    parser.add_argument("--prompt", type=str, default='Generate a cat.')
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--local-rank", type=int, default=-1)
    args = parser.parse_args()
    
    generator = BagelImageGeneratorAccelerate(
        model_path=args.model_path,
        resolution=args.resolution,
        max_memory_per_gpu=args.max_memory_per_gpu
    )

    if args.think:
        image, thinking = generator.generate_with_thinking(
            args.prompt,
            cfg_scale=args.cfg_scale,
            num_timesteps=args.num_timesteps
        )
        print("Thinking process:", thinking)
        with open(args.output_path.replace(".png", "_thinking.txt"), "w") as f:
            f.write(thinking)
    else:
        image = generator.generate(
            args.prompt,
            cfg_scale=args.cfg_scale,
            num_timesteps=args.num_timesteps
        )
    image.save(args.output_path)
    print(f"Image saved to {args.output_path}")