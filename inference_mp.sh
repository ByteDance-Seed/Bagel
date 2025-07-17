CUDA_VISIBLE_DEVICES=0,1 python generate_mp.py --model_path /path/to/your/model --prompt 'Generate a cat.'

CUDA_VISIBLE_DEVICES=0,1 python edit_mp.py --model_path /path/to/your/model --prompt 'Change the cat to a dog.' --input_image ./asset/cat.jpg --think