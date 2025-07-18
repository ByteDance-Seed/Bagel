import asyncio
import websockets
import numpy as np
from msgpack_numpy import Packer, unpackb
import argparse

ur5_cleaning_command =[
    'pick up yellow cup',
    "pick up the tea bottle",
    "pick up the plastic lid",
    "pick up orange short cup",
    "pick up orange plate",
    "pick up spoon next to the tall yellow cup",
    "pick up aluminum foil tray",
    "pick up chopstick",
    "pick up blue bowl",
    "pick up black food container",
    "pick up beige bowl",
    "pick up orange wrapper",
    "bus the table",
]


ur5_roll_out =[
    # 'pick up yellow cup',
    "put yellow cup in bin",
    "pick up beige bowl",
    "put beige bowl in bin",
    "pick up orange wrapper",
    # "throw away orange wrapper",
    # "pick up aluminum foil tray",
    # "throw away aluminum foil tray",
    # "pick up plastic bottle",
    # "throw away plastic bottle",
    # "pick up chopstick",
    # "put chopstick in bin",
    # "pick up blue bowl",
    # "put blue bowl in bin",
    # "pick up orange plate",
    # "put orange plate in bin",
]

packer = Packer()

ROLL_OUT=True
if ROLL_OUT:
    editing_command = ur5_roll_out
else:
    editing_command = ur5_cleaning_command

async def connect_and_send(args):
    uri = f"ws://0.0.0.0:{args.port}"
    try:
        # Increase the timeout time by setting ping_interval and ping_timweout
        async with websockets.connect(uri, ping_interval=20, ping_timeout=12000) as ws:

            from PIL import Image

            source_image_path = "generated_images/generation_inspect/input_0_source_0.png"
            source_image_path_2 = "generated_images/generation_inspect/input_0_source_1.png"
            source_image_path_3 = "generated_images/generation_inspect/input_0_source_2.png"

            for i, editing_instruction in enumerate(editing_command):
                source_image = np.array(Image.open(source_image_path).resize((224, 224)))
                source_image_2 = np.array(Image.open(source_image_path_2).resize((224, 224)))
                source_image_3 = np.array(Image.open(source_image_path_3).resize((224, 224)))
                inputs = {
                    "observation/base_0_camera/rgb/image": source_image,
                    "observation/wrist_0_camera/rgb/image": source_image_2,
                    "observation/left_wrist_0_camera/rgb/image": source_image_2,
                    "observation/right_wrist_0_camera/rgb/image": source_image_3,
                    "raw_text": "fold all laundry", #editing_instruction,
                }
                args = (inputs,) 
                kwargs = {
                    "cfg_text_scale": 6.0,
                    "cfg_img_scale": 1.25
                }
                data = packer.pack((args, kwargs))

                import time
                start_time = time.time()
                await ws.send(data)
                response = await ws.recv()
                end_time = time.time()
                print(f"Time taken for sending and receiving: {end_time - start_time} seconds")

                output = unpackb(response)

                prompt = editing_instruction.replace(" ", "_")
                if ROLL_OUT:
                    prompt = f"turn{i}_{prompt}"
                import os
                file_count = len([name for name in os.listdir("generated_images/roll_outs") if os.path.isfile(os.path.join("generated_images/roll_outs", name))])
                target_image_path = f"generated_images/roll_outs/ur5s4_b0_edited_{prompt}_b0_{file_count}.png"
                target_image_path_2 = f"generated_images/roll_outs/ur5s4_b0_edited_{prompt}_w0.png"
                target_image_path_3 = f"generated_images/roll_outs/ur5s4_b0_edited_{prompt}_w1.png"

                Image.fromarray(output['future/observation/base_0_camera/rgb/image']).save(target_image_path)
                # Image.fromarray(output['future/observation/wrist_0_camera/rgb/image']).save(target_image_path_2)
                Image.fromarray(output['future/observation/left_wrist_0_camera/rgb/image']).save(target_image_path_2)
                Image.fromarray(output['future/observation/right_wrist_0_camera/rgb/image']).save(target_image_path_3)
                print(f"Image saved as {target_image_path}")
                print(f"Image saved as {target_image_path_2}")
                print(f"Image saved as {target_image_path_3}")

                if ROLL_OUT:
                    source_image_path = target_image_path
                    source_image_path_2 = target_image_path_2
                    source_image_path_3 = target_image_path_3

    except websockets.ConnectionClosedOK:
        print("Server closed the connection – goodbye!")
    except websockets.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--image_key_str", type=str, default="image_0,image_2")
    parser.add_argument("--image_save_dir", type=str, default="port_8000")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(connect_and_send(args))
