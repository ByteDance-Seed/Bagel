import asyncio
import websockets
import numpy as np
from msgpack_numpy import Packer, unpackb

ur5_cleaning_command =[
    "pick up the tea bottle",
    "pick up the plastic lid",
    'pick up yellow cup',
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
    # "put beige bowl in bin",
    # "pick up orange wrapper",
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

ROLL_OUT=False
if ROLL_OUT:
    editing_command = ur5_roll_out
else:
    editing_command = ur5_cleaning_command

async def connect_and_send():
    uri = "ws://0.0.0.0:8000"
    try:
        # Increase the timeout time by setting ping_interval and ping_timweout
        async with websockets.connect(uri, ping_interval=20, ping_timeout=12000) as ws:

            from PIL import Image

            source_image_path = "roll_outs/drop_bowl.png"

            for i, editing_instruction in enumerate(editing_command):
                source_image = np.array(Image.open(source_image_path).resize((224, 224)))
            
                inputs = {
                    "observation/base_0_camera/rgb/image": source_image,
                    "observation/left_wrist_0_camera/rgb/image": source_image,
                    "raw_text": editing_instruction,
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
                target_image_path = f"roll_outs_test/ur5s4_b0_edited_{prompt}.png"
                output_image = Image.fromarray(output['future/observation/base_0_camera/rgb/image'])
                output_image.save(target_image_path)
                print(f"Image saved as {target_image_path}")

                if ROLL_OUT:
                    source_image_path = target_image_path
    
    except websockets.ConnectionClosedOK:
        print("Server closed the connection – goodbye!")
    except websockets.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")

if __name__ == "__main__":
    asyncio.run(connect_and_send())
