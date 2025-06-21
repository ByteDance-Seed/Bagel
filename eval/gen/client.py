import asyncio
import websockets
import numpy as np
from msgpack_numpy import Packer, unpackb

ur5_cleaning_text =[
    'pick up yellow cup',
    "put yellow cup in bin",
    "pick up chopstick",
    "put chopstick in bin",
    "pick up foil tray",
    "throw away foil tray",
    "pick up orange plate",
    "put orange plate in bin",
    "pick up blue bowl",
    "throw away blue bowl",
]
async def connect_and_send():
    uri = "ws://localhost:8788"
    try:
        # Increase the timeout time by setting ping_interval and ping_timweout
        async with websockets.connect(uri, ping_interval=20, ping_timeout=12000) as ws:

            from PIL import Image

            source_image_path = "roll_outs/ur5s4_b0_source.png"

            # First image generation call 
            # Convert the image to a numpy array
            for i in range(len(ur5_cleaning_text)):
                source_image = np.array(Image.open(source_image_path).resize((224, 224)))
            
                inputs = {
                    "observation/base_0_camera/rgb/image": source_image,
                    "raw_text": ur5_cleaning_text[i],
                }
                packer = Packer()
                data = packer.pack(inputs)

                import time
                start_time = time.time()
                await ws.send(data)
                response = await ws.recv()
                end_time = time.time()
                print(f"Time taken for sending and receiving: {end_time - start_time} seconds")
                output = unpackb(response)
                target_image_path = f"roll_outs/ur5s4_b0_edited_{i}.png"
                output_image = Image.fromarray(output['future/observation/base_0_camera/rgb/image'])
                output_image.save(target_image_path)
                print(f"Image saved as {target_image_path}")
                source_image_path = target_image_path
        
            # # Second image generation call 
            # inputs = {
            #     "observation/base_0_camera/rgb/image": output['future/observation/base_0_camera/rgb/image'],
            #     "raw_text": "make her dress green",
            # }
            # packer = Packer()
            # data = packer.pack(inputs)

            # await ws.send(data)
            # response = await ws.recv()
            # output = unpackb(response)
            # output_image = Image.fromarray(output['future/observation/base_0_camera/rgb/image'])
            # output_image.save("output2.png")
            # print("Image saved as output2.png")

    except websockets.ConnectionClosedOK:
        print("Server closed the connection – goodbye!")
    except websockets.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")

if __name__ == "__main__":
    asyncio.run(connect_and_send())
