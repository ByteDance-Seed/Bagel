import asyncio
import websockets
import numpy as np
from msgpack_numpy import Packer, unpackb

async def connect_and_send():
    uri = "ws://localhost:8765"
    try:
        # Increase the timeout time by setting ping_interval and ping_timeout
        async with websockets.connect(uri, ping_interval=20, ping_timeout=1200) as ws:

            source_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
            inputs = {
                "observation/base_0_camera/rgb/image": source_image,
                "raw_text": "pick up the carrot",
            }
            packer = Packer()
            data = packer.pack(inputs)

            await ws.send(data)
            response = await ws.recv()
            output = unpackb(response)
            print(output)

    except websockets.ConnectionClosedOK:
        print("Server closed the connection – goodbye!")
    except websockets.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")

if __name__ == "__main__":
    asyncio.run(connect_and_send())
