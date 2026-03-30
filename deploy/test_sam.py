import asyncio
from pixano_inference.client import PixanoInferenceClient
from pixano_inference.schemas import SegmentationRequest


async def main():
    client = PixanoInferenceClient.connect("http://127.0.0.1:7463")
    print(f"Connected! GPUs: {client.num_gpus}")

    models = await client.list_models()
    for m in models:
        print(f"  - {m.name} (capability={m.capability})")

    # Send segmentation request with a point prompt on the bus image
    request = SegmentationRequest(
        model="sam2-image",
        image="https://ultralytics.com/images/bus.jpg",
        points=[[[400, 500]]],  # point on the bus
        labels=[[1]],  # foreground
    )
    result = await client.segmentation(request)
    print(f"Status: {result.status}")
    print(f"Processing time: {result.processing_time:.3f}s")
    print(f"Masks: {len(result.data.masks)}")
    print(f"Scores: {result.data.scores.to_numpy().tolist()}")
    for i, mask_list in enumerate(result.data.masks):
        for j, mask_rle in enumerate(mask_list):
            m = mask_rle.to_mask()
            coverage = m.sum() / m.size * 100
            print(f"  mask[{i}][{j}] shape={m.shape}, coverage={coverage:.1f}%")
    print("SAM2 e2e test PASSED")


asyncio.run(main())
