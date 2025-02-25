# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================


import re

import pytest
import responses
from fastapi import HTTPException
from pytest_httpx import HTTPXMock

from pixano_inference.client import PixanoInferenceClient
from pixano_inference.pydantic import ModelInfo
from pixano_inference.settings import Settings


URL = "http://localhost:8081"


class TestPixanoInferenceClient:
    def test_init_client(self):
        client = PixanoInferenceClient(url=URL)
        assert client.url == URL

        client = PixanoInferenceClient(url=f"{URL}/")
        assert client.url == URL

        with pytest.raises(ValueError, match="Invalid URL, got 'wrongurl'."):
            client = PixanoInferenceClient(url="wrongurl")

    @responses.activate
    def test_connect(self):
        settings = Settings(num_cpus=1, num_gpus=0)
        response = responses.Response(method="GET", url=f"{URL}/app/settings/", json=settings.model_dump(), status=200)
        responses.add(response)
        expected_output = settings.model_dump()
        expected_output.update({"url": URL})

        client = PixanoInferenceClient.connect(URL)
        assert client.model_dump() == expected_output

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
    async def test_valid_rest_call(
        self, httpx_mock: HTTPXMock, simple_pixano_inference_client: PixanoInferenceClient, method: str
    ):
        data = {"image_url": "https://example.com/image.jpg"}
        httpx_mock.add_response(json=data)
        output_response = await simple_pixano_inference_client._rest_call("data/", method)
        assert output_response.status_code == 200
        assert output_response.json() == data

    @pytest.mark.asyncio
    async def test_invalid_rest_call(
        self, httpx_mock: HTTPXMock, simple_pixano_inference_client: PixanoInferenceClient
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                r"Invalid REST call method. Expected one of ['GET', 'POST', 'PUT', 'DELETE'], "
                r"but got 'WRONG_METHOD'."
            ),
        ):
            await simple_pixano_inference_client._rest_call("data/", "WRONG_METHOD")

        httpx_mock.add_response(json={"error": "Invalid data."}, status_code=400)

        with pytest.raises(HTTPException) as exc_info:
            await simple_pixano_inference_client._rest_call("data/", "POST")
            assert exc_info.value.status_code == 400
            assert exc_info.value.response.json() == {
                "error": "Invalid data.",
            }

    @pytest.mark.asyncio
    async def test_get(self, httpx_mock: HTTPXMock, simple_pixano_inference_client: PixanoInferenceClient):
        data = {"image_url": "https://example.com/image.jpg"}
        httpx_mock.add_response(json=data)
        assert (await simple_pixano_inference_client.get("data/")).json() == data

    @pytest.mark.asyncio
    async def test_post(self, httpx_mock: HTTPXMock, simple_pixano_inference_client: PixanoInferenceClient):
        data = {"image_url": "https://example.com/image.jpg"}
        httpx_mock.add_response(json=data)
        assert (await simple_pixano_inference_client.post("data/", json=data)).json() == data

    @pytest.mark.asyncio
    async def test_put(self, httpx_mock: HTTPXMock, simple_pixano_inference_client: PixanoInferenceClient):
        data = {"image_url": "https://example.com/image.jpg"}
        httpx_mock.add_response(json=data)
        assert (await simple_pixano_inference_client.put("data/", json=data)).json() == data

    @pytest.mark.asyncio
    async def test_delete(self, httpx_mock: HTTPXMock, simple_pixano_inference_client: PixanoInferenceClient):
        httpx_mock.add_response()
        assert (await simple_pixano_inference_client.delete("data/")).status_code == 200

    @pytest.mark.asyncio
    async def test_list_models(self, httpx_mock: HTTPXMock, simple_pixano_inference_client: PixanoInferenceClient):
        models_info = [
            ModelInfo(**{"name": "sam", "provider": "transformers", "task": "image_mask_generation"}),
            ModelInfo(**{"name": "sam2", "provider": "sam2", "task": "video_mask_generation"}),
        ]
        httpx_mock.add_response(json=[m.model_dump() for m in models_info])

        assert (await simple_pixano_inference_client.list_models()) == models_info

    @pytest.mark.asyncio
    async def test_delete_model(self, httpx_mock: HTTPXMock, simple_pixano_inference_client: PixanoInferenceClient):
        httpx_mock.add_response()
        assert await simple_pixano_inference_client.delete_model("model_name") is None
