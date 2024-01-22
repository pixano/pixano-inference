# @Copyright: CEA-LIST/DIASI/SIALV/LVA (2023)
# @Author: CEA-LIST/DIASI/SIALV/LVA <pixano@cea.fr>
# @License: CECILL-C
#
# This software is a collaborative computer program whose purpose is to
# generate and explore labeled data for computer vision applications.
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
#
# http://www.cecill.info

import pyarrow as pa
import shortuuid
import tensorflow as tf
import tensorflow_hub as hub
from pixano.core import BBox, Image
from pixano.models import InferenceModel
from pixano.utils import coco_names_91


class FasterRCNN(InferenceModel):
    """TensorFlow Hub FasterRCNN Model

    Attributes:
        name (str): Model name
        model_id (str): Model ID
        device (str): Model GPU or CPU device
        description (str): Model description
        model (tf.keras.Model): TensorFlow model
    """

    def __init__(
        self,
        model_id: str = "",
        device: str = "/GPU:0",
    ) -> None:
        """Initialize model

        Args:
            model_id (str, optional): Previously used ID, generate new ID if "". Defaults to "".
            device (str, optional): Model GPU or CPU device (e.g. "/GPU:0", "/CPU:0"). Defaults to "/GPU:0".
        """

        super().__init__(
            name="FasterRCNN_R50",
            model_id=model_id,
            device=device,
            description="From TensorFlow Hub. FasterRCNN model, with ResNet50 architecture.",
        )

        # Model
        with tf.device(self.device):
            self.model = hub.load(
                "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1"
            )

    def preannotate(
        self,
        batch: pa.RecordBatch,
        views: list[str],
        uri_prefix: str,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Inference pre-annotation for a batch

        Args:
            batch (pa.RecordBatch): Input batch
            views (list[str]): Dataset views
            uri_prefix (str): URI prefix for media files
            threshold (float, optional): Confidence threshold. Defaults to 0.0.

        Returns:
            list[dict]: Processed rows
        """

        rows = []

        for view in views:
            # TF.Hub Models don't support image batches, so iterate manually
            for x in range(batch.num_rows):
                # Preprocess image
                im = Image.from_dict(batch[view][x].as_py())
                im.uri_prefix = uri_prefix
                im = im.as_pillow()
                im_tensor = tf.expand_dims(tf.keras.utils.img_to_array(im), 0)
                im_tensor = tf.image.convert_image_dtype(im_tensor, dtype="uint8")

                # Inference
                output = self.model(im_tensor)

                # Process model outputs
                rows.extend(
                    [
                        {
                            "id": shortuuid.uuid(),
                            "item_id": batch["id"][x].as_py(),
                            "view_id": view,
                            "bbox": BBox.from_xyxy(
                                [
                                    float(output["detection_boxes"][0][i][1]),
                                    float(output["detection_boxes"][0][i][0]),
                                    float(output["detection_boxes"][0][i][3]),
                                    float(output["detection_boxes"][0][i][2]),
                                ],
                                confidence=float(output["detection_scores"][0][i]),
                            ).to_dict(),
                            "category": coco_names_91(
                                output["detection_classes"][0][i]
                            ),
                        }
                        for i in range(int(output["num_detections"]))
                        if output["detection_scores"][0][i] > threshold
                    ]
                )

        return rows
