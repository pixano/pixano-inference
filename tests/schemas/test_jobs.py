# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""The job status envelope: a completed tracking job carries a typed ``TrackingOutput``."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pixano_inference.schemas import JobStatus, TrackedFrame, TrackedObject, TrackingOutput


def test_a_completed_job_parses_its_payload_as_tracking_output():
    # What the server sends: TrackingOutput.model_dump(by_alias=True), wrapped by serialize_job.
    wire = {
        "jobId": "job-1",
        "status": "completed",
        "detail": None,
        "data": {"frames": [{"frameIndex": 2, "objects": [{"trackId": 7, "box": [1, 2, 3, 4], "score": 0.5}]}]},
        "metadata": {"capability": "tracking"},
        "timestamp": "2026-10-08T09:00:00Z",
        "processingTime": 1.5,
    }

    job = JobStatus.model_validate(wire)

    assert job.data == TrackingOutput(
        frames=[TrackedFrame(frame_index=2, objects=[TrackedObject(track_id=7, box=[1, 2, 3, 4], score=0.5)])]
    )
    assert job.data.tracks()[7][0][0] == 2
    assert job.timestamp == datetime(2026, 10, 8, 9, 0, tzinfo=timezone.utc)
    # Dumped back by alias, it is the same frame the sync route would serialize.
    assert job.model_dump(mode="json", by_alias=True)["data"] == {
        "frames": [
            {
                "frameIndex": 2,
                "objects": [{"trackId": 7, "box": [1.0, 2.0, 3.0, 4.0], "score": 0.5, "class": None, "mask": None}],
            }
        ]
    }


def test_a_running_job_has_no_payload_and_a_0_7_0_server_sends_no_timestamp():
    job = JobStatus.model_validate({"jobId": "job-1", "status": "running"})

    assert job.data is None and job.timestamp is None and job.processing_time == 0.0


def test_snake_case_still_builds_a_status():
    job = JobStatus(job_id="j", status="completed", data=TrackingOutput(frames=[]), processing_time=2.0)

    assert job.model_dump(by_alias=True)["jobId"] == "j"


def test_a_payload_that_is_not_a_tracking_output_is_rejected():
    with pytest.raises(ValidationError, match="frames"):
        JobStatus.model_validate({"jobId": "job-1", "status": "completed", "data": {"masks": []}})
