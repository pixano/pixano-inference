# Troubleshooting

## The server hangs on startup

Startup is bounded and interruptible: `Ctrl-C` exits within a few seconds at any point, and every
wait on Ray fails with a diagnostic rather than blocking forever. Two knobs on `RayServeConfig`
control the budgets:

| Setting | Default | Bounds |
| --- | --- | --- |
| `serve_start_timeout_s` | `120.0` | Waiting for Ray Serve's controller actor to come up. |
| `deploy_timeout_s` | `600.0` | Waiting for one model's Serve application to reach `RUNNING`. |

These exist because Ray's own calls have no timeout: `serve.start()` does a bare `ray.get` on the
controller, and `serve.run()` waits with `timeout_s=-1` regardless of its `blocking` argument. If a
Ray node cannot schedule the controller, those calls never return on their own.

If startup aborts with **"the Ray node died (the raylet exited)"**, the message names the
`raylet.err` to read. The most common causes follow.

## `on_read bad version` — Ray assigned the same port twice

```
(raylet) Runtime Env Agent timed out in 30000ms. Status: Disconnected: on_read bad version,
         bytes_transferred 0, address: <ip>, port: <port>
         reason_message: "Raylet could not connect to Runtime Env Agent"
```

Ray picks free ports by binding a socket, reading the port number, then closing it — so a port can
be handed out twice. When the raylet's object-manager gRPC server and the runtime-env agent land on
the same port, the raylet's HTTP client reaches the gRPC (HTTP/2) server, fails to parse the reply,
and kills the node 30 s in. Compare the ports in `raylet.out` (`ObjectManager server started,
listening on port N`) with `runtime_env_agent.log` (`Listening to address ..., port N`) to confirm.

`ray.init()` does not expose `object_manager_port`, so the fix is to start the cluster yourself with
explicit ports and point the server at it:

```bash
ray start --head --node-ip-address=127.0.0.1 \
    --object-manager-port=8076 --runtime-env-agent-port=8077
```

```python
config = RayServeConfig(ray_address="auto")
```

## Ray reports a TPU (or no GPU) on a GPU machine

Ray's accelerator autodetection probes for TPUs with `glob.glob("/dev/accel*")`. On modern Linux
kernels `/dev/accel` exists as a plain DRM-subsystem directory with no TPU present, so Ray registers
`TPU: 1.0` and never registers the NVIDIA GPU. Which accelerator wins varies from run to run,
because Ray iterates its managers out of a `set`.

The symptom is a deployment that fails its preflight check with
`Insufficient GPUs to deploy '<model>': need 1, available 0` even though `nvidia-smi` and
`torch.cuda.is_available()` both see the GPU. Check with:

```python
import ray; ray.init(); print(ray.cluster_resources())
```

Set the count explicitly so nothing is autodetected:

```python
config = RayServeConfig(num_gpus=1)
```

## Running under `uv run`

Ray >= 2.53 detects a `uv run` parent process and relaunches every worker through `uv run` inside an
uploaded copy of the working directory. That copy excludes `pyproject.toml`, `uv.lock` and `.venv`
(see `pixano_inference.ray.utils._DEFAULT_EXCLUDES`), leaving `uv` with no project to resolve, so
the server disables Ray's hook before `ray.init()` and workers inherit the active interpreter
instead. To opt back into Ray's behaviour, set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=1` **and** remove the
project files from the excludes.
