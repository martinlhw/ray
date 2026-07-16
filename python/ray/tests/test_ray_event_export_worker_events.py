import json
import logging
import textwrap

import grpc
import pytest

import ray.dashboard.consts as dashboard_consts
from ray._common.network_utils import find_free_port
from ray._common.test_utils import wait_for_condition
from ray._private import ray_constants
from ray._private.test_utils import run_string_as_driver_nonblocking
from ray._raylet import GcsClient

logger = logging.getLogger(__name__)


_EVENT_AGGREGATOR_AGENT_TARGET_PORT = find_free_port()
_EVENT_AGGREGATOR_AGENT_TARGET_IP = "127.0.0.1"
_EVENT_AGGREGATOR_AGENT_TARGET_ADDR = (
    f"http://{_EVENT_AGGREGATOR_AGENT_TARGET_IP}:{_EVENT_AGGREGATOR_AGENT_TARGET_PORT}"
)


@pytest.fixture(scope="module")
def httpserver_listen_address():
    return (_EVENT_AGGREGATOR_AGENT_TARGET_IP, _EVENT_AGGREGATOR_AGENT_TARGET_PORT)


# Worker lifecycle events are produced by GCS, gated on RAY_enable_ray_event.
_cluster_with_aggregator_target = pytest.mark.parametrize(
    ("preserve_proto_field_name", "ray_start_cluster_head_with_env_vars"),
    [
        pytest.param(
            preserve_proto_field_name,
            {
                "env_vars": {
                    "RAY_enable_ray_event": "1",
                    "RAY_ray_events_report_interval_ms": 100,
                    "RAY_DASHBOARD_AGGREGATOR_AGENT_EVENTS_EXPORT_ADDR": _EVENT_AGGREGATOR_AGENT_TARGET_ADDR,
                    "RAY_DASHBOARD_AGGREGATOR_AGENT_PRESERVE_PROTO_FIELD_NAME": (
                        "1" if preserve_proto_field_name is True else "0"
                    ),
                },
            },
        )
        for preserve_proto_field_name in [True, False]
    ],
    indirect=["ray_start_cluster_head_with_env_vars"],
)


def wait_until_grpc_channel_ready(
    gcs_address: str, node_ids: list[str], timeout: int = 5
):
    gcs_client = GcsClient(address=gcs_address)

    def get_dashboard_agent_address(node_id: str):
        return gcs_client.internal_kv_get(
            f"{dashboard_consts.DASHBOARD_AGENT_ADDR_NODE_ID_PREFIX}{node_id}".encode(),
            namespace=ray_constants.KV_NAMESPACE_DASHBOARD,
            timeout=dashboard_consts.GCS_RPC_TIMEOUT_SECONDS,
        )

    wait_for_condition(
        lambda: all(
            get_dashboard_agent_address(node_id) is not None for node_id in node_ids
        )
    )
    grpc_ports = [
        json.loads(get_dashboard_agent_address(node_id))[2] for node_id in node_ids
    ]
    targets = [f"127.0.0.1:{grpc_port}" for grpc_port in grpc_ports]
    for target in targets:
        channel = grpc.insecure_channel(target)
        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
        except grpc.FutureTimeoutError:
            return False
    return True


def get_and_validate_events(httpserver, validation_func):
    event_data = []
    for http_log in httpserver.log:
        req, _ = http_log
        data = json.loads(req.data)
        event_data.extend(data)

    try:
        validation_func(event_data)
        return True
    except Exception:
        return False


def run_driver_script_and_wait_for_events(script, httpserver, cluster, validation_func):
    httpserver.expect_request("/", method="POST").respond_with_data("", status=200)
    node_ids = [node.node_id for node in cluster.list_all_nodes()]
    assert wait_until_grpc_channel_ready(cluster.gcs_address, node_ids)
    run_string_as_driver_nonblocking(script)
    wait_for_condition(lambda: get_and_validate_events(httpserver, validation_func))


def _keys(preserve_proto_field_name: bool) -> dict:
    """JSON key names for a WorkerLifecycleEvent under either proto->JSON naming mode."""
    if preserve_proto_field_name:
        return dict(
            event_type="event_type",
            wle="worker_lifecycle_event",
            worker_id="worker_id",
            worker_type="worker_type",
            state_transitions="state_transitions",
            death_info="death_info",
            exit_type="exit_type",
        )
    return dict(
        event_type="eventType",
        wle="workerLifecycleEvent",
        worker_id="workerId",
        worker_type="workerType",
        state_transitions="stateTransitions",
        death_info="deathInfo",
        exit_type="exitType",
    )


class TestWorkerLifecycleEvents:
    @_cluster_with_aggregator_target
    def test_worker_lifecycle_exported_and_driver_excluded(
        self,
        ray_start_cluster_head_with_env_vars,
        httpserver,
        preserve_proto_field_name,
    ):
        # A normal task worker (ALIVE, then DEAD when the job ends) plus an actor
        # killed via ray.kill (DEAD with INTENDED_SYSTEM_EXIT). Drivers must NOT
        # produce a worker lifecycle event -- they are covered by driver job events.
        script = textwrap.dedent(
            """
            import ray
            ray.init()

            @ray.remote
            def t():
                return 1
            ray.get(t.remote())

            @ray.remote
            class A:
                def ping(self):
                    return "ok"
            a = A.remote()
            ray.get(a.ping.remote())
            ray.kill(a)
            """
        )

        k = _keys(preserve_proto_field_name)

        def validate_events(events):
            worker_events = [
                e for e in events if e.get(k["event_type"]) == "WORKER_LIFECYCLE_EVENT"
            ]
            assert worker_events, "no WORKER_LIFECYCLE_EVENT received"

            saw_alive = False
            saw_killed_dead = False
            for e in worker_events:
                w = e[k["wle"]]
                # Filter check: no worker lifecycle event should describe a driver.
                assert w.get(k["worker_type"]) != "DRIVER"
                # Identity is always present.
                assert w.get(k["worker_id"])
                for st in w.get(k["state_transitions"], []):
                    if st.get("state") == "ALIVE":
                        saw_alive = True
                    if st.get("state") == "DEAD":
                        death_info = st.get(k["death_info"], {})
                        if death_info.get(k["exit_type"]) == "INTENDED_SYSTEM_EXIT":
                            saw_killed_dead = True

            assert saw_alive, "no ALIVE worker transition observed"
            assert (
                saw_killed_dead
            ), "no DEAD worker transition with INTENDED_SYSTEM_EXIT (ray.kill)"

        run_driver_script_and_wait_for_events(
            script,
            httpserver,
            ray_start_cluster_head_with_env_vars,
            validate_events,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
