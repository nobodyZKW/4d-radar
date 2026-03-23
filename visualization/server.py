from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from api.compare import compare_api
from api.experiments import experiment_detail_api, experiment_metrics_api, list_experiments_api
from api.metrics import ablations_api, decode_tuning_api
from api.samples import list_samples_api, sample_detail_api, sample_predictions_api
from data_adapters.sample_predictions import SamplePredictionProvider
from services.compare_service import CompareService
from services.experiment_service import ExperimentService
from services.registry_service import RegistryService
from services.sample_viewer_service import SampleViewerService


VIS_ROOT = Path(__file__).resolve().parent
WEB_ROOT = VIS_ROOT / "web"


class DashboardHandler(SimpleHTTPRequestHandler):
    registry: RegistryService | None = None
    experiment_service: ExperimentService | None = None
    compare_service: CompareService | None = None
    sample_service: SampleViewerService | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def _send_json(self, payload: Dict[str, Any], code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_name: str) -> None:
        file_path = WEB_ROOT / html_name
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "page not found")
            return
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_api(self, path: str, query: Dict[str, List[str]]) -> bool:
        registry = self.registry
        experiment_service = self.experiment_service
        compare_service_obj = self.compare_service
        sample_service = self.sample_service
        if registry is None or experiment_service is None or compare_service_obj is None or sample_service is None:
            self._send_json({"ok": False, "message": "server not initialized"}, code=500)
            return True

        if path == "/api/health":
            self._send_json({"ok": True, "message": "", "app": registry.get_app_info()})
            return True

        if path == "/api/app-info":
            self._send_json({"ok": True, "message": "", "app": registry.get_app_info()})
            return True

        if path == "/api/family-overview":
            self._send_json(experiment_service.get_family_overview())
            return True

        if path == "/api/experiments":
            self._send_json(list_experiments_api(experiment_service, query))
            return True

        if path.startswith("/api/experiments/"):
            parts = path.split("/")
            if len(parts) >= 4 and parts[3]:
                exp_id = parts[3]
                if len(parts) == 5 and parts[4] == "metrics":
                    data = experiment_metrics_api(experiment_service, exp_id)
                else:
                    data = experiment_detail_api(experiment_service, exp_id)
                code = 200 if data.get("ok", False) else 404
                self._send_json(data, code=code)
                return True

        if path == "/api/compare":
            self._send_json(compare_api(compare_service_obj, query))
            return True

        if path == "/api/samples":
            self._send_json(list_samples_api(sample_service, query))
            return True

        if path.startswith("/api/samples/"):
            parts = path.split("/")
            if len(parts) >= 4 and parts[3]:
                sample_id = parts[3]
                try:
                    if len(parts) == 5 and parts[4] == "predictions":
                        data = sample_predictions_api(sample_service, sample_id, query)
                    else:
                        data = sample_detail_api(sample_service, sample_id, query)
                    self._send_json(data, code=200)
                except Exception as exc:
                    self._send_json({"ok": False, "message": str(exc)}, code=400)
                return True

        if path == "/api/ablations":
            self._send_json(ablations_api(experiment_service, query))
            return True

        if path == "/api/decode-tuning":
            self._send_json(decode_tuning_api(experiment_service, query))
            return True

        if path.startswith("/data/image/"):
            sample_name = path.split("/data/image/", 1)[1]
            if not sample_name.endswith(".jpg"):
                self.send_error(HTTPStatus.BAD_REQUEST, "Only .jpg is supported")
                return True
            image_path = sample_service.provider.image_dir / sample_name
            if not image_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                return True
            data = image_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return True

        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path.startswith("/api/") or path.startswith("/data/image/"):
            handled = self._handle_api(path, query)
            if not handled:
                self._send_json({"ok": False, "message": "api not found"}, code=404)
            return

        route_map = {
            "/": "index.html",
            "/experiments": "experiments.html",
            "/compare": "compare.html",
            "/sample-viewer": "sample_viewer.html",
            "/v1_5-results": "v1_5_results.html",
            "/official-eval": "official_eval.html",
            "/ablations": "ablations.html",
            # compatibility routes
            "/baseline-results": "experiment_detail.html",
            "/v1-results": "experiment_detail.html",
            "/split-player": "sample_viewer.html",
        }
        if path in route_map:
            self._send_html(route_map[path])
            return

        if path.startswith("/experiments/"):
            self._send_html("experiment_detail.html")
            return

        # static file fallback
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser("4D Radar Unified Visualization")
    parser.add_argument("--host", type=str, default="")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    if not WEB_ROOT.exists():
        raise FileNotFoundError(f"Web folder not found: {WEB_ROOT}")

    registry = RegistryService(VIS_ROOT)
    exp_service = ExperimentService(registry)
    cmp_service = CompareService(exp_service)
    sample_provider = SamplePredictionProvider(
        data_root=registry.data_root,
        artifact_root=registry.artifact_root,
        experiments=registry.get_experiment_map(),
    )
    sample_service = SampleViewerService(registry, exp_service, sample_provider)

    DashboardHandler.registry = registry
    DashboardHandler.experiment_service = exp_service
    DashboardHandler.compare_service = cmp_service
    DashboardHandler.sample_service = sample_service

    app_info = registry.get_app_info()
    host = args.host or registry.app_config.get("app", {}).get("host", "127.0.0.1")
    port = args.port or int(registry.app_config.get("app", {}).get("port", 8090))

    print(f"[visualization] project root: {app_info['project_root']}")
    print(f"[visualization] data root: {app_info['data_root']}")
    print(f"[visualization] artifact root: {app_info['artifact_root']}")
    print(f"[visualization] open: http://{host}:{port}")

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
