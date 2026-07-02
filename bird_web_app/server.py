from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import socket
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
STATIC_ROOT = APP_ROOT / "static"
CLASSES_PATH = PROJECT_ROOT / "datasets" / "CUB_200_2011" / "CUB_200_2011" / "classes.txt"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ckpt"
    / "pipeline"
    / "convnext_region_448_final"
    / "final"
    / ("se" + "ed_2024")
    / "convnextv2_tiny_dca_region_final.pt"
)

MODEL_NAME = "convnextv2_tiny_dca_region"
MODEL_TEST_TOP1 = 90.52468070417673
IMAGE_SIZE = 448
RESIZE_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"

sys.path.insert(0, str(PROJECT_ROOT))
from model import build_convnextv2_tiny_dca_region  # noqa: E402


class BirdRecognizer:
    _title_cache: dict[str, str | None] = {}

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self._load_classes()
        self.model = self._load_model()

    def _load_classes(self) -> list[str]:
        if not CLASSES_PATH.is_file():
            raise FileNotFoundError(f"缺少类别文件：{CLASSES_PATH}")

        classes: list[str] = []
        with CLASSES_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                _, raw_name = line.split(maxsplit=1)
                classes.append(raw_name)

        if len(classes) != 200:
            raise ValueError(f"应有 200 个类别，实际读取到 {len(classes)} 个")
        return classes

    def _load_model(self) -> torch.nn.Module:
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"缺少模型权重文件：{CHECKPOINT_PATH}")

        model = build_convnextv2_tiny_dca_region(
            num_classes=len(self.classes),
            pretrained=False,
            image_size=IMAGE_SIZE,
            fpn_channels=256,
        )
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "模型权重与结构不匹配。"
                f"缺失参数：{missing[:5]}；多余参数：{unexpected[:5]}"
            )
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _display_name(raw_name: str) -> str:
        name = raw_name.split(".", maxsplit=1)[-1]
        return name.replace("_", " ")

    @staticmethod
    def _english_wikipedia_title(species_name: str) -> str | None:
        cache_key = f"en:{species_name}"
        if cache_key in BirdRecognizer._title_cache:
            return BirdRecognizer._title_cache[cache_key]

        for title in BirdRecognizer._candidate_wikipedia_titles(species_name):
            if BirdRecognizer._summary_payload(title) is not None:
                BirdRecognizer._title_cache[cache_key] = title
                return title

        # 先按鸟类语境查，再按原英文名查，减少误入非鸟类页面的概率。
        for search_text in (f"{species_name} bird", species_name):
            query = urllib.parse.urlencode(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": search_text,
                    "format": "json",
                    "srlimit": "1",
                }
            )
            request = urllib.request.Request(
                f"{WIKIPEDIA_SEARCH_URL}?{query}",
                headers={"User-Agent": "bird-species-recognition-web/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=4) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                continue

            results = payload.get("query", {}).get("search", [])
            if not results:
                continue
            title = results[0].get("title")
            if isinstance(title, str) and title.strip() and not title.lower().startswith("list of "):
                BirdRecognizer._title_cache[cache_key] = title
                return title

        BirdRecognizer._title_cache[cache_key] = None
        return None

    @staticmethod
    def _candidate_wikipedia_titles(species_name: str) -> list[str]:
        words = species_name.split()
        title_case = species_name.title()
        sentence_case = species_name[:1].upper() + species_name[1:].lower()
        candidates = [title_case, sentence_case, species_name]
        if len(words) >= 3:
            hyphenated = f"{words[0]}-{words[1]} " + " ".join(words[2:])
            candidates.extend([hyphenated.title(), hyphenated[:1].upper() + hyphenated[1:].lower()])
        if len(words) == 2:
            candidates.append(f"{words[0]} {words[1].lower()}")

        seen: set[str] = set()
        unique_candidates: list[str] = []
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _summary_payload(page_title: str) -> dict[str, Any] | None:
        title = urllib.parse.quote(page_title.replace(" ", "_"))
        request = urllib.request.Request(
            WIKIPEDIA_SUMMARY_URL.format(title=title),
            headers={"User-Agent": "bird-species-recognition-web/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=4) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _online_description(species_name: str) -> dict[str, str | None]:
        page_title = BirdRecognizer._english_wikipedia_title(species_name)
        if not page_title:
            return {"text": None, "source": None, "url": None, "title": None}

        payload = BirdRecognizer._summary_payload(page_title)
        if payload is None:
            return {"text": None, "source": None, "url": None, "title": page_title}

        extract = payload.get("extract")
        page_url = payload.get("content_urls", {}).get("desktop", {}).get("page")
        display_title = payload.get("title")
        if not isinstance(extract, str) or not extract.strip():
            return {"text": None, "source": None, "url": None, "title": page_title}

        return {
            "text": extract.strip(),
            "source": "English Wikipedia",
            "url": page_url if isinstance(page_url, str) else None,
            "title": display_title if isinstance(display_title, str) else page_title,
        }

    @staticmethod
    def _preprocess(image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = TF.resize(image, RESIZE_SIZE, antialias=True)
        image = TF.center_crop(image, [IMAGE_SIZE, IMAGE_SIZE])
        tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return tensor.unsqueeze(0)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        # 输入图片统一按训练时的 448 尺寸预处理，保证网站推理和实验设置一致。
        tensor = self._preprocess(image).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_probabilities, top_indices = torch.topk(probabilities, k=5)

        # 类别只来自本地模型；结果直接使用 CUB 类别的英文鸟名。
        top5: list[dict[str, Any]] = []
        for probability, index in zip(top_probabilities.tolist(), top_indices.tolist()):
            raw_name = self.classes[index]
            english_name = self._display_name(raw_name)
            top5.append(
                {
                    "rank": len(top5) + 1,
                    "class_id": index + 1,
                    "raw_name": raw_name,
                    "name": english_name,
                    "display_name": english_name,
                    "confidence": probability,
                    "confidence_percent": round(probability * 100, 2),
                }
            )

        winner = top5[0]
        online_description = self._online_description(winner["name"])
        model_description = (
            f"The model identifies this bird as {winner['display_name']} with "
            f"{winner['confidence_percent']}% confidence. This fallback description is based only "
            f"on model output. Among the 200 CUB fine-grained bird classes, the next strongest "
            f"candidates are {top5[1]['display_name']} ({top5[1]['confidence_percent']}%) and "
            f"{top5[2]['display_name']} ({top5[2]['confidence_percent']}%)."
        )

        return {
            "model": MODEL_NAME,
            "test_top1_percent": round(MODEL_TEST_TOP1, 4),
            "image_size": IMAGE_SIZE,
            "prediction": winner,
            "top5": top5,
            "description": online_description["text"] or model_description,
            "model_description": model_description,
            "description_source": online_description["source"] or "Model output",
            "description_url": online_description["url"],
        }


recognizer: BirdRecognizer | None = None


def get_recognizer() -> BirdRecognizer:
    global recognizer
    if recognizer is None:
        recognizer = BirdRecognizer()
    return recognizer


class BirdRequestHandler(BaseHTTPRequestHandler):
    server_version = "BirdSpeciesWeb/1.0"

    def do_HEAD(self) -> None:
        if self.path in {"/", "/index.html"}:
            path = STATIC_ROOT / "index.html"
            if not path.is_file():
                self._send_error(HTTPStatus.NOT_FOUND, "页面不存在")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            return
        self._send_error(HTTPStatus.NOT_FOUND, "页面不存在")

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if self.path.startswith("/static/"):
            relative = self.path.removeprefix("/static/").split("?", maxsplit=1)[0]
            target = (STATIC_ROOT / relative).resolve()
            if not str(target).startswith(str(STATIC_ROOT.resolve())):
                self._send_error(HTTPStatus.FORBIDDEN, "没有访问权限")
                return
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self._send_file(target, content_type)
            return
        if self.path == "/api/health":
            payload = {
                "ok": True,
                "model": MODEL_NAME,
                "device": str(get_recognizer().device),
            }
            self._send_json(payload)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "接口不存在")

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self._send_error(HTTPStatus.NOT_FOUND, "接口不存在")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_error(HTTPStatus.BAD_REQUEST, "没有上传图片")
            return
        if content_length > MAX_UPLOAD_BYTES:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "图片过大，请上传 15MB 以内的图片")
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": str(content_length),
                },
            )
            image_field = form["image"] if "image" in form else None
            if image_field is None or not getattr(image_field, "file", None):
                self._send_error(HTTPStatus.BAD_REQUEST, "没有上传图片")
                return

            image = Image.open(image_field.file)
            result = get_recognizer().predict(image)
            self._send_json(result)
        except Exception as exc:
            traceback.print_exc()
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "文件不存在")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def get_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="鸟类识别网站服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", default=8000, type=int, help="监听端口")
    parser.add_argument("--warmup", action="store_true", help="启动时预先加载模型")
    args = parser.parse_args()

    if args.warmup:
        get_recognizer()

    server = ThreadingHTTPServer((args.host, args.port), BirdRequestHandler)
    lan_ip = get_lan_ip()
    print(f"本机地址：http://127.0.0.1:{args.port}")
    print(f"局域网地址：http://{lan_ip}:{args.port}")
    print("按 Ctrl+C 停止服务。")
    server.serve_forever()


if __name__ == "__main__":
    main()
