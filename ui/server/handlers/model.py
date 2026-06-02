"""模型选股：模型列表、预测排名、单股时序、回测报告、实时选股。"""
import logging

from .. import app

logger = logging.getLogger(__name__)


def _service():
    svc = getattr(app, "model_service", None)
    if svc is None:
        raise RuntimeError("ModelService not initialized")
    return svc


def list_models(rh):
    """GET /api/models"""
    rh._json_response({"models": _service().list_models()})


def run_prediction(rh, rest_path):
    """POST /api/models/{alias}/run

    Body: {"date": "YYYY-MM-DD", "top_k": 30}
    实时运行模型预测，返回目标日期的选股排名。
    """
    # rest_path 格式: "alias/run" 或 "alias"
    parts = rest_path.strip("/").split("/", 1)
    alias = parts[0]
    action = parts[1] if len(parts) > 1 else ""

    if action != "run":
        return rh._json_response({"error": f"Unknown POST action: {action}"}, status=404)

    body = rh._read_body()
    date = body.get("date")
    top_k = int(body.get("top_k", 30))

    if not date:
        return rh._json_response({"error": "Missing required field: date"}, status=400)

    try:
        result = _service().run_prediction(alias, date=date, top_k=top_k)
        rh._json_response(result)
    except KeyError as exc:
        rh._json_response({"error": str(exc)}, status=404)
    except (ValueError, FileNotFoundError) as exc:
        rh._json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("run_prediction error for %s", alias)
        rh._json_response({"error": str(exc)}, status=500)


def model_detail(rh, rest_path):
    """GET /api/models/{rest_path}

    rest_path 格式：
      - "{alias}/info"          → 模型元信息 + 指标
      - "{alias}/predictions"   → 某日选股排名 (?date=YYYY-MM-DD&top_k=30)
      - "{alias}/dates"         → 可用预测日期列表
      - "{alias}/stock/{inst}"  → 单股预测时序
      - "{alias}/report"        → 回测报告
    """
    parts = rest_path.strip("/").split("/", 1)
    if len(parts) < 2:
        return rh._json_response({"error": "Invalid model path. Expected /api/models/{alias}/{action}"}, status=400)

    alias = parts[0]
    action_and_rest = parts[1]
    action_parts = action_and_rest.split("/", 1)
    action = action_parts[0]

    try:
        if action == "info":
            info = _service().get_model_info(alias)
            rh._json_response(info)

        elif action == "predictions":
            params = rh._query_params()
            date = params.get("date")
            top_k = int(params.get("top_k", 30))
            result = _service().get_predictions(alias, date=date, top_k=top_k)
            rh._json_response(result)

        elif action == "dates":
            dates = _service().get_available_dates(alias)
            rh._json_response({"dates": dates})

        elif action == "stock":
            if len(action_parts) < 2 or not action_parts[1]:
                return rh._json_response({"error": "Missing instrument in /stock/{instrument}"}, status=400)
            instrument = action_parts[1]
            result = _service().get_stock_prediction(alias, instrument)
            rh._json_response(result)

        elif action == "report":
            result = _service().get_backtest_report(alias)
            rh._json_response(result)

        else:
            rh._json_response({"error": f"Unknown model action: {action}"}, status=404)

    except KeyError as exc:
        rh._json_response({"error": str(exc)}, status=404)
    except (ValueError, FileNotFoundError) as exc:
        rh._json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("model_detail error for %s/%s", alias, action)
        rh._json_response({"error": str(exc)}, status=500)
