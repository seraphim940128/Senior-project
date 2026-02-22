# python run_rehab_coach.py --src 0 --use-llm
# python run_rehab_coach.py --src 0/1
# python run_rehab_coach.py --src "test_video/elbow_flexion_right_75.MOV"
# "test_video/shoulder_forward_elevation_80.MOV"
# python run_rehab_coach.py --src 0 --target-action shoulder_forward_elevation
# python run_rehab_coach.py --src 0 --use-llm --pose-task-model pose_landmarker_heavy.task --target-action shoulder_forward_elevation
# python run_rehab_coach.py --src 0 --use-llm --llm-model gpt-5-mini --pose-task-model pose_landmarker_heavy.task --target-action shoulder_forward_elevation
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def put_text_chinese(img: np.ndarray, text: str, position: tuple, text_color: tuple, font_size: int = 24) -> np.ndarray:
    import cv2
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    
    try:
        font = ImageFont.truetype("msjh.ttc", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("simhei.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
    rgb_color = (text_color[2], text_color[1], text_color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime rehab coach pipeline.")
    parser.add_argument("--src", type=str, default="0", help="Camera index or video path.")
    parser.add_argument("--target-action", type=str, default=None, help="Target action to track and count.")
    parser.add_argument("--weights", type=str, default=None, help="Action classifier .pt path.")
    parser.add_argument("--yolo", type=str, default=None, help="YOLO weights path.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stable-frames", type=int, default=None)
    parser.add_argument("--stable-conf", type=float, default=None)
    parser.add_argument("--cooldown", type=float, default=None)
    parser.add_argument("--unity", action="store_true", help="Enable Unity TCP output.")
    parser.add_argument("--unity-host", type=str, default="127.0.0.1")
    parser.add_argument("--unity-port", type=int, default=5500)
    parser.add_argument("--unity-conf", type=float, default=None, help="Unity command confidence threshold.")
    parser.add_argument("--unity-cooldown", type=float, default=None, help="Unity command cooldown seconds.")
    parser.add_argument("--use-llm", action="store_true", help="Enable Layer3 LLM generation.")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--pose-task-model",
        type=str,
        default=None,
        help="Optional MediaPipe task model (.task) path for PoseLandmarker backend.",
    )
    parser.add_argument("--self-check", action="store_true", help="Run dependency and file checks then exit.")
    parser.add_argument("--no-window", action="store_true", help="Disable cv2 preview window.")
    return parser.parse_args()


def _module_exists(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def run_self_check(args: argparse.Namespace) -> int:
    from rehab_coach.config import AppConfig

    required_modules = ["cv2", "numpy", "torch", "torchvision", "ultralytics", "mediapipe"]
    missing = [name for name in required_modules if not _module_exists(name)]

    config = build_config(args, AppConfig)
    checks: Dict[str, bool] = {
        "action_weights_exists": config.models.action_weights.exists(),
        "yolo_weights_exists": config.models.yolo_weights.exists(),
    }

    print("=== Self Check ===")
    print(f"Missing modules: {missing if missing else 'None'}")
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if missing:
        return 1
    if not all(checks.values()):
        return 1
    return 0


def pick_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return torch.device(device_arg)


def build_config(args: argparse.Namespace, app_config_cls):
    config = app_config_cls()
    if args.weights:
        config.models.action_weights = Path(args.weights)
    if args.yolo:
        config.models.yolo_weights = Path(args.yolo)
    if args.stable_frames is not None:
        config.action.stable_frames = args.stable_frames
    if args.stable_conf is not None:
        config.action.confidence_threshold = args.stable_conf
    if args.cooldown is not None:
        config.action.cooldown_s = args.cooldown
    if args.unity_conf is not None:
        config.unity_confidence_threshold = args.unity_conf
    if args.unity_cooldown is not None:
        config.unity_command_cooldown_s = args.unity_cooldown
    config.unity_host = args.unity_host
    config.unity_port = args.unity_port
    return config


def _print_startup(config, device) -> None:
    print("=== Rehab Coach Startup ===")
    print(f"Device: {device}")
    print(f"Action weights: {config.models.action_weights}")
    print(f"YOLO weights: {config.models.yolo_weights}")
    print(f"Stable frames/conf: {config.action.stable_frames} / {config.action.confidence_threshold}")
    print(f"Feedback cooldown(s): {config.action.cooldown_s}")
    print(f"Unity command conf: {config.unity_confidence_threshold}")
    print(f"Unity command cooldown(s): {config.unity_command_cooldown_s}")

def print_session_report(history: list):
    """
    在程式結束時列印結構化的動作統計報告
    """
    if not history:
        print("\n=== Session Report: No actions detected ===\n")
        return

    print("\n" + "=" * 80)
    print(f"SESSION REPORT (Total Actions: {len(history)})")
    print(f"{'No.':<4} {'Action Name':<30} {'Dur(s)':<8} {'Conf':<6} {'Quality'}")
    print("-" * 80)

    for i, item in enumerate(history, 1):
        action = item.get("action", "Unknown")
        seg = item.get("segment", {})
        dur = seg.get("duration_s", 0.0)
        conf = seg.get("confidence_mean", 0.0)

        posture = item.get("posture_summary", {})
        issues = []
        if posture.get("primary_joint_range") == "insufficient": issues.append("Range")
        if posture.get("compensation") == "excessive": issues.append("Comp")
        if posture.get("symmetry") == "imbalanced": issues.append("Sym")
        if posture.get("stability") == "unstable": issues.append("Unstable")

        status = "Good" if not issues else f"Issues: {','.join(issues)}"
        
        print(f"{i:<4} {action:<30} {dur:<8.1f} {conf:<6.2f} {status}")
    
    print("=" * 80 + "\n")

def main() -> int:
    args = build_args()
    if args.self_check:
        return run_self_check(args)

    try:
        import cv2
    except Exception as exc:
        print(f"Missing dependency: cv2 ({exc})")
        return 1

    try:
        from rehab_coach.config import AppConfig
        from rehab_coach.layer1_action import Layer1ActionRecognizer
        from rehab_coach.layer2_pose import Layer2PoseEvaluator, PoseExtractor
        from rehab_coach.layer3_feedback import Layer3FeedbackGenerator
        from rehab_coach.pipeline import RehabCoachPipeline
        from rehab_coach.unity_socket import UnitySocketClient
    except Exception as exc:
        print(f"Import error: {exc}")
        return 1

    config = build_config(args, AppConfig)
    if not config.models.action_weights.exists():
        print(f"Action weights not found: {config.models.action_weights}")
        return 1
    if not config.models.yolo_weights.exists():
        print(f"YOLO weights not found: {config.models.yolo_weights}")
        return 1

    try:
        device = pick_device(args.device)
    except Exception as exc:
        print(str(exc))
        return 1

    layer1 = Layer1ActionRecognizer(config=config, device=device)
    pose_extractor = PoseExtractor(task_model_path=args.pose_task_model)
    if pose_extractor.backend == "none":
        print(
            "PoseExtractor disabled. Reason: "
            f"{pose_extractor.backend_reason}. "
            "Layer2 segment metrics will not be produced."
        )
    layer2 = Layer2PoseEvaluator(config.baseline)
    layer3 = Layer3FeedbackGenerator(use_llm=args.use_llm, model=args.llm_model)
    pipeline = RehabCoachPipeline(
        layer1=layer1,
        pose_extractor=pose_extractor,
        layer2=layer2,
        cooldown_s=config.action.cooldown_s,
    )
    unity_client = UnitySocketClient(
        host=config.unity_host,
        port=config.unity_port,
        enabled=args.unity,
    )
    last_unity_command_ts = 0.0
    layer3_feedback = None

    src = int(args.src) if args.src.isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open source: {args.src}")
        pipeline.close()
        return 1

    session_history = []

    target_action_count = 0
    action_status_msg = "Waiting for action..."

    _print_startup(config, device)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            now = time.time()
            output = pipeline.process_frame(frame, now)
            prediction = output["prediction"]
            summary = output["summary"]

            if summary is not None:
                session_history.append(summary)

                print("\n" + "="*50)
                print(f"[除錯資訊] 動作片段(Segment)結束，觸發判定邏輯")
                print(f"[除錯資訊] 模型辨識出之動作 (summary action): {summary.get('action')}")
                print(f"[除錯資訊] 當前設定之目標動作 (args.target_action): {args.target_action}")
                print(f"[除錯資訊] 片段持續時間: {summary.get('segment', {}).get('duration_s')} 秒")
                print(f"[除錯資訊] 片段平均信心度: {summary.get('segment', {}).get('confidence_mean')}")
                print("="*50)

                print(json.dumps(summary, ensure_ascii=False))

                # if args.target_action:
                #     if summary.get("action") == args.target_action:
                #         is_success = True 
                        
                #         if is_success:
                #             target_action_count += 1
                #             action_status_msg = "Action Completed!"
                #             print(f"\n[系統提示] 動作完成！ {args.target_action} 當前完成次數: {target_action_count}\n")
                #     else:
                #         print(f"\n[系統提示] 動作名稱不符，不列入計數。 (需要 {args.target_action}，實際為 {summary.get('action')})\n")

                if args.target_action and summary.get("action") == args.target_action:
                        posture = summary.get("posture_summary", {})
                        
                        is_success = (
                            posture.get("primary_joint_range") != "insufficient" and
                            posture.get("compensation") != "excessive" and
                            posture.get("symmetry") != "imbalanced" and
                            posture.get("stability") != "unstable"
                        )
                        
                        if is_success:
                            target_action_count += 1
                            action_status_msg = "Success!"
                            print(f"\n[系統提示] 動作成功判定！ {args.target_action} 當前完成次數: {target_action_count}\n")
                        else:
                            action_status_msg = "Failed (Posture Issue)"
                            print(f"\n[系統提示] 動作完成，但不列入計數（姿態品質未達標）。\n")

            feedback_event = output["feedback_event"]

            if summary is not None:
                print(json.dumps(summary, ensure_ascii=False))

            if feedback_event is not None:
                layer3_feedback = layer3.generate(feedback_event)
                print(
                    json.dumps(
                        {
                            "action": feedback_event.get("action"),
                            "coach_text": layer3_feedback.get("coach_text"),
                            "ui_hint": layer3_feedback.get("ui_hint"),
                        },
                        ensure_ascii=False,
                    )
                )

            if prediction is not None:
                command = config.unity_action_map.get(prediction.action_label)
                should_send_command = (
                    args.unity
                    and command is not None
                    and prediction.confidence >= config.unity_confidence_threshold
                    and (now - last_unity_command_ts >= config.unity_command_cooldown_s)
                )
                if should_send_command and unity_client.send_command(command):
                    last_unity_command_ts = now

            if not args.no_window:
                if prediction is not None:
                    text = (
                        f"{prediction.action_label} "
                        f"{prediction.confidence * 100.0:.1f}% "
                        f"stable={prediction.is_stable}"
                    )
                    color = (20, 200, 40) if prediction.is_supported else (0, 180, 255)
                    cv2.putText(
                        frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

                if prediction.is_supported and prediction.stable_frames > 0:
                        bar_x, bar_y = 10, 45
                        bar_w, bar_h = 200, 15
                        
                        progress = min(1.0, prediction.stable_count / prediction.stable_frames)
                        filled_w = int(bar_w * progress)
                        
                        bar_color = (0, 255, 0) if prediction.is_stable else (0, 255, 255)
                        
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                        if filled_w > 0:
                            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)

                if layer3_feedback is not None:
                    frame = put_text_chinese(
                        frame,
                        layer3_feedback.get("ui_hint", ""),
                        (10, 85),
                        text_color=(0, 90, 255),
                        font_size=22
                    )

                if args.target_action:
                    frame = put_text_chinese(
                        frame,
                        f"Target: {args.target_action} | Count: {target_action_count}",
                        (10, 115),
                        text_color=(255, 255, 0),
                        font_size=20
                    )
                    frame = put_text_chinese(
                        frame,
                        f"Status: {action_status_msg}",
                        (10, 145),
                        text_color=(255, 255, 0),
                        font_size=20
                    )

                cv2.imshow("Rehab Coach", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pipeline.close()
        cv2.destroyAllWindows()
        print_session_report(session_history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
