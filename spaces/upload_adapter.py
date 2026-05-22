"""
LoRA 어댑터를 HuggingFace Hub에 업로드하는 스크립트

사용법:
    python spaces/upload_adapter.py --repo your-username/vlm-defect-inspector-lora

사전 조건:
    huggingface-cli login  (또는 HF_TOKEN 환경변수 설정)
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT        = Path(__file__).parent.parent
ADAPTER_DIR = ROOT / "models" / "checkpoints" / "best_exp"

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True, help="HF Hub repo ID (예: username/vlm-defect-inspector-lora)")
parser.add_argument("--private", action="store_true", help="비공개 repo로 생성")
args = parser.parse_args()

if not ADAPTER_DIR.exists():
    raise FileNotFoundError(f"체크포인트 없음: {ADAPTER_DIR}\n05_experiments.ipynb를 먼저 실행하세요.")

api = HfApi()
api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
api.upload_folder(
    folder_path=str(ADAPTER_DIR),
    repo_id=args.repo,
    repo_type="model",
    commit_message="Upload QLoRA best_combo adapter",
)
print(f"\n업로드 완료: https://huggingface.co/{args.repo}")
print(f"\nSpaces 환경변수에 추가하세요:")
print(f"  HF_ADAPTER_REPO = {args.repo}")
