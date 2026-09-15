"""导出 v4.1 的 FP4 实验及失败日志，供 GitHub 网页上传。"""
import argparse
from pathlib import Path
from scripts.export_v3_results import ROOT, export_results as export_archive

def export_results(root, run_id=None, **kwargs):
    return export_archive(root, run_id, version="4.1", **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(export_results(args.project_root, args.run_id))
