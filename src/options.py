from __future__ import annotations

from pathlib import Path

from src.pipeline import PipelineConfig


def add_pipeline_arguments(parser, root: Path):
    parser.add_argument("--shelf-model", type=Path, default=root / "best.pt")
    parser.add_argument("--empty-model", type=Path, default=root / "empty_shelf_yolo11m_best.pt")
    parser.add_argument("--store-map", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, default=root / "configs" / "pose_geometry.yaml")
    parser.add_argument("--shelf-conf", type=float, default=0.25)
    parser.add_argument("--empty-conf", type=float, default=0.10)
    parser.add_argument("--shelf-imgsz", type=int, default=960)
    parser.add_argument("--empty-imgsz", type=int, default=640)
    parser.add_argument("--roi-padding-ratio", type=float, default=0.02)
    parser.add_argument("--empty-roi-mode", choices=("full_shelf", "tiles"), default="full_shelf")
    parser.add_argument("--min-mask-overlap", type=float, default=0.30)
    parser.add_argument("--dedup-iou", type=float, default=0.50)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--device")
    return parser


def pipeline_config_from_args(args):
    return PipelineConfig(
        shelf_conf=args.shelf_conf,
        empty_conf=args.empty_conf,
        shelf_imgsz=args.shelf_imgsz,
        empty_imgsz=args.empty_imgsz,
        roi_padding_ratio=args.roi_padding_ratio,
        empty_roi_mode=args.empty_roi_mode,
        min_mask_overlap=args.min_mask_overlap,
        dedup_iou=args.dedup_iou,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        device=args.device,
    )
