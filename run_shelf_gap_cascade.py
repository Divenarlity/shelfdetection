from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from run_empty_shelf_test import find_unicode_font
from src.cascade import (
    assign_shelf_ids, box_iou, context_tiles, contour_polygon, deduplicate, local_to_global, mask_bbox,
    mask_box_relation, padded_roi, section_for_box, validate_model_contract,
)
from src.geometry import classify_section


ROOT = Path(__file__).resolve().parent


def unique_run_dir(camera_id, source):
    base = ROOT / "runs" / "shelf_gap_cascade" / f"{camera_id}_{source.stem}"
    candidate, index = base, 1
    while candidate.exists():
        candidate = Path(f"{base}_{index}")
        index += 1
    candidate.mkdir(parents=True)
    return candidate


def put_text(image_bgr, items):
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw, font = ImageDraw.Draw(image), find_unicode_font(20)
    for xy, text, color in items:
        x, y = map(int, xy)
        size_box = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = size_box[2] - size_box[0], size_box[3] - size_box[1]
        x = max(2, min(x, image.width - text_width - 4))
        y = max(2, min(y, image.height - text_height - 4))
        box = draw.textbbox((x, y), text, font=font)
        draw.rectangle((box[0]-2, box[1]-2, box[2]+2, box[3]+2), fill=(255,255,255))
        draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def build_parser():
    parser = argparse.ArgumentParser(description="Raf maskesi -> raf ROI -> empty_shelf cascade inference")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--camera-id", default="ROBOT")
    parser.add_argument("--operation-mode", choices=("cascade","mapping","localization","mapping_update"), default="cascade")
    parser.add_argument("--store-id", default="MARKET-001")
    parser.add_argument("--map-dir", type=Path)
    parser.add_argument("--mapping-config", type=Path, default=ROOT/"configs"/"mapping.yaml")
    parser.add_argument("--session-id", default="mapping_pass_01")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-map", action="store_true")
    parser.add_argument("--id-mode", choices=("legacy_roi", "pose_geometry"), default="legacy_roi")
    parser.add_argument("--store-map", type=Path)
    parser.add_argument("--frame-metadata", type=Path)
    parser.add_argument("--pose-config", type=Path, default=ROOT/"configs"/"pose_geometry.yaml")
    parser.add_argument("--shelf-model", type=Path, default=ROOT/"best.pt")
    parser.add_argument("--empty-model", type=Path, default=ROOT/"empty_shelf_yolo11m_best.pt")
    parser.add_argument("--shelf-conf", type=float, default=0.25)
    parser.add_argument("--empty-conf", type=float, default=0.10)
    parser.add_argument("--shelf-imgsz", type=int, default=960)
    parser.add_argument("--empty-imgsz", type=int, default=640)
    parser.add_argument("--roi-padding-ratio", type=float, default=0.02)
    parser.add_argument("--min-mask-overlap", type=float, default=0.30)
    parser.add_argument("--dedup-iou", type=float, default=0.50)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--device")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--save-rois", action="store_true")
    parser.add_argument("--rebuild-shelf-map", action="store_true")
    return parser


def extract_shelves(result, image_shape):
    if result.masks is None:
        raise RuntimeError("Raf segmentasyon modeli maske üretmedi; kutu tabanlı fallback yasaktır.")
    h, w = image_shape[:2]
    masks = result.masks.data.cpu().numpy()
    confidences = result.boxes.conf.cpu().tolist()
    shelves = []
    for raw_mask, confidence in zip(masks, confidences):
        mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST) >= 0.5
        bbox = mask_bbox(mask)
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        ys, xs = np.where(mask)
        shelves.append({
            "mask": mask, "bbox": bbox, "centroid": [float(xs.mean()), float(ys.mean())],
            "segmentation_confidence": float(confidence),
        })
    return shelves


def predict_empty_rois(model, crops, conf, imgsz, device=None):
    """Boşluk modelini yalnızca raf crop listesi üzerinde çalıştırır."""
    if not crops:
        return []
    shapes={crop.shape for crop in crops}
    if len(shapes) != 1:
        raise ValueError("Boşluk modeli batch girdileri eşit boyutlu context tile'lar olmalıdır.")
    kwargs = {"source": crops, "conf": conf, "imgsz": imgsz, "rect": True,
              "batch": len(crops), "verbose": False}
    if device:
        kwargs["device"] = device
    return model.predict(**kwargs)


def run(args, shelf_model=None, empty_model=None):
    from ultralytics import YOLO

    started = time.perf_counter()
    stage_timings = {}
    for value, name in ((args.shelf_conf,"--shelf-conf"),(args.empty_conf,"--empty-conf"),
                        (args.min_mask_overlap,"--min-mask-overlap"),(args.dedup_iou,"--dedup-iou")):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} 0 ile 1 arasında olmalıdır.")
    source = args.source.resolve()
    frame_metadata = None
    if args.id_mode == "pose_geometry":
        from src.pose_geometry import load_frame_metadata
        if source.is_dir():
            metadata_files = sorted(source.glob("*.json"))
            if len(metadata_files) != 1:
                raise ValueError("pose_geometry klasör kaynağında tam bir frame metadata JSON beklenir.")
            args.frame_metadata = metadata_files[0]
        if not args.frame_metadata or not args.store_map:
            raise ValueError("pose_geometry için --frame-metadata ve --store-map zorunludur.")
        frame_metadata = load_frame_metadata(args.frame_metadata)
        image_path = Path(frame_metadata["image_path"])
        source = image_path if image_path.is_absolute() else args.frame_metadata.parent/image_path
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Görüntü okunamadı: {source}")
    h, w = image.shape[:2]
    stage_timings["input_load_seconds"] = time.perf_counter()-started
    shelf_path, empty_path = args.shelf_model.resolve(), args.empty_model.resolve()
    shelf_model = shelf_model or YOLO(str(shelf_path))
    empty_model = empty_model or YOLO(str(empty_path))
    shelf_names = validate_model_contract(shelf_model, "segment", "shelves", "Raf modeli")
    empty_names = validate_model_contract(empty_model, "detect", "empty_shelf", "Boşluk modeli")
    print(f"Raf modeli: {shelf_path}; task={shelf_model.task}; sınıflar={list(shelf_names.values())}")
    print(f"Boşluk modeli: {empty_path}; task={empty_model.task}; sınıflar={list(empty_names.values())}")
    model_ready = time.perf_counter()
    stage_timings["model_load_seconds"] = model_ready-started-stage_timings["input_load_seconds"]

    shelf_kwargs = {"source": image, "conf": args.shelf_conf, "imgsz": args.shelf_imgsz, "verbose": False}
    if args.device: shelf_kwargs["device"] = args.device
    shelf_result = shelf_model.predict(**shelf_kwargs)[0]  # Ana görüntü yalnızca raf modeline gider.
    shelves = extract_shelves(shelf_result, image.shape)
    shelf_done = time.perf_counter()
    stage_timings["shelf_segmentation_seconds"] = shelf_done-model_ready
    map_path = ROOT/"configs"/"shelf_maps"/f"{args.camera_id}.json"
    if args.id_mode == "pose_geometry":
        from src.pose_geometry import PoseConfig, assign_pose_ids, generate_candidates, load_store_map
        raw_config = yaml.safe_load(args.pose_config.read_text(encoding="utf-8"))
        pose_config = PoseConfig(**raw_config)
        candidates = generate_candidates(load_store_map(args.store_map), frame_metadata, pose_config)
        shelves = assign_pose_ids(shelves, candidates, image.shape, pose_config)
    else:
        candidates = []
        shelves = assign_shelf_ids(shelves, args.camera_id, map_path, image.shape, args.rebuild_shelf_map)
    geometry_done = time.perf_counter()
    stage_timings["geometry_mapping_seconds"] = geometry_done-shelf_done
    output_dir = unique_run_dir(args.camera_id, source)
    shared_report = ROOT/"outputs"/"model_report.json"
    if shared_report.exists():
        shutil.copy2(shared_report, output_dir/"model_report.json")
    rois_dir = output_dir/"rois"
    if args.save_rois: rois_dir.mkdir()

    regions = []
    for shelf in shelves:
        roi = padded_roi(shelf["bbox"], image.shape, args.roi_padding_ratio)
        shelf["roi_bbox"] = roi
        shelf["tiles"] = []
        tile_boxes = context_tiles(roi, image.shape, args.tile_size, args.tile_overlap)
        if not tile_boxes:
            shelf["status"] = "ROI_SKIPPED"
            continue
        for tile_index, tile_box in enumerate(tile_boxes, 1):
            x1,y1,x2,y2 = tile_box
            tile = image[y1:y2, x1:x2].copy()
            if tile.size == 0:
                continue
            tile_id=f"TILE-{tile_index:02d}"
            region={"shelf":shelf,"tile_id":tile_id,"bbox":tile_box,"image":tile}
            regions.append(region)
            shelf["tiles"].append({"tile_id":tile_id,"global_bbox_xyxy":tile_box,
                                   "width":x2-x1,"height":y2-y1})
            print(f"{shelf['shelf_id']} {tile_id} boşluk modeli girdisi: "
                  f"crop=({x1},{y1},{x2},{y2}), boyut={x2-x1}x{y2-y1}")
            if args.save_rois:
                prefix=f"{shelf['shelf_id']}_{tile_id}"
                cv2.imwrite(str(rois_dir/f"{prefix}_input.jpg"), tile)
                cv2.imwrite(str(rois_dir/f"{prefix}_mask.png"),
                            shelf["mask"][y1:y2,x1:x2].astype(np.uint8)*255)

    region_images=[region["image"] for region in regions]
    empty_results = predict_empty_rois(
        empty_model, region_images, args.empty_conf, args.empty_imgsz, args.device
    )  # Tam görüntü asla bu modele gönderilmez.
    empty_done = time.perf_counter()
    stage_timings["empty_detection_seconds"] = empty_done-geometry_done
    diagnostic_results = predict_empty_rois(
        empty_model, region_images, 0.01, args.empty_imgsz, args.device
    ) if args.save_rois and region_images else []

    raw_count, rejected, accepted = 0, [], []
    debug_images = {}
    diagnostic_regions = []
    diagnostic_iter=diagnostic_results if diagnostic_results else [None]*len(regions)
    for region, result, diagnostic_result in zip(regions, empty_results, diagnostic_iter):
        shelf, tile_id, tile_box = region["shelf"], region["tile_id"], region["bbox"]
        prefix=f"{shelf['shelf_id']}_{tile_id}"
        debug_images[prefix]={"accepted":region["image"].copy(),"rejected":region["image"].copy(),
                              "raw_conf001":region["image"].copy()}
        diagnostic_raw=[]
        if diagnostic_result is not None and diagnostic_result.boxes is not None:
            for box,confidence in zip(diagnostic_result.boxes.xyxy.cpu().tolist(),
                                      diagnostic_result.boxes.conf.cpu().tolist()):
                diagnostic_raw.append({"confidence":float(confidence),"roi_bbox_xyxy":box,
                                       "global_bbox_xyxy":local_to_global(box,tile_box,image.shape)})
                x1,y1,x2,y2=map(int,box)
                cv2.rectangle(debug_images[prefix]["raw_conf001"],(x1,y1),(x2,y2),(255,0,255),2)
        boxes = [] if result.boxes is None else list(zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()))
        region_accepted, region_rejected = [], []
        for local_box, confidence, class_id in boxes:
            if empty_names[int(class_id)] != "empty_shelf":
                continue
            raw_count += 1
            global_box = local_to_global(local_box, tile_box, image.shape)
            accepted_by_mask, overlap, center_inside = mask_box_relation(
                shelf["mask"], global_box, args.min_mask_overlap)
            record = {
                "confidence": float(confidence), "roi_bbox_xyxy": [float(v) for v in local_box],
                "global_bbox_xyxy": global_box, "assigned_shelf_id": shelf["shelf_id"],
                "tile_id":tile_id,"tile_global_bbox_xyxy":tile_box,
                "center_inside_mask": center_inside, "mask_overlap_ratio": overlap,
                "touches_tile_edge": bool(
                    local_box[0] <= 1 or local_box[1] <= 1
                    or local_box[2] >= region["image"].shape[1]-1
                    or local_box[3] >= region["image"].shape[0]-1
                ),
            }
            lx1,ly1,lx2,ly2 = map(int,local_box)
            if not accepted_by_mask:
                record["reason"] = "center_outside_and_mask_overlap_below_threshold"
                rejected.append(record)
                region_rejected.append(record)
                cv2.rectangle(debug_images[prefix]["rejected"],(lx1,ly1),(lx2,ly2),(0,140,255),2)
                continue
            candidate = shelf.get("projected_candidate")
            if candidate and len(candidate["polygon"]) == 4:
                center = ((global_box[0]+global_box[2])/2, (global_box[1]+global_box[3])/2)
                record["section"] = classify_section(center, candidate["polygon"], ("SOL","ORTA","SAĞ"), 0.02)
            else:
                record["section"] = section_for_box(global_box, shelf["bbox"])
            record["message"] = f"{shelf['shelf_id']} rafının {record['section']} bölümünde boşluk var"
            accepted.append(record)
            region_accepted.append(record)
            cv2.rectangle(debug_images[prefix]["accepted"],(lx1,ly1),(lx2,ly2),(0,0,255),2)
        diagnostic_regions.append({"shelf_id":shelf["shelf_id"],"tile_id":tile_id,
            "global_bbox_xyxy":tile_box,"input_width":tile_box[2]-tile_box[0],
            "input_height":tile_box[3]-tile_box[1],
            "model_call":{"imgsz":args.empty_imgsz,"rect":True,"batch":len(regions),
                          "conf":args.empty_conf},
            "raw_conf001_predictions":diagnostic_raw,"raw_predictions":region_accepted+region_rejected,
            "mask_rejected":region_rejected})

    final, removed = deduplicate(accepted, args.dedup_iou)
    duplicate_records=[item for item in accepted if item not in final]
    for shelf in shelves:
        detections = [item for item in final if item["assigned_shelf_id"] == shelf["shelf_id"]]
        sections = {key: sum(item["section"] == key for item in detections) for key in ("SOL","ORTA","SAĞ")}
        shelf["detections"], shelf["sections"] = detections, sections
        shelf["empty_space_count"] = len(detections)
        shelf["status"] = shelf.get("status") or ("EMPTY_SPACE_DETECTED" if detections else "NO_EMPTY_SPACE")
        shelf["mask_polygon"] = contour_polygon(shelf["mask"])

    if args.save_rois:
        for prefix, variants in debug_images.items():
            for kind, debug in variants.items():
                debug=put_text(debug,[((5,5),prefix,(220,0,0))])
                cv2.imwrite(str(rois_dir/f"{prefix}_{kind}.jpg"),debug)

    annotated = image.copy()
    overlay = image.copy()
    colors = [(255,100,0),(0,180,255),(180,0,255),(0,200,100),(200,150,0),(100,0,220)]
    text_items = []
    for index, shelf in enumerate(shelves):
        color = colors[index % len(colors)]
        overlay[shelf["mask"]] = color
        contour = np.asarray(shelf["mask_polygon"],np.int32)
        if len(contour)>=3: cv2.polylines(annotated,[contour],True,color,2)
        candidate = shelf.get("projected_candidate")
        if candidate:
            cv2.polylines(annotated,[np.rint(candidate["polygon"]).astype(np.int32)],True,(255,255,0),3)
        x1,y1,_,_ = shelf["bbox"]
        empty_text = "Boşluk yok" if not shelf["detections"] else f"{len(shelf['detections'])} boşluk"
        mapping = f" | map={shelf.get('mapping_confidence', 1.0):.2f}" if args.id_mode=="pose_geometry" else ""
        text_items.append(((x1,max(0,y1-24)),f"{shelf['shelf_id']}{mapping} | {empty_text}",(0,0,0)))
    annotated = cv2.addWeighted(annotated,.65,overlay,.35,0)
    for detection in final:
        x1,y1,x2,y2 = map(int,detection["global_bbox_xyxy"])
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),3)
        text_items.append(((x1,y1+3),
            f"{detection['assigned_shelf_id']} | {detection['section']} | Boşluk ({detection['confidence']:.2f})",
            (220,0,0)))
    if not shelves:
        text_items.append(((10,10),"Raf tespit edilmedi; boşluk analizi yapılmadı",(220,0,0)))
    annotated = put_text(annotated,text_items)
    annotated_path = output_dir/"annotated.jpg"
    if args.save_annotated or True:
        cv2.imwrite(str(annotated_path),annotated)

    json_shelves = []
    for shelf in shelves:
        json_shelves.append({
            "shelf_id":shelf["shelf_id"],"segmentation_confidence":shelf["segmentation_confidence"],
            "mask_polygon":shelf["mask_polygon"],"mask_bbox_xyxy":shelf["bbox"],
            "roi_bbox_xyxy":shelf.get("roi_bbox"),"tiles":shelf.get("tiles",[]),
            "centroid":shelf["centroid"],"status":shelf["status"],
            "mapping_confidence":shelf.get("mapping_confidence"),
            "mapping_scores":shelf.get("mapping_scores"),
            "empty_space_count":shelf["empty_space_count"],"sections":shelf["sections"],
            "empty_count":shelf["empty_space_count"],
            "empty_sections":[key for key,value in shelf["sections"].items() if value],
            "detections":shelf["detections"],
        })
    elapsed = time.perf_counter() - started
    stage_timings["postprocess_report_seconds"] = elapsed-(empty_done-started)
    payload = {
        "inference_mode":"shelf_roi_cascade","operation_mode":args.operation_mode,
        "id_mode":"persistent_pose_map" if args.operation_mode=="localization" else args.id_mode,
        "frame_id":frame_metadata.get("frame_id") if frame_metadata else None,
        "camera_id":args.camera_id,"source_image":str(source),
        "geometric_candidates":[{"shelf_id":c["shelf_id"],"polygon":c["polygon"].tolist(),
            "distance":c["distance"],"front_alignment":c["front_alignment"],
            "visible_area":c["visible_area"]} for c in candidates],
        "image_width":w,"image_height":h,"shelf_model":str(shelf_path),"empty_shelf_model":str(empty_path),
        "shelf_confidence_threshold":args.shelf_conf,"empty_confidence_threshold":args.empty_conf,
        "roi_padding_ratio":args.roi_padding_ratio,"tile_size":args.tile_size,
        "tile_overlap":args.tile_overlap,"min_mask_overlap":args.min_mask_overlap,
        "dedup_iou":args.dedup_iou,"detected_shelf_count":len(shelves),
        "detected_mask_count":len(shelves),"visible_map_candidate_count":len(candidates),
        "matched_count":sum(s["shelf_id"]!="UNKNOWN_SHELF" for s in shelves),
        "unknown_count":sum(s["shelf_id"]=="UNKNOWN_SHELF" for s in shelves),
        "rejected_empty_count":len(rejected),"duplicate_removed_count":removed,
        "timing":{**stage_timings,"total_seconds":elapsed,"fps":1.0/max(elapsed,1e-9)},
        "roi_inference_count":len(regions),"raw_roi_detection_count":raw_count,
        "accepted_detection_count":len(accepted),"rejected_outside_mask_count":len(rejected),
        "removed_duplicate_count":removed,"final_empty_space_count":len(final),
        "shelves":json_shelves,"rejected_detections":rejected,
    }
    (output_dir/"results.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    diagnostic_candidates=[prediction for region in diagnostic_regions
                           for prediction in region["raw_conf001_predictions"]]
    for prediction in diagnostic_candidates:
        prediction["target_iou"]=box_iou(prediction["global_bbox_xyxy"],[196,251,302,313])
    nearest_target=max(diagnostic_candidates,key=lambda item:item["target_iou"],default=None)
    diagnostics={"inference_strategy":"fixed_square_context_tiles_equal_shape_batch",
        "source_image":str(source),"target_regression_bbox_xyxy":[196,251,302,313],
        "tile_size":args.tile_size,"tile_overlap":args.tile_overlap,
        "regions":diagnostic_regions,"mask_rejected":rejected,
        "duplicate_detections_removed":duplicate_records,
        "nearest_target_prediction_conf001":nearest_target,"final_predictions":final,
        "stage_counts":{"tiles":len(regions),"raw_conf010":raw_count,
                        "mask_rejected":len(rejected),"duplicates_removed":removed,"final":len(final)}}
    (output_dir/"diagnostics.json").write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output_dir/"shelf_summary.csv").open("w",newline="",encoding="utf-8-sig") as file:
        writer=csv.writer(file); writer.writerow(["shelf_id","status","empty_space_count","SOL","ORTA","SAĞ"])
        for shelf in json_shelves:
            writer.writerow([shelf["shelf_id"],shelf["status"],shelf["empty_space_count"],
                             shelf["sections"]["SOL"],shelf["sections"]["ORTA"],shelf["sections"]["SAĞ"]])
    display_id_mode="persistent_pose_map" if args.operation_mode=="localization" else args.id_mode
    print(f"\nKamera: {args.camera_id}\nID modu: {display_id_mode}\nRaf işleme modu: shelf_roi_cascade\nTespit edilen raf: {len(shelves)}")
    if args.id_mode=="pose_geometry":
        print(f"Eşleşen bilinen raf: {sum(s['shelf_id']!='UNKNOWN_SHELF' for s in shelves)}")
        print(f"UNKNOWN_SHELF: {sum(s['shelf_id']=='UNKNOWN_SHELF' for s in shelves)}")
    print(f"Boşluk modeli tarafından işlenen ROI/tile: {len(regions)}\nHam ROI boşluk tahmini: {raw_count}")
    print(f"Maske dışında reddedilen: {len(rejected)}\nKaldırılan duplicate: {removed}\nFinal boşluk sayısı: {len(final)}")
    for shelf in json_shelves:
        parts=[f"{key} bölümünde {value} boşluk" for key,value in shelf["sections"].items() if value]
        print(f"{shelf['shelf_id']}: {', '.join(parts) if parts else 'Boşluk yok'}")
    print(f"Çıktı klasörü: {output_dir.resolve()}")
    if args.show:
        cv2.imshow("shelf_gap_cascade",annotated); cv2.waitKey(0); cv2.destroyAllWindows()
    return payload, output_dir


def main():
    args=build_parser().parse_args()
    if args.operation_mode in ("mapping","mapping_update"):
        if not args.map_dir: raise ValueError("--map-dir mapping modunda zorunludur.")
        from src.persistent_runner import run_mapping
        run_mapping(args,args.operation_mode=="mapping_update")
    else:
        if args.operation_mode=="localization":
            if not args.map_dir: raise ValueError("--map-dir localization modunda zorunludur.")
            from src.persistent_map import ShelfMapStore
            from src.persistent_runner import persistent_map_as_geometry
            persistent=ShelfMapStore(args.map_dir,args.store_id).load()
            runtime_map=args.map_dir/"localization_runtime_map.json"
            runtime_map.write_text(json.dumps(persistent_map_as_geometry(persistent),indent=2),encoding="utf-8")
            args.store_map=runtime_map;args.id_mode="pose_geometry"
            print(f"Sistem modu: localization\nID modu: persistent_pose_map")
            metadata_files=sorted(args.source.glob("*.json")) if args.source.is_dir() else [args.source]
            if not metadata_files: raise ValueError("Localization kaynağında frame metadata bulunamadı.")
            import argparse as _argparse
            for metadata in metadata_files:
                frame_args=_argparse.Namespace(**vars(args))
                frame_args.source=metadata;frame_args.frame_metadata=metadata
                run(frame_args)
        else:
            run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
