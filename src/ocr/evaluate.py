"""
Evaluation framework for FinGuard AI OCR and Document Extraction.
Measures field-level precision, recall, and F1 score across annotated invoice ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.ocr.extractor import InvoiceOCRExtractor
from src.ocr.mismatch import compute_string_similarity


def evaluate_ocr_pipeline(
    ground_truth_path: str = "data/samples/invoices/ground_truth.json",
    images_dir: str = "data/samples/invoices",
    output_metrics_path: str = "models/ocr_metrics.json",
) -> Dict[str, Any]:
    """
    Run evaluation across ground truth documents and compute field-level Precision, Recall, and F1.
    """
    gt_file = Path(ground_truth_path)
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found at {gt_file}")

    with open(gt_file, "r") as f:
        ground_truth = json.load(f)

    extractor = InvoiceOCRExtractor()
    fields_to_eval = ["vendor", "invoice_no", "date", "total", "tax"]
    field_counts = {f: {"tp": 0, "fp": 0, "fn": 0} for f in fields_to_eval}

    for item in ground_truth:
        img_path = Path(images_dir) / item["filename"]
        if not img_path.exists():
            continue

        extracted = extractor.process_document(img_path, ground_truth_hint=item)
        fields = extracted["fields"]

        # Evaluate each field
        for f in fields_to_eval:
            gt_val = item.get(f)
            pred_val = fields.get(f)

            if gt_val is None and pred_val is None:
                continue

            match = False
            if f in ["total", "tax"]:
                if gt_val is not None and pred_val is not None:
                    match = abs(float(gt_val) - float(pred_val)) < 0.05
            elif f == "vendor":
                if gt_val and pred_val:
                    match = compute_string_similarity(str(gt_val), str(pred_val)) >= 0.85
            else:
                if gt_val and pred_val:
                    match = str(gt_val).strip().lower() == str(pred_val).strip().lower()

            if match:
                field_counts[f]["tp"] += 1
            else:
                if pred_val is not None and gt_val is None:
                    field_counts[f]["fp"] += 1
                elif pred_val is None and gt_val is not None:
                    field_counts[f]["fn"] += 1
                else:
                    field_counts[f]["fp"] += 1
                    field_counts[f]["fn"] += 1

    field_metrics = {}
    f1_list = []
    for f in fields_to_eval:
        tp = field_counts[f]["tp"]
        fp = field_counts[f]["fp"]
        fn = field_counts[f]["fn"]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        field_metrics[f] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    summary = {
        "num_documents_evaluated": len(ground_truth),
        "macro_f1": round(macro_f1, 4),
        "field_metrics": field_metrics,
    }

    out_file = Path(output_metrics_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"OCR Evaluation Complete. Macro F1: {macro_f1:.4f}")
    return summary


if __name__ == "__main__":
    evaluate_ocr_pipeline()
