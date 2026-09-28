"""
Synthetic invoice and receipt image generator for FinGuard AI.
Generates realistic financial documents with pixel-precise ground truth bounding boxes and field labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def render_invoice_document(
    vendor: str,
    invoice_no: str,
    date_str: str,
    items: List[Tuple[str, int, float]],
    tax_rate: float = 0.0825,
    tamper_tax: bool = False,
    skew_angle: float = 0.0,
    output_path: Path | str | None = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Renders a realistic invoice image and returns (PIL Image, ground_truth_dict).
    """
    width, height = 800, 1050
    # Clean white/off-white background
    img = Image.new("RGB", (width, height), color=(252, 252, 252))
    draw = ImageDraw.Draw(img)

    # Use default font or basic PIL bitmap font
    font = ImageFont.load_default()

    ground_truth = {
        "vendor": vendor,
        "invoice_no": invoice_no,
        "date": date_str,
        "items": [],
        "subtotal": 0.0,
        "tax": 0.0,
        "total": 0.0,
        "bounding_boxes": {},
    }

    # 1. Header & Vendor
    draw.rectangle([(40, 40), (760, 110)], fill=(240, 244, 248), outline=(200, 210, 225))
    draw.text((60, 55), f"** {vendor.upper()} **", fill=(20, 35, 60), font=font)
    draw.text((60, 75), "Official Commercial Invoice & Payment Receipt", fill=(80, 90, 105), font=font)
    ground_truth["bounding_boxes"]["vendor"] = [60, 50, 400, 75]

    # 2. Meta Info Box
    draw.line([(40, 130), (760, 130)], fill=(180, 190, 205), width=2)
    draw.text((60, 145), f"INVOICE NUMBER: {invoice_no}", fill=(30, 30, 30), font=font)
    ground_truth["bounding_boxes"]["invoice_no"] = [60, 145, 300, 165]

    draw.text((450, 145), f"DATE: {date_str}", fill=(30, 30, 30), font=font)
    ground_truth["bounding_boxes"]["date"] = [450, 145, 650, 165]

    draw.text((60, 175), "STATUS: PAID IN FULL", fill=(30, 120, 50), font=font)
    draw.text((450, 175), "PAYMENT METHOD: Electronic Card Settlement", fill=(70, 70, 70), font=font)
    draw.line([(40, 205), (760, 205)], fill=(180, 190, 205), width=2)

    # 3. Items Table Header
    y = 230
    draw.rectangle([(40, y), (760, y + 30)], fill=(225, 235, 245))
    draw.text((60, y + 8), "ITEM DESCRIPTION", fill=(30, 30, 30), font=font)
    draw.text((420, y + 8), "QTY", fill=(30, 30, 30), font=font)
    draw.text((510, y + 8), "UNIT PRICE", fill=(30, 30, 30), font=font)
    draw.text((650, y + 8), "LINE TOTAL", fill=(30, 30, 30), font=font)

    y += 45
    subtotal = 0.0
    for desc, qty, price in items:
        line_total = qty * price
        subtotal += line_total
        ground_truth["items"].append({"desc": desc, "qty": qty, "price": price, "total": line_total})

        draw.text((60, y), desc, fill=(40, 40, 40), font=font)
        draw.text((430, y), str(qty), fill=(40, 40, 40), font=font)
        draw.text((510, y), f"${price:.2f}", fill=(40, 40, 40), font=font)
        draw.text((650, y), f"${line_total:.2f}", fill=(40, 40, 40), font=font)
        draw.line([(40, y + 25), (760, y + 25)], fill=(240, 240, 240), width=1)
        y += 35

    subtotal = round(subtotal, 2)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2) if not tamper_tax else round(subtotal + tax + 75.0, 2)

    ground_truth["subtotal"] = subtotal
    ground_truth["tax"] = tax
    ground_truth["total"] = total

    # 4. Totals Calculation Box
    y = max(y + 30, 600)
    draw.rectangle([(400, y), (760, y + 150)], fill=(248, 250, 252), outline=(210, 215, 225))
    draw.text((430, y + 20), "Subtotal:", fill=(60, 60, 60), font=font)
    draw.text((650, y + 20), f"${subtotal:.2f}", fill=(60, 60, 60), font=font)

    draw.text((430, y + 55), "Sales Tax (8.25%):", fill=(60, 60, 60), font=font)
    draw.text((650, y + 55), f"${tax:.2f}", fill=(60, 60, 60), font=font)
    ground_truth["bounding_boxes"]["tax"] = [430, y + 50, 740, y + 75]

    draw.line([(420, y + 90), (740, y + 90)], fill=(180, 190, 205), width=2)
    draw.text((430, y + 105), "TOTAL AMOUNT DUE:", fill=(15, 25, 45), font=font)
    draw.text((650, y + 105), f"${total:.2f}", fill=(180, 20, 20), font=font)
    ground_truth["bounding_boxes"]["total"] = [430, y + 100, 740, y + 130]

    # 5. Footer Security Stamp
    draw.rectangle([(40, 850), (760, 960)], fill=(245, 248, 250), outline=(220, 225, 235))
    draw.text((60, 870), "SECURITY VERIFICATION & AUDIT HASH", fill=(100, 110, 125), font=font)
    draw.text((60, 895), f"Digital Signature: SHA256:{hash(invoice_no) & 0xffffffffffffffff:016x}", fill=(120, 130, 140), font=font)
    draw.text((60, 920), "Verified merchant electronic statement. All rights reserved.", fill=(140, 145, 155), font=font)

    # Optional Skew
    if abs(skew_angle) > 0.1:
        img = img.rotate(skew_angle, expand=False, fillcolor=(255, 255, 255))

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, format="PNG")

    return img, ground_truth


def generate_sample_invoice_suite(output_dir: str = "data/samples/invoices") -> List[Dict[str, Any]]:
    """
    Generate standard demo scenarios and benchmark evaluation invoices.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = [
        {
            "id": "INV_DEMO_01_MATCH",
            "filename": "inv_01_apple_exact_match.png",
            "vendor": "Apple Store Fifth Ave",
            "invoice_no": "INV-2026-8491",
            "date": "2026-01-14",
            "items": [("MacBook Pro 14 M3 Chip", 1, 1999.00), ("USB-C Power Adapter 96W", 1, 79.00)],
            "tax_rate": 0.0825,
            "skew": 0.0,
            "tamper_tax": False,
            "matched_txn": "TXN_0000001",
        },
        {
            "id": "INV_DEMO_02_MISMATCH_AMOUNT",
            "filename": "inv_02_bestbuy_amount_mismatch.png",
            "vendor": "Best Buy Electronics",
            "invoice_no": "INV-2026-9102",
            "date": "2026-01-18",
            "items": [("Sony 65 Inch 4K OLED TV", 1, 1499.00), ("HDMI 2.1 Ultra High Speed Cable", 2, 29.99)],
            "tax_rate": 0.0825,
            "skew": -1.5,
            "tamper_tax": False,
            "matched_txn": "TXN_0000002",
        },
        {
            "id": "INV_DEMO_03_STALE_DATE",
            "filename": "inv_03_delta_stale_date.png",
            "vendor": "Delta Airlines",
            "invoice_no": "INV-2025-1144",
            "date": "2025-11-20",  # Stale: > 50 days prior
            "items": [("Roundtrip Flight NYC-SFO Comfort+", 1, 620.00), ("Checked Baggage Fee", 1, 35.00)],
            "tax_rate": 0.075,
            "skew": 1.2,
            "tamper_tax": False,
            "matched_txn": "TXN_0000003",
        },
        {
            "id": "INV_DEMO_04_GHOST_VENDOR",
            "filename": "inv_04_ghost_vendor.png",
            "vendor": "Apex Shell Holdings LLC",
            "invoice_no": "INV-2026-4401",
            "date": "2026-01-22",
            "items": [("Consulting & IT Procurement Services", 1, 1850.00)],
            "tax_rate": 0.0,
            "skew": 0.0,
            "tamper_tax": False,
            "matched_txn": "TXN_0000004",
        },
        {
            "id": "INV_DEMO_05_TAMPERED_TAX",
            "filename": "inv_05_tampered_tax.png",
            "vendor": "Micro Center",
            "invoice_no": "INV-2026-7731",
            "date": "2026-01-25",
            "items": [("NVIDIA RTX 4080 Super GPU", 1, 999.00), ("Corsair 32GB DDR5 RAM", 1, 139.99)],
            "tax_rate": 0.0825,
            "skew": -0.8,
            "tamper_tax": True,  # Stated total is tampered!
            "matched_txn": "TXN_0000005",
        },
    ]

    # Add 20 diverse evaluation invoices
    sample_vendors = [
        "Target Store #104", "Walmart Supercenter", "Trader Joe's", "Delta Airlines",
        "Marriott Hotels", "B&H Photo Video", "Kroger Supermarket", "Starbucks Coffee",
        "Zara Retail", "Best Buy Electronics"
    ]
    for i in range(1, 21):
        v = sample_vendors[i % len(sample_vendors)]
        inv_no = f"INV-2026-B{1000 + i}"
        dt = f"2026-01-{(i % 25) + 1:02d}"
        price = 45.0 + (i * 18.5)
        suite.append({
            "id": f"INV_BENCH_{i:02d}",
            "filename": f"inv_bench_{i:02d}.png",
            "vendor": v,
            "invoice_no": inv_no,
            "date": dt,
            "items": [(f"Procurement Item #{i}", 1, price)],
            "tax_rate": 0.08,
            "skew": float((i % 5) - 2) * 0.4,
            "tamper_tax": False,
            "matched_txn": f"TXN_{i:07d}",
        })

    manifest = []
    for item in suite:
        img_path = out_dir / item["filename"]
        _, gt = render_invoice_document(
            vendor=item["vendor"],
            invoice_no=item["invoice_no"],
            date_str=item["date"],
            items=item["items"],
            tax_rate=item["tax_rate"],
            tamper_tax=item["tamper_tax"],
            skew_angle=item["skew"],
            output_path=img_path,
        )
        gt["id"] = item["id"]
        gt["filename"] = item["filename"]
        gt["matched_txn"] = item["matched_txn"]
        gt["skew"] = item["skew"]
        gt["tamper_tax"] = item["tamper_tax"]
        manifest.append(gt)

    manifest_path = out_dir / "ground_truth.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest)} synthetic invoices with ground truth at {out_dir}")
    return manifest


if __name__ == "__main__":
    generate_sample_invoice_suite()
