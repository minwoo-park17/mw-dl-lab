from __future__ import annotations

from pathlib import Path
from PIL import Image, UnidentifiedImageError

# 변환 대상 확장자만: 필요하면 여기에 추가/삭제
IN_EXTS = {".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}  # png는 제외(재저장 안 함)

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def to_png(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    with Image.open(src) as im:
        # PNG 저장에 안전한 모드로 정리
        if im.mode in ("P", "LA"):
            im = im.convert("RGBA")
        elif im.mode == "CMYK":
            im = im.convert("RGB")
        elif im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA") if "A" in im.mode else im.convert("RGB")

        im.save(dst, format="PNG", optimize=True)

def convert_folder_images_to_png(input_dir: str, output_dir: str = "output_png") -> None:
    in_root = Path(input_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not in_root.is_dir():
        raise FileNotFoundError(f"입력 폴더가 없음: {in_root}")

    converted, skipped, errors = 0, 0, 0

    for src in in_root.rglob("*"):
        if not src.is_file():
            continue

        ext = src.suffix.lower()
        if ext not in IN_EXTS:
            skipped += 1
            continue

        rel = src.relative_to(in_root)
        dst = (out_root / rel).with_suffix(".png")

        try:
            to_png(src, dst)
            converted += 1
            print(f"[OK] {rel} -> {dst.relative_to(out_root)}")
        except (UnidentifiedImageError, OSError) as e:
            errors += 1
            print(f"[ERR] {rel} ({e})")

    print("\n=== DONE ===")
    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped} (not target ext)")
    print(f"Errors:    {errors}")
    print(f"Output:    {out_root}")

if __name__ == "__main__":
    # 여기만 네 폴더로 바꿔서 실행
    convert_folder_images_to_png(
        input_dir=rf"D:\Dataset\food",
        output_dir=rf"D:\Dataset\food",
    )
