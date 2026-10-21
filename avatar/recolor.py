"""Turn the CC0 VRoid sample 'Sakurada Fumiriya' into a black-haired version.

Rewrites the embedded hair and eyebrow textures of a VRM 0.x (.glb) file and rebuilds
the binary chunk, keeping every other byte of the model as it was.
"""

import io
import json
import struct
import sys

from PIL import Image, ImageOps

HAIR_IMAGES = {"M00_000_HairBack_00", "F00_000_Hair_00_01", "F00_000_Hair_00_02", "M00_000_FaceBrow_00"}
HAIR_MATERIALS = ("_HAIR",)


def darken(png_bytes):
    """Keep the strand shading but map it into near-black with a cool blue-grey sheen."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    r, g, b, a = im.split()
    lum = ImageOps.grayscale(im.convert("RGB"))
    # Original light hair spans roughly 90..255; squeeze into 18..95.
    dark = lum.point(lambda v: int(18 + (v / 255) ** 1.6 * 77))
    rgb = Image.merge("RGB", (dark, dark.point(lambda v: int(v * 1.02)), dark.point(lambda v: min(255, int(v * 1.12)))))
    out = Image.merge("RGBA", (*rgb.split(), a))
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def blank(png_bytes):
    """Same size, fully black: no emission."""
    im = Image.open(io.BytesIO(png_bytes))
    buf = io.BytesIO()
    Image.new("RGB", im.size, (0, 0, 0)).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def main(src, dst):
    data = open(src, "rb").read()
    json_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + json_len])
    bin_start = 20 + json_len + 8
    bin_len = struct.unpack("<I", data[20 + json_len:24 + json_len])[0]
    blob = data[bin_start:bin_start + bin_len]

    views = gltf["bufferViews"]
    chunks = [blob[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]] for v in views]
    for img in gltf["images"]:
        if img.get("name") == "F00_000_Hair_00_spe":
            # Used as the hair's emission map: left orange, it glows as red streaks on black hair.
            chunks[img["bufferView"]] = blank(chunks[img["bufferView"]])
            print("blanked", img["name"])
        elif img.get("name") in HAIR_IMAGES:
            chunks[img["bufferView"]] = darken(chunks[img["bufferView"]])
            print("recoloured", img["name"])

    # Hair materials: neutral tint and a dark shade colour so the MToon shadow isn't brown.
    for mat in gltf["extensions"]["VRM"]["materialProperties"]:
        if mat["name"].endswith(HAIR_MATERIALS) or "_HAIR_" in mat["name"]:
            vec = mat["vectorProperties"]
            vec["_Color"] = [1, 1, 1, 1]
            vec["_ShadeColor"] = [0.35, 0.36, 0.42, 1]
    for mat in gltf["materials"]:
        if "_HAIR" in mat["name"]:
            mat.setdefault("pbrMetallicRoughness", {})["baseColorFactor"] = [1, 1, 1, 1]

    new_blob = bytearray()
    for v, c in zip(views, chunks):
        while len(new_blob) % 4:
            new_blob.append(0)
        v["byteOffset"] = len(new_blob)
        v["byteLength"] = len(c)
        new_blob += c
    while len(new_blob) % 4:
        new_blob.append(0)
    gltf["buffers"][0]["byteLength"] = len(new_blob)
    gltf["extensions"]["VRM"]["meta"]["title"] = "Desk Buddy (from VRoid sample 桜田 史利矢)"

    js = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(new_blob)
    out = struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
    out += struct.pack("<I4s", len(new_blob), b"BIN\x00") + bytes(new_blob)
    open(dst, "wb").write(out)
    print(dst, len(out) // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
