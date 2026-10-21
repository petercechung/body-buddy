"""Builds public/buddy.vrm: a stylized desk buddy, modelled entirely by this script.

    python3 avatar/build_avatar.py            # needs numpy and Pillow

No Blender, no photos, no downloaded assets: every mesh is a tube, ellipsoid or ribbon
generated here, skin weights are assigned per body part from bone positions, the face is
a separate mesh whose morph targets drive the VRM expressions, and textures (shirt print,
hair gloss, face palette) are drawn with Pillow. The output is plain glTF 2.0 + VRMC_vrm 1.0
+ VRMC_materials_mtoon, written byte by byte below.

Axes follow VRM 1.0: metres, +Y up, the avatar faces +Z, its left hand is on +X, T-pose.
"""
import io
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "public" / "buddy.vrm"
rng = np.random.default_rng(7)

# --- skeleton -----------------------------------------------------------------

# name, parent, world position of the joint (T-pose)
BONES = [
    ("hips", None, (0, 0.92, 0)),
    ("spine", "hips", (0, 1.00, 0)),
    ("chest", "spine", (0, 1.12, 0)),
    ("upperChest", "chest", (0, 1.23, 0)),
    ("neck", "upperChest", (0, 1.37, -0.005)),
    ("head", "neck", (0, 1.46, 0)),
    ("leftShoulder", "upperChest", (0.025, 1.325, -0.005)),
    ("leftUpperArm", "leftShoulder", (0.155, 1.325, -0.01)),
    ("leftLowerArm", "leftUpperArm", (0.40, 1.325, -0.01)),
    ("leftHand", "leftLowerArm", (0.625, 1.325, -0.01)),
    ("rightShoulder", "upperChest", (-0.025, 1.325, -0.005)),
    ("rightUpperArm", "rightShoulder", (-0.155, 1.325, -0.01)),
    ("rightLowerArm", "rightUpperArm", (-0.40, 1.325, -0.01)),
    ("rightHand", "rightLowerArm", (-0.625, 1.325, -0.01)),
    ("leftUpperLeg", "hips", (0.085, 0.88, 0)),
    ("leftLowerLeg", "leftUpperLeg", (0.088, 0.48, 0.005)),
    ("leftFoot", "leftLowerLeg", (0.09, 0.085, -0.005)),
    ("leftToes", "leftFoot", (0.09, 0.02, 0.11)),
    ("rightUpperLeg", "hips", (-0.085, 0.88, 0)),
    ("rightLowerLeg", "rightUpperLeg", (-0.088, 0.48, 0.005)),
    ("rightFoot", "rightLowerLeg", (-0.09, 0.085, -0.005)),
    ("rightToes", "rightFoot", (-0.09, 0.02, 0.11)),
]
BONE = {name: i for i, (name, _, _) in enumerate(BONES)}
BPOS = {name: np.array(p, float) for name, _, p in BONES}
NB = len(BONES)

HEAD_C = np.array([0, 1.565, 0.0])
HEAD_R = np.array([0.096, 0.112, 0.104])
EYE_Y, EYE_X = 1.548, 0.038


def smooth(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def chain(coord, joints, blend):
    """Weights along a chain of bones: joints = [(coord where bone starts, bone name), ...] ascending.
    Returns a dense (N, NB) weight matrix, blending linearly over +-blend around each joint."""
    coord = np.asarray(coord, float)
    W = np.zeros((len(coord), NB))
    cuts = [c for c, _ in joints[1:]]
    # piecewise: fraction belonging to bone k = step(c_k) - step(c_{k+1})
    steps = [np.ones_like(coord)] + [np.clip((coord - c + blend) / (2 * blend), 0, 1) for c in cuts] + [np.zeros_like(coord)]
    for k, (_, name) in enumerate(joints):
        W[:, BONE[name]] += steps[k] - steps[k + 1]
    return W


def solo(n, name):
    W = np.zeros((n, NB))
    W[:, BONE[name]] = 1
    return W


# --- geometry helpers ------------------------------------------------------------


class Part:
    """Vertices + triangles for one material, plus dense skin weights and optional morph deltas."""

    def __init__(self, pos, tris, uv, weights, smooth_group=True):
        self.pos = np.asarray(pos, float)
        self.tris = np.asarray(tris, np.int64).reshape(-1, 3)
        self.uv = np.asarray(uv, float)
        self.w = weights
        self.smooth_group = smooth_group
        self.morphs = {}


def orient(pos, tris, ref):
    """Flip triangles whose normal points against ref (per-triangle outward hint)."""
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    n = np.cross(b - a, c - a)
    bad = (n * ref).sum(1) < 0
    tris = tris.copy()
    tris[bad] = tris[bad][:, [0, 2, 1]]
    return tris


def tube(path, ru, rv, hint, segs=24, power=2.0, caps=(True, True), arc_u=False, tip=(False, False)):
    """A tube along path (n,3) with elliptical (or superelliptical) sections; ru along the frame
    axis closest to `hint`, rv along the other. Ring starts at -U so a seam sits opposite `hint`."""
    path = np.asarray(path, float)
    n = len(path)
    ru = np.broadcast_to(np.asarray(ru, float), (n,))
    rv = np.broadcast_to(np.asarray(rv, float), (n,))
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    hint = np.broadcast_to(np.asarray(hint, float), (n, 3))
    U = hint - (hint * T).sum(1, keepdims=True) * T
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = np.cross(T, U)
    ang = math.pi + 2 * math.pi * np.arange(segs + 1) / segs
    c, s = np.cos(ang), np.sin(ang)
    e = 2.0 / power
    cu, sv = np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e
    P = path[:, None, :] + ru[:, None, None] * cu[None, :, None] * U[:, None, :] + rv[:, None, None] * sv[None, :, None] * V[:, None, :]
    pos = P.reshape(-1, 3)
    if arc_u:
        d = np.linalg.norm(np.diff(P, axis=1), axis=2)
        cum = np.concatenate([np.zeros((n, 1)), np.cumsum(d, 1)], 1)
        u = cum / np.maximum(cum[:, -1:], 1e-9)
    else:
        u = np.broadcast_to(np.arange(segs + 1) / segs, (n, segs + 1))
    seglen = np.linalg.norm(np.diff(path, axis=0), axis=1)
    v = np.concatenate([[0], np.cumsum(seglen)]) / max(seglen.sum(), 1e-9)
    uv = np.stack([u, np.broadcast_to(v[:, None], (n, segs + 1))], -1).reshape(-1, 2)
    tris = []
    for i in range(n - 1):
        for j in range(segs):
            a, b = i * (segs + 1) + j, i * (segs + 1) + j + 1
            c2, d2 = a + segs + 1, b + segs + 1
            tris += [(a, c2, b), (b, c2, d2)]
    tris = np.array(tris)
    ref = pos[tris].mean(1) - np.repeat(path[:-1], segs * 2, axis=0)
    tris = orient(pos, tris, ref)
    extra_pos, extra_uv, extra_tris = [], [], []
    base = len(pos)
    for end, on in ((0, caps[0]), (n - 1, caps[1])):
        if not on:
            continue
        ring = np.arange(segs) + end * (segs + 1)
        ci = base + len(extra_pos)
        extra_pos.append(path[end])
        extra_uv.append(uv[ring[0]])
        t = np.array([(ci, ring[j], ring[(j + 1) % segs]) for j in range(segs)])
        out = -T[end] if end == 0 else T[end]
        extra_tris.append((t, out))
    if extra_pos:
        pos = np.vstack([pos, extra_pos])
        uv = np.vstack([uv, extra_uv])
        for t, out in extra_tris:
            tris = np.vstack([tris, orient(pos, t, np.broadcast_to(out, (len(t), 3)))])
    return pos, tris, uv


def ellipsoid(center, radii, rings=32, segs=48, deform=None):
    th = np.linspace(0, math.pi, rings + 1)
    ph = np.linspace(0, 2 * math.pi, segs + 1)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(TH) * np.sin(PH), np.cos(TH), np.sin(TH) * np.cos(PH)], -1).reshape(-1, 3)
    p = d * radii
    if deform is not None:
        p = deform(p, d)
    pos = p + center
    uv = np.stack([PH / (2 * math.pi), TH / math.pi], -1).reshape(-1, 2)
    tris = []
    for i in range(rings):
        for j in range(segs):
            a, b = i * (segs + 1) + j, i * (segs + 1) + j + 1
            tris += [(a, b, a + segs + 1), (b, b + segs + 1, a + segs + 1)]
    tris = np.array(tris)
    tris = orient(pos, tris, pos[tris].mean(1) - center)
    return pos, tris, uv, d


def strip(points, half_width, z_off=0.0):
    """Flat ribbon in the XY plane along a 2D polyline, facing +Z; half_width per point."""
    pts = np.asarray(points, float)
    hw = np.broadcast_to(np.asarray(half_width, float), (len(pts),))
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    top, bot = pts + nrm * hw[:, None], pts - nrm * hw[:, None]
    pos2 = np.empty((2 * len(pts), 2))
    pos2[0::2], pos2[1::2] = top, bot
    tris = []
    for i in range(len(pts) - 1):
        a = 2 * i
        tris += [(a, a + 1, a + 2), (a + 1, a + 3, a + 2)]
    return pos2, np.array(tris)


def fan(poly):
    """Convex polygon (k,2) -> fan triangles around its centroid."""
    poly = np.asarray(poly, float)
    pts = np.vstack([poly.mean(0), poly])
    k = len(poly)
    return pts, np.array([(0, 1 + j, 1 + (j + 1) % k) for j in range(k)])


# --- the body ---------------------------------------------------------------------

SKIN = "skin"
parts = {m: [] for m in ["skin", "shirt", "pants", "shoes", "sole", "hair", "face", "glasses"]}


def add(mat, pos, tris, uv, W, **kw):
    p = Part(pos, tris, uv, W, **kw)
    parts[mat].append(p)
    return p


def torso_weights(pos):
    y, ax = pos[:, 1], np.abs(pos[:, 0])
    W = chain(y, [(0, "hips"), (1.00, "spine"), (1.12, "chest"), (1.23, "upperChest"), (1.37, "neck")], 0.045)
    side = np.where(pos[:, 0] >= 0, 1, -1)
    sh = smooth(0.06, 0.13, ax) * smooth(1.20, 1.29, y)
    arm = smooth(0.13, 0.17, ax) * sh
    for s, pre in ((1, "left"), (-1, "right")):
        m = side == s
        W[m] *= (1 - sh[m])[:, None]
        W[m, BONE[pre + "Shoulder"]] += sh[m] - arm[m]
        W[m, BONE[pre + "UpperArm"]] += arm[m]
    return W


# shirt torso: (y, half depth, half width, z offset)
TORSO = np.array([
    (0.835, 0.108, 0.152, 0.004), (0.88, 0.104, 0.148, 0.004), (0.96, 0.096, 0.140, 0.004),
    (1.04, 0.094, 0.138, 0.006), (1.12, 0.100, 0.146, 0.010), (1.20, 0.104, 0.160, 0.010),
    (1.26, 0.100, 0.172, 0.006), (1.305, 0.092, 0.170, 0.0), (1.335, 0.080, 0.150, -0.004),
    (1.355, 0.066, 0.110, -0.005), (1.368, 0.054, 0.066, -0.005),
])
ys = np.linspace(TORSO[0, 0], TORSO[-1, 0], 34)
tor = np.stack([np.interp(ys, TORSO[:, 0], TORSO[:, k]) for k in range(4)], 1)
path = np.stack([np.zeros_like(ys), ys, tor[:, 3]], 1)
pos, tris, uv = tube(path, tor[:, 1], tor[:, 2], (0, 0, 1), segs=56, power=2.6, caps=(True, False), arc_u=True)
uv[:, 1] = 1 - (pos[:, 1] - ys[0]) / (ys[-1] - ys[0])  # image top = collar
SHIRT_V = (ys[0], ys[-1])
add("shirt", pos, tris, uv, torso_weights(pos))

# crew-neck rib
ang = np.linspace(0, 2 * math.pi, 33)
ring = np.stack([0.050 * np.sin(ang), np.full_like(ang, 1.368), -0.005 + 0.046 * np.cos(ang)], 1)
pos, tris, uv = tube(ring, 0.006, 0.006, (0, 1, 0), segs=8, caps=(False, False))
uv[:] = (0.5, 0.02)
add("shirt", pos, tris, uv, torso_weights(pos))

# neck
path = np.stack([np.zeros(10), np.linspace(1.30, 1.52, 10), np.full(10, -0.008)], 1)
pos, tris, uv = tube(path, 0.040, 0.041, (0, 0, 1), segs=20, caps=(False, False))
W = chain(pos[:, 1], [(0, "upperChest"), (1.37, "neck"), (1.46, "head")], 0.03)
add(SKIN, pos, tris, uv, W)


def head_deform(p, d):
    p = p.copy()
    s = np.clip(-d[:, 1], 0, 1)  # lower half
    # jaw: stays wide down to the jaw angle, then narrows to the chin
    p[:, 0] *= 1 - 0.20 * s ** 2
    front = d[:, 2] > 0
    p[front, 2] *= 1 - 0.10 * s[front] ** 2
    p[~front, 2] *= 1 - 0.30 * s[~front]
    p[:, 1] *= np.where(d[:, 1] < 0, 1 - 0.06 * s, 1)
    # slightly flatter, wider cheeks: push the lower front sides out a bit
    cheek = np.exp(-((d[:, 1] + 0.35) / 0.25) ** 2) * np.clip(d[:, 2], 0, 1)
    p[:, 0] *= 1 + 0.06 * cheek
    nose = np.exp(-(d[:, 0] / 0.06) ** 2 - ((d[:, 1] + 0.40) / 0.12) ** 2) * (d[:, 2] > 0)
    p[:, 2] += 0.010 * nose
    # a defined chin: the lower front centre reaches further down and a little forward
    chin = np.exp(-(d[:, 0] / 0.32) ** 2 - ((d[:, 1] + 0.80) / 0.22) ** 2) * (d[:, 2] > 0)
    p[:, 1] -= 0.012 * chin
    p[:, 2] += 0.024 * chin
    # jaw line: the underside from the chin back to below the ears stays low and fairly level
    under = smooth(0.55, 0.9, s) * smooth(-0.6, 0.2, d[:, 2])
    p[:, 1] -= 0.012 * under * (1 - np.abs(d[:, 2]))
    return p


hpos, htris, huv, _ = ellipsoid(HEAD_C, HEAD_R, rings=40, segs=56, deform=head_deform)
add(SKIN, hpos, htris, huv, solo(len(hpos), "head"))
HEAD_TRIS = hpos[htris]

# arms: skin from inside the torso to the wrist, short white sleeve on top
for side, pre in ((1, "left"), (-1, "right")):
    xs = np.linspace(0.10, 0.635, 30)
    r = np.interp(xs, [0.10, 0.18, 0.30, 0.40, 0.48, 0.62, 0.635], [0.046, 0.042, 0.038, 0.032, 0.033, 0.024, 0.024])
    path = np.stack([side * xs, np.full_like(xs, 1.325), np.full_like(xs, -0.01)], 1)
    pos, tris, uv = tube(path, r, r * 1.05, (0, 1, 0), segs=20, caps=(False, True))
    x = np.abs(pos[:, 0])
    W = chain(x, [(0, pre + "Shoulder"), (0.155, pre + "UpperArm"), (0.40, pre + "LowerArm"), (0.625, pre + "Hand")], 0.03)
    add(SKIN, pos, tris, uv, W)
    xs = np.linspace(0.095, 0.30, 14)
    r = np.interp(xs, [0.095, 0.16, 0.25, 0.30], [0.060, 0.057, 0.054, 0.055])
    path = np.stack([side * xs, np.full_like(xs, 1.325), np.full_like(xs, -0.01)], 1)
    pos, tris, uv = tube(path, r, r * 1.05, (0, 1, 0), segs=24, caps=(False, False))
    uv[:] = (0.0, 0.98)
    x = np.abs(pos[:, 0])
    W = chain(x, [(0, pre + "Shoulder"), (0.155, pre + "UpperArm")], 0.035)
    add("shirt", pos, tris, uv, W)
    # mitten hand + thumb
    c = np.array([side * 0.675, 1.322, -0.008])
    pos, tris, uv, _ = ellipsoid(c, np.array([0.050, 0.017, 0.037]), rings=12, segs=16)
    add(SKIN, pos, tris, uv, solo(len(pos), pre + "Hand"))
    path = np.array([[side * 0.640, 1.318, 0.018], [side * 0.665, 1.314, 0.040], [side * 0.688, 1.312, 0.050]])
    pos, tris, uv = tube(path, 0.0105, 0.0095, (0, 1, 0), segs=10)
    add(SKIN, pos, tris, uv, solo(len(pos), pre + "Hand"))

# trousers: pelvis block + loose legs
ys = np.linspace(0.76, 0.95, 10)
rz = np.interp(ys, [0.76, 0.85, 0.95], [0.086, 0.094, 0.088])
rx = np.interp(ys, [0.76, 0.85, 0.95], [0.142, 0.140, 0.130])
path = np.stack([np.zeros_like(ys), ys, np.full_like(ys, 0.002)], 1)
pos, tris, uv = tube(path, rz, rx, (0, 0, 1), segs=40, power=2.4)
W = np.zeros((len(pos), NB))
legw = smooth(0.90, 0.80, pos[:, 1]) * smooth(0.01, 0.06, np.abs(pos[:, 0]))
W[:, BONE["hips"]] = 1 - legw
W[pos[:, 0] >= 0, BONE["leftUpperLeg"]] = legw[pos[:, 0] >= 0]
W[pos[:, 0] < 0, BONE["rightUpperLeg"]] = legw[pos[:, 0] < 0]
add("pants", pos, tris, uv, W)
for side, pre in ((1, "left"), (-1, "right")):
    ys = np.linspace(0.86, 0.075, 26)
    r = np.interp(ys, [0.075, 0.2, 0.48, 0.75, 0.86], [0.074, 0.072, 0.074, 0.076, 0.066])
    xs = np.interp(ys, [0.075, 0.48, 0.86], [0.092, 0.090, 0.078])
    path = np.stack([side * xs, ys, np.full_like(ys, 0.004)], 1)
    pos, tris, uv = tube(path, r, r * 0.96, (0, 0, 1), segs=24, caps=(False, True))
    W = chain(-pos[:, 1], [(-9, "hips"), (-0.90, pre + "UpperLeg"), (-0.48, pre + "LowerLeg")], 0.04)
    add("pants", pos, tris, uv, W)
    # running shoe: rounded tube heel -> toe, white sole underneath
    zs = np.linspace(-0.075, 0.175, 22)
    h = np.interp(zs, [-0.075, -0.05, 0.03, 0.10, 0.155, 0.175], [0.050, 0.060, 0.062, 0.040, 0.030, 0.012])
    w = np.interp(zs, [-0.075, -0.05, 0.05, 0.12, 0.165, 0.175], [0.020, 0.040, 0.050, 0.052, 0.036, 0.012])
    path = np.stack([np.full_like(zs, side * 0.092), 0.022 + h, zs], 1)
    pos, tris, uv = tube(path, h, w, (0, 1, 0), segs=24, power=3.0)
    pos[:, 1] = np.maximum(pos[:, 1], 0.02)
    W = chain(pos[:, 2], [(-9, pre + "Foot"), (0.11, pre + "Toes")], 0.03)
    add("shoes", pos, tris, uv, W)
    path = np.stack([np.full_like(zs, side * 0.092), np.full_like(zs, 0.012), zs], 1)
    pos, tris, uv = tube(path, 0.012, w + 0.006, (0, 1, 0), segs=20, power=4.0)
    W = chain(pos[:, 2], [(-9, pre + "Foot"), (0.11, pre + "Toes")], 0.03)
    add("sole", pos, tris, uv, W)


# --- face: flat features projected onto the head, with morph targets --------------------


def surface_z(xy):
    """Front-most z of the head surface at each (x, y), by ray casting along -Z."""
    xy = np.asarray(xy, float)
    A, B, C = HEAD_TRIS[:, 0], HEAD_TRIS[:, 1], HEAD_TRIS[:, 2]
    out = np.full(len(xy), np.nan)
    for i0 in range(0, len(xy), 256):
        p = xy[i0:i0 + 256, None, :]
        v0, v1 = (B - A)[None, :, :2], (C - A)[None, :, :2]
        v2 = p - A[None, :, :2]
        den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        b1 = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den
        b2 = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den
        inside = (b1 >= -1e-9) & (b2 >= -1e-9) & (b1 + b2 <= 1 + 1e-9)
        z = A[None, :, 2] + b1 * (B - A)[None, :, 2] + b2 * (C - A)[None, :, 2]
        z = np.where(inside, z, -np.inf)
        out[i0:i0 + 256] = z.max(1)
    return out


FACE_PAL = {  # palette cell -> colour; the face texture is a 4x4 grid of flat colours
    "white": (250, 250, 252), "iris": (58, 42, 38), "pupil": (22, 16, 16), "lash": (34, 28, 30),
    "brow": (52, 42, 40), "mouth": (150, 70, 70), "line": (120, 64, 60), "blush": (240, 170, 160),
    "nose": (215, 160, 140), "irislight": (110, 82, 70),
}
PAL_KEYS = list(FACE_PAL)


def pal_uv(key):
    i = PAL_KEYS.index(key)
    return ((i % 4 + 0.5) / 4, (i // 4 + 0.5) / 4)


MORPHS = ["blinkLeft", "blinkRight", "happy", "sad", "relaxed", "surprised", "aa", "ih", "ou", "ee", "oh"]
face_items = []  # (xy (k,2), tris, layer, colour key, {morph: xy target})


def feature(xy, tris, layer, color, morphs):
    face_items.append((np.asarray(xy, float), np.asarray(tris), layer, color, morphs))


def lid_y(x, cx, side):
    # sleepy upper lid: fairly flat, a touch lower at the outer corner
    t = (x - cx) * side / 0.018
    return EYE_Y + 0.0055 - 0.002 * t + 0.002 * (1 - t ** 2)


def clip_top(poly, cx, side):
    poly = poly.copy()
    poly[:, 1] = np.minimum(poly[:, 1], lid_y(poly[:, 0], cx, side))
    return poly


def ellipse(cx, cy, a, b, k=28):
    t = np.linspace(0, 2 * math.pi, k, endpoint=False)
    return np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], 1)


for side, pre in ((1, "Left"), (-1, "Right")):
    cx = side * EYE_X
    A_, B_ = 0.0175, 0.0185
    close_y = EYE_Y - 0.004

    def eye_morphs(xy, lash=False):
        dx = (xy[:, 0] - cx) / A_
        arc = 1 - np.clip(dx, -1, 1) ** 2
        lid = lid_y(xy[:, 0], cx, side)
        off = xy[:, 1] - lid if lash else (xy[:, 1] - lid) * 0.03
        m = {}
        blink = xy.copy(); blink[:, 1] = close_y - 0.004 * arc + off
        m["blink" + pre] = blink
        happy = xy.copy(); happy[:, 1] = close_y + 0.002 + 0.006 * arc + off
        m["happy"] = happy
        relaxed = xy.copy(); relaxed[:, 1] = close_y - 0.001 - 0.002 * arc + (off if lash else (xy[:, 1] - lid) * 0.35)
        m["relaxed"] = relaxed
        sad = xy.copy()
        sad[:, 1] = np.where(xy[:, 1] > EYE_Y - 0.004, xy[:, 1] - 0.003 - 0.002 * (dx * side < 0), xy[:, 1])
        m["sad"] = sad
        sur = xy.copy(); sur[:, 1] = np.where(xy[:, 1] > EYE_Y - 0.01, xy[:, 1] + (0.006 if lash else 0.005 * (xy[:, 1] - EYE_Y + 0.01) / 0.02), xy[:, 1])
        m["surprised"] = sur
        return m

    sclera = clip_top(ellipse(cx, EYE_Y, A_, B_, 36), cx, side)
    xy, tr = fan(sclera)
    feature(xy, tr, 1, "white", eye_morphs(xy))
    iris = clip_top(ellipse(cx + side * 0.001, EYE_Y - 0.002, 0.0118, 0.0158, 32), cx, side)
    xy, tr = fan(iris)
    feature(xy, tr, 2, "iris", eye_morphs(xy))
    irl = clip_top(ellipse(cx + side * 0.001, EYE_Y - 0.008, 0.0085, 0.0075, 24), cx, side)
    xy, tr = fan(irl)
    feature(xy, tr, 3, "irislight", eye_morphs(xy))
    pup = clip_top(ellipse(cx + side * 0.001, EYE_Y - 0.0015, 0.0055, 0.0085, 24), cx, side)
    xy, tr = fan(pup)
    feature(xy, tr, 4, "pupil", eye_morphs(xy))
    for hx, hy, hr in ((-0.005, 0.002, 0.0034), (0.005, -0.009, 0.0017)):
        xy, tr = fan(ellipse(cx + hx, EYE_Y + hy, hr, hr, 14))
        feature(xy, tr, 5, "white", eye_morphs(xy))
    # upper lash: thick line along the lid with a small flick at the outer corner
    xs = np.linspace(cx - side * 0.019, cx + side * 0.021, 18)
    pts = np.stack([xs, lid_y(xs, cx, side) + 0.0012], 1)
    pts[-1] += (side * 0.003, -0.002)
    hw = 0.0018 + 0.0007 * np.sin(np.linspace(0, math.pi, 18))
    hw[0] = 0.0008
    xy, tr = strip(pts, hw)
    feature(xy, tr, 6, "lash", eye_morphs(xy, lash=True))
    # thin lower lash under the outer half
    xs = np.linspace(cx + side * 0.004, cx + side * 0.017, 8)
    pts = np.stack([xs, EYE_Y - 0.017 + 0.007 * ((xs - cx) * side / 0.018) ** 3], 1)
    xy, tr = strip(pts, 0.0006)
    m = eye_morphs(xy, lash=True)
    for k in ("blink" + pre, "happy", "relaxed"):
        m[k] = xy.copy()
    feature(xy, tr, 6, "line", m)
    # brow: soft, slightly straight
    xs = np.linspace(cx - side * 0.016, cx + side * 0.018, 12)
    t = (xs - cx) * side / 0.018
    pts = np.stack([xs, EYE_Y + 0.037 + 0.003 * (1 - t ** 2) - 0.002 * t], 1)
    xy, tr = strip(pts, 0.0019 - 0.0008 * (t + 1) / 2)
    tt = (xy[:, 0] - cx) * side / 0.018
    m = {}
    m["sad"] = xy + np.stack([0 * tt, 0.006 * np.clip(-tt, 0, 1) - 0.001], 1)
    m["happy"] = xy + np.stack([0 * tt, 0.003 + 0 * tt], 1)
    m["surprised"] = xy + np.stack([0 * tt, 0.006 + 0 * tt], 1)
    m["relaxed"] = xy + np.stack([0 * tt, -0.001 + 0 * tt], 1)
    feature(xy, tr, 1, "brow", m)
    # blush hatching
    for k in range(3):
        bx = side * (0.052 + 0.0055 * k)
        pts = np.array([[bx - side * 0.002, EYE_Y - 0.030], [bx + side * 0.002, EYE_Y - 0.024]])
        xy, tr = strip(pts, 0.0007)
        feature(xy, tr, 1, "blush", {})

# nose: a short soft stroke
xy, tr = strip(np.array([[0.002, 1.509], [-0.001, 1.506], [-0.004, 1.5055]]), 0.0009)
feature(xy, tr, 1, "nose", {})

# mouth: a strip between an upper and a lower lip curve; neutral is a thin, slightly smiling line
MOUTH_Y, MW = 1.479, 0.0105
NM = 15
t = np.linspace(-1, 1, NM)


def mouth(width=1.0, smile=0.0025, open_=0.0, up=0.0, shape=0.8):
    x = t * MW * width
    yu = MOUTH_Y + smile * t ** 2 + up * (1 - t ** 2)
    yl = yu - 0.0013 - open_ * np.clip(1 - t ** 2, 0, 1) ** shape
    return np.concatenate([np.stack([x, yu], 1), np.stack([x, yl], 1)])


mtris = []
for i in range(NM - 1):
    mtris += [(i, NM + i, i + 1), (i + 1, NM + i, NM + i + 1)]
feature(mouth(), mtris, 1, "mouth", {
    "aa": mouth(0.95, 0.002, 0.020, 0.002), "ih": mouth(1.15, 0.003, 0.007), "ou": mouth(0.55, 0.0, 0.011, 0.001, 0.5),
    "ee": mouth(1.25, 0.004, 0.009), "oh": mouth(0.75, 0.0, 0.016, 0.001, 0.6),
    "happy": mouth(1.35, 0.007, 0.010, 0.0, 0.9), "sad": mouth(0.9, -0.0045, 0.0005, 0.0015),
    "relaxed": mouth(1.1, 0.004, 0.0), "surprised": mouth(0.6, 0.0, 0.013, 0.001, 0.5),
})

# project everything onto the head; each layer sits a little further out
allxy = np.vstack([it[0] for it in face_items] + [m for it in face_items for m in it[4].values()])
zs = surface_z(allxy)
assert not np.isnan(zs).any() and np.isfinite(zs).all(), "face feature off the head"
zmap = {}
k = 0


def take(n):
    global k
    out = zs[k:k + n]
    k += n
    return out


base_z = [take(len(it[0])) for it in face_items]
morph_z = [{name: take(len(m)) for name, m in it[4].items()} for it in face_items]
fpos, ftris, fuv = [], [], []
fmorph = {m: [] for m in MORPHS}
off = 0
for (xy, tr, layer, color, morphs), z0, mz in zip(face_items, base_z, morph_z):
    lift = 0.0009 + 0.0006 * layer
    p = np.column_stack([xy, z0 + lift])
    fpos.append(p)
    ftris.append(tr + off)
    fuv.append(np.tile(pal_uv(color), (len(xy), 1)))
    for m in MORPHS:
        if m in morphs:
            fmorph[m].append(np.column_stack([morphs[m], mz[m] + lift]) - p)
        else:
            fmorph[m].append(np.zeros_like(p))
    off += len(xy)
fpos = np.vstack(fpos)
face = add("face", fpos, np.vstack(ftris), np.vstack(fuv), solo(len(fpos), "head"), smooth_group=False)
face.morphs = {m: np.vstack(v) for m, v in fmorph.items()}
face.normals = np.tile([0, 0, 1.0], (len(fpos), 1))

# --- glasses: thin round dark-grey metal frames --------------------------------------------

for side in (1, -1):
    cx, cy, R = side * (EYE_X + 0.001), EYE_Y - 0.001, 0.0235
    a = np.linspace(0, 2 * math.pi, 49)
    zr = surface_z(np.array([[cx + side * 0.012, cy], [cx, cy]])).max() + 0.011
    ring = np.stack([cx + R * np.cos(a), cy + R * 0.94 * np.sin(a), zr - 0.004 * (side * np.cos(a) > 0) * np.cos(a) ** 2], 1)
    pos, tris, uv = tube(ring, 0.0013, 0.0016, (0, 0, 1), segs=8, caps=(False, False))
    add("glasses", pos, tris, uv, solo(len(pos), "head"))
    # temple arm back over the ear (hidden by hair)
    ox = cx + side * R
    path = np.array([[ox, cy + 0.004, zr - 0.004], [side * 0.090, cy + 0.006, 0.040], [side * 0.103, cy + 0.006, 0.0], [side * 0.104, cy - 0.004, -0.045]])
    pos, tris, uv = tube(path, 0.0012, 0.0016, (0, 1, 0), segs=6)
    add("glasses", pos, tris, uv, solo(len(pos), "head"))
zb = surface_z(np.array([[0.0, EYE_Y + 0.002]]))[0] + 0.010
b = np.linspace(0, 1, 9)
bridge = np.stack([-0.0145 + 0.029 * b, EYE_Y + 0.004 + 0.003 * np.sin(b * math.pi), np.full_like(b, zb)], 1)
pos, tris, uv = tube(bridge, 0.0012, 0.0012, (0, 0, 1), segs=6)
add("glasses", pos, tris, uv, solo(len(pos), "head"))

# --- hair: a cap, a back curtain, and ribbon strands from a near-middle parting -----------------

HC = HEAD_C + (0, 0.004, -0.004)
HR = np.array([0.114, 0.131, 0.118])
PART_X = 0.008


def on_shell(d, scale=1.0):
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return HC + d * HR * scale


def slerp(a, b, t):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    om = math.acos(np.clip(a @ b, -1, 1))
    if om < 1e-6:
        return np.outer(1 - t, a) + np.outer(t, b)
    return (np.sin((1 - t)[:, None] * om) * a + np.sin(t[:, None] * om) * b) / math.sin(om)


def strand(start_dir, end_dir, end_y, width, thick, scale=1.0, flare=0.03, curl=0.008, n_top=12, n_drop=10, taper=0.33):
    """Over the shell from start_dir to end_dir, then (if end_y is below the end point) hang straight down."""
    t = np.linspace(0, 1, n_top)
    d = slerp(np.asarray(start_dir, float), np.asarray(end_dir, float), t)
    bulge = 1 + 0.03 * np.sin(t * math.pi) + rng.normal(0, 0.006)
    top = HC + d * HR * scale * bulge[:, None]
    pts = [top]
    if end_y < top[-1, 1] - 0.01:
        s = np.linspace(0, 1, n_drop + 1)[1:]
        horiz = top[-1] - HC
        horiz[1] = 0
        hn = horiz / max(np.linalg.norm(horiz), 1e-9)
        drop = top[-1] + np.outer(s, (0, end_y - top[-1, 1], 0))
        drop += np.outer(flare * s ** 1.5 - curl * s ** 6, hn)
        drop += np.outer(np.sin(s * math.pi) * rng.normal(0, 0.004), (1, 0, 0))
        pts.append(drop)
    pts = np.vstack(pts)
    n = len(pts)
    v = np.linspace(0, 1, n)
    w = width * np.clip((1 - v) / taper, 0, 1) ** 0.7 + 0.0008
    th = thick * (1 - 0.7 * v) + 0.0006
    normal = pts - HC
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    pos, tris, uv = tube(pts, th, w, normal, segs=8, caps=(True, False))
    return pos, tris, uv


def sph(theta_deg, phi_deg):
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    return np.array([math.sin(th) * math.sin(ph), math.cos(th), math.sin(th) * math.cos(ph)])


def add_hair(pos, tris, uv):
    add("hair", pos, tris, uv, solo(len(pos), "head"))


# cap: the shell minus the face opening
def cap_deform(p, d):
    return p


cpos, ctris, cuv, cdir = ellipsoid(HC, HR * 0.95, rings=36, segs=56)
theta = np.degrees(np.arccos(np.clip(cdir[:, 1], -1, 1)))
phi = np.degrees(np.abs(np.arctan2(cdir[:, 0], cdir[:, 2])))
limit = 44 + (118 - 44) * smooth(60, 110, phi)
keep = theta <= limit
tk = keep[ctris].all(1)
add_hair(cpos, ctris[tk], cuv * (1, 0.5))

# curtain behind and beside the face down to shoulder length
phis = np.radians(np.linspace(80, 280, 40))
ys = np.linspace(HC[1] + 0.01, 1.40, 12)
P, Ph = np.meshgrid(ys, phis, indexing="ij")
rx, rz = HR[0] * 0.93, HR[2] * 0.93
flare = 1 + 0.03 * ((HC[1] - P) / (HC[1] - 1.40)).clip(0, 1) ** 1.5
pos = np.stack([rx * np.sin(Ph) * flare, P, HC[2] + rz * np.cos(Ph) * flare], -1).reshape(-1, 3)
pos[:, 1] += 0.012 * np.cos(np.arange(len(pos)) * 1.7) * (pos[:, 1] < 1.41)
nphi = len(phis)
tris = []
for i in range(len(ys) - 1):
    for j in range(nphi - 1):
        a = i * nphi + j
        tris += [(a, a + 1, a + nphi), (a + 1, a + nphi + 1, a + nphi)]
tris = np.array(tris)
tris = orient(pos, tris, pos[tris].mean(1) - np.array([0, 0, HC[2]]))
uv = np.stack([np.zeros(len(pos)), 0.3 + 0.7 * (HC[1] - pos[:, 1]) / (HC[1] - 1.40)], 1)
add_hair(pos, tris, uv)

part_front = np.array([PART_X / HR[0], math.cos(math.radians(38)), math.sin(math.radians(38))])
part_back = np.array([PART_X / HR[0], math.cos(math.radians(-25)), math.sin(math.radians(-25))])

# side and back strands, both sides, two layers
for layer, (scale, width, n) in enumerate(((0.985, 0.040, 16), (1.03, 0.034, 15))):
    for side in (1, -1):
        for i in range(n):
            f = i / (n - 1)
            phi = side * (82 + 98 * f + rng.normal(0, 3))
            start = slerp(part_front, part_back, np.array([min(1, f * 1.15)]))[0] if layer else sph(4 + rng.normal(0, 2), side * 90)
            start = start.copy()
            start[0] = side * abs(start[0]) + side * 0.02
            end_y = 1.405 - 0.012 * f + rng.normal(0, 0.008)
            if abs(phi) < 75:
                end_y += 0.02  # face-framing locks a touch shorter
            add_hair(*strand(start, sph(92, phi), end_y, width, 0.011, scale=scale, flare=0.012 + 0.012 * layer, curl=0.018))

# crown strands over the back
for i in range(12):
    phi = 150 + 60 * i / 11 + rng.normal(0, 3)
    add_hair(*strand(sph(28, 180 + rng.normal(0, 10)), sph(92, phi), 1.395 + rng.normal(0, 0.008), 0.040, 0.011, scale=1.02, flare=0.02, curl=0.018))

# no fringe: front hair is swept back from a near-middle parting over the top to the sides
for side in (1, -1):
    for i in range(9):
        f = i / 8
        start = sph(44 - 4 * f + rng.normal(0, 2), side * (6 + 16 * f))
        phi = side * (84 + 36 * f + rng.normal(0, 4))
        end_y = 1.415 - 0.02 * f + rng.normal(0, 0.006)
        add_hair(*strand(start, sph(95, phi), end_y, 0.040, 0.013, scale=1.04 + 0.01 * rng.normal(), flare=0.006, curl=0.016, n_top=16))
    # a few loose strands slipping forward over the temple, down beside the cheek
    for i in range(3):
        start = sph(40 + 3 * i, side * (10 + 6 * i))
        add_hair(*strand(start, sph(86, side * (66 + 6 * i)), 1.50 - 0.025 * i + rng.normal(0, 0.006), 0.020, 0.008, scale=1.0, flare=0.0, curl=0.006, taper=0.5))

# Anime proportions: grow everything rigidly attached to the head about a point under the chin.
HEAD_SCALE, HEAD_PIVOT = 1.08, np.array([0, 1.47, 0.0])
for plist in parts.values():
    for p in plist:
        if (p.w[:, BONE["head"]] == 1).all():
            p.pos = HEAD_PIVOT + (p.pos - HEAD_PIVOT) * HEAD_SCALE
            p.morphs = {k: v * HEAD_SCALE for k, v in p.morphs.items()}

# --- textures ----------------------------------------------------------------------------


def png(img):
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def shirt_texture():
    W_, H_ = 2048, 1024
    img = Image.new("RGB", (W_, H_), (246, 246, 244))
    # artwork drawn at a square scale (px per metre), then fitted into the chest region
    PPM = 4000
    gw, gh = 0.23, 0.15
    art = Image.new("RGB", (int(gw * PPM), int(gh * PPM)), (246, 246, 244))
    d = ImageDraw.Draw(art)
    G = (24, 136, 76)
    LW = 8
    base = art.height - 40
    d.line([(10, base), (art.width - 10, base)], fill=G, width=LW)
    tw, th = 64, 86  # one mahjong tile
    gap = 6
    sym = ["dot", "bam", "zhong", "dot2", "white", "bam3", "dot", "zhong", "bam", "dot2"]

    def tile(x, y, kind):
        d.rounded_rectangle([x, y, x + tw, y + th], radius=12, outline=G, width=LW - 1)
        cx, cy = x + tw / 2, y + th / 2
        if kind == "dot":
            d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], outline=G, width=5)
            d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=G)
        elif kind == "dot2":
            for oy in (-17, 17):
                d.ellipse([cx - 12, cy + oy - 12, cx + 12, cy + oy + 12], outline=G, width=5)
        elif kind == "bam":
            for ox in (-12, 12):
                d.line([(cx + ox, cy - 24), (cx + ox, cy + 24)], fill=G, width=6)
                d.line([(cx + ox - 5, cy), (cx + ox + 5, cy)], fill=G, width=4)
        elif kind == "bam3":
            for ox, oy in ((0, -14), (-13, 14), (13, 14)):
                d.line([(cx + ox, cy + oy - 11), (cx + ox, cy + oy + 11)], fill=G, width=6)
        elif kind == "zhong":
            d.rectangle([cx - 18, cy - 10, cx + 18, cy + 8], outline=G, width=5)
            d.line([(cx, cy - 26), (cx, cy + 26)], fill=G, width=6)
        elif kind == "white":
            d.rectangle([cx - 16, cy - 24, cx + 16, cy + 24], outline=G, width=4)

    heights = [2, 3, 1, 4, 2, 5, 3, 1, 3, 2]
    x0 = 30
    for col, hcount in enumerate(heights):
        x = x0 + col * (tw + gap)
        for r in range(hcount):
            tile(x, base - (r + 1) * (th + gap) + gap // 2, sym[(col + r * 3) % len(sym)])
        top = base - hcount * (th + gap)
        if hcount == 5:  # the landmark: a spire on the tallest stack
            d.line([(x + tw / 2, top - 70), (x + tw / 2, top)], fill=G, width=LW)
            d.polygon([(x + 8, top), (x + tw - 8, top), (x + tw / 2, top - 34)], outline=G, width=LW - 2)
        elif hcount >= 3 and col % 2:
            d.line([(x + 14, top), (x + 14, top - 26)], fill=G, width=5)
    # a sun and two birds, line art
    d.ellipse([art.width - 150, 30, art.width - 70, 110], outline=G, width=LW)
    for bx, by in ((150, 60), (230, 95)):
        d.arc([bx - 22, by - 10, bx, by + 10], 200, 340, fill=G, width=5)
        d.arc([bx, by - 10, bx + 22, by + 10], 200, 340, fill=G, width=5)
    # place onto the chest: horizontal is arc-length fraction around the torso, vertical is height
    circ = 2 * math.pi * math.sqrt((0.104 ** 2 + 0.160 ** 2) / 2) * 1.04
    pw = int(W_ * gw / circ)
    y_top, y_bot = 1.262, 1.262 - gh
    v0 = lambda y: (1 - (y - SHIRT_V[0]) / (SHIRT_V[1] - SHIRT_V[0])) * H_
    ph = int(v0(y_bot) - v0(y_top))
    art = art.resize((pw, ph), Image.LANCZOS)
    img.paste(art, (W_ // 2 - pw // 2, int(v0(y_top))))
    return img


def hair_texture():
    W_, H_ = 16, 256
    v = np.linspace(0, 1, H_)[:, None]
    base = np.array([34, 34, 42], float)
    tip = np.array([22, 22, 28], float)
    col = base * (1 - v) + tip * v
    band = np.exp(-((v - 0.22) / 0.035) ** 2)
    col = col + band * np.array([52, 56, 70])
    arr = np.repeat(col[:, None, :], W_, 1).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr.reshape(H_, W_, 3))


def face_texture():
    img = Image.new("RGB", (4, 4), (255, 0, 255))
    for i, key in enumerate(PAL_KEYS):
        img.putpixel((i % 4, i // 4), FACE_PAL[key])
    return img


# --- glTF / VRM writer ----------------------------------------------------------------------


class GLB:
    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors = [], []

    def view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin += data
        self.views.append(v)
        return len(self.views) - 1

    def acc(self, arr, ctype, typ, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        a = {"bufferView": self.view(arr.tobytes(), target), "componentType": ctype, "count": len(arr), "type": typ}
        if minmax:
            a["min"] = arr.min(0).tolist()
            a["max"] = arr.max(0).tolist()
        self.accessors.append(a)
        return len(self.accessors) - 1


def vertex_normals(part):
    if getattr(part, "normals", None) is not None:
        return part.normals
    p, t = part.pos, part.tris
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    # weld normals across UV seams / coincident vertices
    key = np.round(p / 1e-5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    acc = np.zeros((inv.max() + 1, 3))
    np.add.at(acc, inv, n)
    n = acc[inv]
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), [0, 1, 0])
    return n


def top4(W):
    idx = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, idx, 1)
    w = np.where(w < 1e-4, 0, w)
    s = w.sum(1, keepdims=True)
    assert (s > 0).all(), "vertex without weights"
    w = w / s
    idx = np.where(w > 0, idx, 0)
    return idx.astype(np.uint16), w.astype(np.float32)


def merge(plist):
    pos, nrm, uv, W, tris, morphs = [], [], [], [], [], {}
    off = 0
    for p in plist:
        pos.append(p.pos); nrm.append(vertex_normals(p)); uv.append(p.uv); W.append(p.w)
        tris.append(p.tris + off)
        off += len(p.pos)
    return np.vstack(pos), np.vstack(nrm), np.vstack(uv), np.vstack(W), np.vstack(tris)


def lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).tolist()


def mtoon(name, base, shade, tex=None, outline=0.0, double=False, toony=0.9, shift=-0.05):
    m = {
        "name": name,
        "pbrMetallicRoughness": {"baseColorFactor": lin(base) + [1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0},
        "doubleSided": double,
        "extensions": {"VRMC_materials_mtoon": {
            "specVersion": "1.0", "transparentWithZWrite": False, "renderQueueOffsetNumber": 0,
            "shadeColorFactor": lin(shade), "shadingShiftFactor": shift, "shadingToonyFactor": toony,
            "giEqualizationFactor": 0.9, "matcapFactor": [0, 0, 0], "parametricRimColorFactor": [0, 0, 0],
            "parametricRimFresnelPowerFactor": 5.0, "parametricRimLiftFactor": 0.0, "rimLightingMixFactor": 1.0,
            "outlineWidthMode": "worldCoordinates" if outline else "none", "outlineWidthFactor": outline,
            "outlineColorFactor": lin((40, 34, 36)), "outlineLightingMixFactor": 1.0,
            "uvAnimationScrollXSpeedFactor": 0, "uvAnimationScrollYSpeedFactor": 0, "uvAnimationRotationSpeedFactor": 0,
        }},
    }
    if tex is not None:
        m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex}
        m["extensions"]["VRMC_materials_mtoon"]["shadeMultiplyTexture"] = {"index": tex}
    return m


def build():
    g = GLB()
    images, textures = [], []

    def texture(img, nearest=False):
        images.append({"name": f"tex{len(images)}", "mimeType": "image/png", "bufferView": g.view(png(img))})
        textures.append({"source": len(images) - 1, "sampler": 1 if nearest else 0})
        return len(textures) - 1

    t_shirt = texture(shirt_texture())
    t_hair = texture(hair_texture())
    t_face = texture(face_texture(), nearest=True)
    materials = [
        mtoon("Skin", (255, 228, 212), (236, 178, 160), outline=0.0012),
        mtoon("Shirt", (255, 255, 255), (196, 200, 214), tex=t_shirt, outline=0.0012),
        mtoon("Trousers", (40, 40, 46), (22, 22, 28), outline=0.0012),
        mtoon("Shoes", (204, 207, 212), (150, 154, 166), outline=0.0012),
        mtoon("Sole", (244, 244, 244), (200, 200, 210), outline=0.0010),
        mtoon("Hair", (255, 255, 255), (150, 150, 175), tex=t_hair, outline=0.0010, double=True),
        mtoon("Face", (255, 255, 255), (255, 255, 255), tex=t_face, double=True, toony=1.0, shift=-1.0),
        mtoon("Glasses", (70, 72, 78), (40, 42, 48), double=True),
    ]
    mat_index = {"skin": 0, "shirt": 1, "pants": 2, "shoes": 3, "sole": 4, "hair": 5, "face": 6, "glasses": 7}

    nodes = []
    for name, parent, p in BONES:
        local = BPOS[name] - (BPOS[parent] if parent else 0)
        nodes.append({"name": name, "translation": [float(x) for x in local]})
    for name, parent, _ in BONES:
        if parent:
            nodes[BONE[parent]].setdefault("children", []).append(BONE[name])
    ibm = np.zeros((NB, 16), np.float32)
    for i, (name, _, _) in enumerate(BONES):
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = -BPOS[name]
        ibm[i] = m.T.ravel()  # column-major
    skin = {"joints": list(range(NB)), "skeleton": 0, "inverseBindMatrices": g.acc(ibm, 5126, "MAT4")}

    meshes = []

    def primitive(plist, mat, with_morphs=False):
        pos, nrm, uv, W, tris = merge(plist)
        j, w = top4(W)
        attrs = {
            "POSITION": g.acc(pos.astype(np.float32), 5126, "VEC3", 34962, True),
            "NORMAL": g.acc(nrm.astype(np.float32), 5126, "VEC3", 34962),
            "TEXCOORD_0": g.acc(uv.astype(np.float32), 5126, "VEC2", 34962),
            "JOINTS_0": g.acc(j, 5123, "VEC4", 34962),
            "WEIGHTS_0": g.acc(w, 5126, "VEC4", 34962),
        }
        prim = {"attributes": attrs, "indices": g.acc(tris.astype(np.uint32).ravel(), 5125, "SCALAR", 34963), "material": mat_index[mat]}
        if with_morphs:
            prim["targets"] = [{"POSITION": g.acc(np.vstack([p.morphs[m] for p in plist]).astype(np.float32), 5126, "VEC3", None, True)} for m in MORPHS]
        return prim, len(pos)

    counts = {}
    mesh_defs = [("Body", ["skin", "shirt", "pants", "shoes", "sole"]), ("Hair", ["hair", "glasses"]), ("Face", ["face"])]
    for mname, mats in mesh_defs:
        prims = []
        for mat in mats:
            prim, nv = primitive(parts[mat], mat, with_morphs=(mname == "Face"))
            prims.append(prim)
            counts[mat] = nv
        mesh = {"name": mname, "primitives": prims}
        if mname == "Face":
            mesh["extras"] = {"targetNames": MORPHS}
        meshes.append(mesh)
        nodes.append({"name": mname, "mesh": len(meshes) - 1, "skin": 0})
    face_node = len(nodes) - 1

    def expr(*binds, block=False):
        return {
            "morphTargetBinds": [{"node": face_node, "index": MORPHS.index(b), "weight": 1.0} for b in binds],
            "isBinary": False, "overrideBlink": "block" if block else "none", "overrideLookAt": "none", "overrideMouth": "none",
        }

    preset = {
        "happy": expr("happy", block=True), "sad": expr("sad"), "relaxed": expr("relaxed", block=True),
        "surprised": expr("surprised", block=True), "aa": expr("aa"), "ih": expr("ih"), "ou": expr("ou"),
        "ee": expr("ee"), "oh": expr("oh"), "blink": expr("blinkLeft", "blinkRight"),
        "blinkLeft": expr("blinkLeft"), "blinkRight": expr("blinkRight"),
    }
    vrm = {
        "specVersion": "1.0",
        "meta": {
            "name": "Buddy", "version": "1.0", "authors": ["body-buddy contributors"],
            "copyrightInformation": "Generated by avatar/build_avatar.py",
            "references": ["https://github.com/petercechung/body-buddy"],
            "licenseUrl": "https://vrm.dev/licenses/1.0/", "avatarPermission": "everyone",
            "allowExcessivelyViolentUsage": False, "allowExcessivelySexualUsage": False,
            "commercialUsage": "personalNonProfit", "allowPoliticalOrReligiousUsage": False,
            "allowAntisocialOrHateUsage": False, "creditNotation": "required", "allowRedistribution": True,
            "modification": "allowModification",
        },
        "humanoid": {"humanBones": {name: {"node": BONE[name]} for name, _, _ in BONES}},
        "expressions": {"preset": preset},
        "lookAt": {"offsetFromHeadBone": [0, 0.09, 0.08], "type": "expression"},
    }
    gltf = {
        "asset": {"version": "2.0", "generator": "body-buddy avatar/build_avatar.py"},
        "extensionsUsed": ["VRMC_vrm", "VRMC_materials_mtoon"],
        "extensions": {"VRMC_vrm": vrm},
        "scene": 0,
        "scenes": [{"nodes": [0, NB, NB + 1, NB + 2]}],
        "nodes": nodes, "meshes": meshes, "skins": [skin], "materials": materials,
        "textures": textures, "images": images,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071},
                     {"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
        "accessors": g.accessors, "bufferViews": g.views, "buffers": [{"byteLength": len(g.bin)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    g.bin += b"\0" * (-len(g.bin) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(g.bin))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(g.bin), 0x004E4942) + bytes(g.bin)
    OUT.write_bytes(out)
    print(f"{OUT}: {len(out) / 1e6:.2f} MB, vertices {sum(counts.values())} ({counts})", file=sys.stderr)


if __name__ == "__main__":
    build()
