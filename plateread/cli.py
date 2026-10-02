"""Command line interface."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .formats import BUILTIN_FORMATS, FormatSet, load_formats
from .lookup import CommandLookup, dominant_colour, verify_candidates
from .ocr import default_engines, tesseract_available
from .pipeline import read_image
from .quality import verdict_sentence
from .synth import (CAMERA_PRESETS, SCENARIOS, make_camera_view,
                    make_sample, make_scenario, random_text)
from .util import imread_any, imwrite_any, resize_max, to_bgr, to_gray


BAR = "-" * 66


def _bar(v: float, width: int = 24) -> str:
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "#" * n + "." * (width - n)


def _print_result(r, index: int, verbose: bool) -> None:
    print(f"\n[{index}] {r.text}")
    print(f"    confidence {_bar(r.confidence)} {r.confidence:5.1%}")
    print(f"    agreement  {_bar(r.agreement)} {r.agreement:5.1%}  "
          f"({r.n_hypotheses} independent reads)")
    print(f"    image      {_bar(r.quality.score)} {r.quality.score:5.1%}  "
          f"[{r.quality.verdict.upper()}]")

    q = r.quality
    if q.char_height_measured:
        print(f"    detail     char height ~{q.char_height_px:.0f}px, stroke "
              f"~{q.stroke_px:.1f}px, dynamic range {q.dynamic_range:.0f}/255")
    else:
        # No characters were found to measure, so any height printed here
        # would be an assumption about a crop that may hold no text at all -
        # and printing a confident number beside a refusal undercuts it.
        print(f"    detail     character height not measurable, "
              f"dynamic range {q.dynamic_range:.0f}/255")
    if q.blur_confident:
        print(f"               motion blur ~{q.blur_length:.0f}px @ {q.blur_angle:.0f} deg")
    if r.rotation:
        print(f"               image was read after rotating {r.rotation} deg")
    if r.plate_format:
        print(f"    format     matches {r.plate_format}  [{r.format_region}]")
    if r.raw_text and r.raw_text != r.text:
        print(f"               grammar corrected {r.raw_text} -> {r.text} "
              f"(--no-formats to see the raw reading)")

    print("    per character:")
    for i, pc in enumerate(r.per_char):
        line = f"      {i + 1}. {pc['char']}   {pc['confidence']:5.1%}"
        if pc["runner_up"]:
            line += f"   next: {pc['runner_up']} ({pc['runner_up_confidence']:.0%})"
        if pc["confusable_with"]:
            line += f"   [confusable with {pc['confusable_with']}]"
        if pc["confidence"] < 0.55:
            line += "   <-- weak"
        print(line)

    if len(r.alternatives) > 1:
        alts = ", ".join(f"{t} ({p:.0%})" for t, p in r.alternatives[1:6])
        print(f"    other readings: {alts}")

    if q.notes:
        print("    image notes:")
        for n in q.notes:
            print(f"      - {n}")

    if verbose:
        print("    hypotheses:")
        for h in sorted(r.hypotheses, key=lambda h: -h.mean_conf)[:14]:
            print(f"      {h.text:<10} {h.mean_conf:5.1%}  {h.engine}/{h.variant}")


def _verdict_banner(r) -> None:
    print(BAR)
    if not r.quality.readable:
        print("!! " + verdict_sentence(r.quality))
        print("!! Reported characters below are the best fit to insufficient")
        print("!! evidence. Do not treat them as an identification.")
    elif not r.trustworthy:
        print(" * " + verdict_sentence(r.quality))
        print(" * Low agreement between processing variants - check the weak")
        print(" * characters and the alternate readings before relying on this.")
    else:
        print("Read is consistent across processing variants and the source")
        print("image carries enough detail to support it.")
    print(BAR)


def cmd_read(args: argparse.Namespace) -> int:
    path = Path(args.image)
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2

    img = imread_any(path)
    if args.max_side:
        img = resize_max(img, args.max_side)

    engines = default_engines(args.font, model_path=args.model,
                              use_model=not args.no_model)
    formats = None
    if not args.no_formats:
        try:
            table = load_formats(args.formats) if args.formats else None
            formats = FormatSet(table, args.region)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    print(f"plateread {__version__}  |  engines: {', '.join(sorted(engines))}")
    print(f"image: {path.name}  {img.shape[1]}x{img.shape[0]}")

    results = read_image(img, aggressive=args.aggressive, pattern=args.pattern,
                         max_candidates=args.candidates, engines=engines,
                         collect_stages=bool(args.debug_dir), formats=formats,
                         descreen={'auto': None, 'on': True,
                                   'off': False}[args.descreen])
    if not results:
        print("\nNo plate-like region could be read.")
        print("Try --aggressive for deconvolution and more restorations,")
        print("--candidates 8 to widen the search, or crop closer to the plate")
        print("by hand and run it again on the crop.")
        return 1

    if args.json:
        payload = [{
            "text": r.text,
            "confidence": round(r.confidence, 4),
            "agreement": round(r.agreement, 4),
            "trustworthy": r.trustworthy,
            "quality": {
                "verdict": r.quality.verdict,
                "char_height_measured": r.quality.char_height_measured,
                "score": round(r.quality.score, 4),
                "char_height_px": round(r.quality.char_height_px, 1),
                "stroke_px": round(r.quality.stroke_px, 2),
                "blur_length_px": round(r.quality.blur_length, 1),
                "blur_angle_deg": round(r.quality.blur_angle, 1),
                "blur_confident": r.quality.blur_confident,
                "dynamic_range": round(r.quality.dynamic_range, 1),
                "notes": r.quality.notes,
            },
            "per_char": r.per_char,
            "alternatives": r.alternatives,
            "n_hypotheses": r.n_hypotheses,
            "bbox": [round(v, 1) for v in r.candidate.bbox],
            "detector": r.candidate.source,
            "rotation_deg": r.rotation,
            "plate_format": r.plate_format,
            "format_region": r.format_region,
            "raw_text": r.raw_text,
        } for r in results[:args.top]]
        print(json.dumps(payload, indent=2))
        return 0

    for i, r in enumerate(results[:args.top], 1):
        _print_result(r, i, args.verbose)

    if args.verify_cmd:
        _verify(results[0], img, args)

    _verdict_banner(results[0])

    if args.debug_dir:
        out = Path(args.debug_dir)
        best = results[0]
        # quads are in the coordinate frame of the pass that produced them, so
        # draw on the image turned the same way
        overlay = to_bgr(img).copy()
        if best.rotation:
            overlay = np.ascontiguousarray(np.rot90(overlay, best.rotation // 90))
        for r in (x for x in results[:args.top] if x.rotation == best.rotation):
            quad = r.candidate.quad.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [quad], True, (0, 220, 255), 2)
            x, y = int(r.candidate.quad[0][0]), int(r.candidate.quad[0][1])
            cv2.putText(overlay, r.text, (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        imwrite_any(out / "00_detection.png", overlay)
        for name, im in best.stages.items():
            safe = name.replace("|", "_").replace(":", "_")
            imwrite_any(out / f"{safe}.png", im)
        print(f"\nDebug images written to {out.resolve()}")

    return 0 if results[0].trustworthy else 1


def _verify(r, img, args) -> None:
    """Cross-check the shortlist against a vehicle source the user supplies."""
    try:
        lookup = CommandLookup(args.verify_cmd, timeout=args.verify_timeout)
    except ValueError as exc:
        print(f"\n    verification skipped: {exc}")
        return

    colour = None
    if r.rotation == 0:
        colour = dominant_colour(img, r.candidate.quad)

    shortlist = r.alternatives[:args.verify_top] or [(r.text, r.confidence)]
    print(f"\n    verifying {len(shortlist)} candidate reading(s) against "
          f"the supplied source")
    if colour:
        print(f"    vehicle in the photo looks {colour}")

    checks = verify_candidates(shortlist, lookup, colour, limit=args.verify_top)
    any_hit = False
    for v in checks:
        if v.record is None:
            print(f"      {v.plate:<10} no record")
            continue
        any_hit = True
        line = f"      {v.plate:<10} {v.record.describe()}"
        if v.colour_match is True:
            line += "   [colour matches the photo]"
        elif v.colour_match is False:
            line += "   [colour does NOT match the photo]"
        print(line)

    if not any_hit:
        print("    nothing resolved - the reading stands on the image alone.")
        return

    best = checks[0]
    if best.record is not None and best.plate != r.text:
        print(f"    verification favours {best.plate} over the image-only "
              f"reading {r.text}")
    elif best.record is not None:
        print(f"    verification is consistent with {r.text}")


def cmd_synth(args: argparse.Namespace) -> int:
    text = args.text or random_text()
    img = make_sample(text=text, style=args.style, rot=args.rotate,
                      persp=args.perspective, motion=args.motion,
                      motion_angle=args.motion_angle, blur=args.blur,
                      scale=args.scale, noise=args.noise, quality=args.jpeg,
                      scene=args.scene, seed=args.seed, moire=args.moire)
    imwrite_any(args.out, img)
    print(f"wrote {args.out}  ({img.shape[1]}x{img.shape[0]})  ground truth: {text}")
    return 0


def cmd_scenario(args: argparse.Namespace) -> int:
    """Render realistic test images with known ground truth."""
    names = sorted(SCENARIOS) if args.name in (None, "all") else [args.name]
    outdir = Path(args.out_dir)
    made = []
    for name in names:
        try:
            img, truth = make_scenario(name, args.text, args.seed)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        path = outdir / f"{name}.png"
        imwrite_any(path, img)
        made.append((name, path, truth, img))
        print(f"{name:<20} {img.shape[1]}x{img.shape[0]:<6} truth {truth:<9} "
              f"{SCENARIOS[name]['note']}")
    print(f"\nwrote {len(made)} image(s) to {outdir.resolve()}")

    if not args.read:
        print("read them back with:")
        print(f"  plateread read {outdir}/<name>.png --aggressive")
        return 0

    engines = default_engines()
    formats = FormatSet()
    print()
    print(f"{'scenario':<20} {'truth':<9} {'read':<11} {'conf':>5} {'image':>13}  ok")
    print(BAR)
    correct = 0
    for name, path, truth, img in made:
        results = read_image(img, aggressive=True, engines=engines,
                             max_candidates=4, formats=formats)
        if results:
            r = results[0]
            got, conf, verdict = r.text, r.confidence, r.quality.verdict
        else:
            got, conf, verdict = "-", 0.0, "none"
        hit = got == truth
        correct += hit
        flag = "yes" if hit else ("(refused)" if verdict == "insufficient" else "NO")
        print(f"{name:<20} {truth:<9} {got:<11} {conf:5.0%} {verdict:>13}  {flag}")
    print(BAR)
    print(f"{correct}/{len(made)} exact")
    print("A row marked (refused) is a correct outcome when the plate is genuinely")
    print("too small to read - the tool declining beats it inventing a plate.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the sequence recogniser."""
    from .model import torch_available
    if not torch_available():
        print("error: PyTorch is not installed. Install it with:", file=sys.stderr)
        print("  python -m pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu", file=sys.stderr)
        return 2
    from .train import TrainConfig, train

    if args.real_dir:
        from .dataset import load_real
        try:
            pairs = load_real(args.real_dir)
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not pairs:
            print(f"error: no labelled images found in {args.real_dir}.",
                  file=sys.stderr)
            print("Provide labels.csv rows of 'filename,PLATE', or name each "
                  "file after its plate (ABC1234.jpg).", file=sys.stderr)
            return 2
        print(f"found {len(pairs)} labelled crop(s) in {args.real_dir}")

    cfg = TrainConfig(steps=args.steps, batch_size=args.batch,
                      lr=args.lr, real_dir=args.real_dir,
                      real_ratio=args.real_ratio, seed=args.seed,
                      out=args.out, val_every=args.val_every)
    best = train(cfg)
    if best.get("val") == "synthetic":
        print()
        print("Validated on synthetic data only, so this number says how well "
              "the model")
        print("learned the generator - not how it will do on your photographs. "
              "Collect")
        print("labelled real crops and pass --real-dir to get a number that "
              "means something.")
    return 0


def cmd_labels(args: argparse.Namespace) -> int:
    """Turn photographs into labelled training crops."""
    from .detect import detect_plates
    from .rectify import rectify, trim_border
    src = Path(args.image_dir)
    if not src.is_dir():
        print(f"error: not a directory: {src}", file=sys.stderr)
        return 2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [p for p in sorted(src.iterdir()) if p.suffix.lower() in exts]
    if not files:
        print(f"error: no images in {src}", file=sys.stderr)
        return 2

    n = 0
    for path in files:
        try:
            img = imread_any(path)
        except OSError:
            continue
        gray = to_gray(img)
        for i, cand in enumerate(detect_plates(img, args.candidates)[:args.candidates]):
            try:
                crop = rectify(gray, cand.quad, target_h=96)
                crop, _ = trim_border(crop)
            except Exception:
                continue
            if crop.size == 0 or crop.shape[0] < 16:
                continue
            imwrite_any(out / f"{path.stem}_c{i}.png", crop)
            n += 1
    print(f"wrote {n} candidate crop(s) to {out.resolve()}")
    print()
    print("Now label them: keep the crops that really are plates, delete the")
    print("rest, and either rename each file after its plate (ABC1234.png) or")
    print("write a labels.csv with 'filename,PLATE' rows. Then:")
    print(f"  plateread train --real-dir {out} --steps 4000")
    return 0


def cmd_cameras(args: argparse.Namespace) -> int:
    """Render and read a plate from each standard camera position."""
    names = sorted(CAMERA_PRESETS) if args.camera in (None, "all") else [args.camera]
    out = Path(args.out_dir)
    # A vehicle travels along the road, so in the image the smear runs along
    # the direction the road projects to - which for an angled camera is not
    # horizontal. Approximating it by the camera's roll keeps the blur
    # direction consistent with the geometry instead of arbitrary.
    degrade = dict(blur=args.blur, noise=args.noise, quality=args.jpeg,
                   motion=args.motion, scale=args.scale)

    made = []
    for name in names:
        d = dict(degrade)
        d["motion_angle"] = (args.motion_angle if args.motion_angle is not None
                             else CAMERA_PRESETS[name]["roll"])
        try:
            img, truth = make_camera_view(name, text=args.text, degrade=d,
                                          plate_frac=args.plate_frac, seed=args.seed)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        imwrite_any(out / f"{name}.png", img)
        made.append((name, truth, img))
        cam = CAMERA_PRESETS[name]
        print(f"{name:<15} yaw {cam['yaw']:>3}  pitch {cam['pitch']:>3}  "
              f"{cam['distance']:>5.1f}m  {img.shape[1]}x{img.shape[0]:<5} "
              f"{cam['note']}")
    print(f"\nwrote {len(made)} image(s) to {out.resolve()}")
    if not args.read:
        return 0

    engines = default_engines(use_model=not args.no_model)
    try:
        formats = FormatSet(None, args.region)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print()
    print(f"{'camera':<15} {'truth':<9} {'read':<11} {'conf':>5} {'image':>13}  ok")
    print(BAR)
    correct = 0
    for name, truth, img in made:
        results = read_image(img, aggressive=True, engines=engines,
                             max_candidates=4, formats=formats)
        if results:
            r = results[0]
            got, conf, verdict = r.text, r.confidence, r.quality.verdict
        else:
            got, conf, verdict = "-", 0.0, "none"
        hit = got == truth
        correct += hit
        flag = "yes" if hit else ("(refused)" if verdict == "insufficient" else "NO")
        print(f"{name:<15} {truth:<9} {got:<11} {conf:5.0%} {verdict:>13}  {flag}")
    print(BAR)
    print(f"{correct}/{len(made)} exact")
    return 0


def cmd_limits(args):
    """Find where each degradation stops being survivable."""
    from .limits import DEFAULT_SWEEPS, sweep_axis
    axes = list(DEFAULT_SWEEPS) if args.axis in (None, "all") else [args.axis]
    if any(a not in DEFAULT_SWEEPS for a in axes):
        print(f"error: unknown axis; known: {', '.join(DEFAULT_SWEEPS)}",
              file=sys.stderr)
        return 2

    engines = default_engines(use_model=not args.no_model)
    formats = FormatSet()
    print(f"plateread {__version__}  |  engines: {', '.join(sorted(engines))}")
    print(f"{args.samples} samples per point, exact-match threshold "
          f"{args.threshold:.0%}")

    results = []
    for axis in axes:
        print()
        print(f"{axis:<10} {'value':>8} {'exact':>7} {'chars':>7} "
              f"{'refused':>8} {'BLUFF':>7}")
        print(BAR)

        def show(a, v, ex, ch, rf, bl):
            print(f"{'':<10} {v:>8} {ex:>7.0%} {ch:>7.0%} {rf:>8.0%} {bl:>7.0%}")

        r = sweep_axis(axis, samples=args.samples, threshold=args.threshold,
                       engines=engines, formats=formats,
                       aggressive=not args.fast, progress=show)
        results.append(r)
        print(BAR)
        if r.limit is None:
            print(f"  fails even at the mildest setting tested")
        else:
            print(f"  operating limit: {axis} = {r.limit}")
        if r.note:
            print(f"  for comparison, {r.note}")

    print()
    print(BAR)
    print("BLUFF is the column that matters when aiming at hard images: a")
    print("wrong plate the tool did NOT flag. A plain failure is survivable")
    print("because the caller knows to distrust it; a confident wrong answer")
    print("is not. Reading further into the hard band is only worth having if")
    print("that column stays near zero.")
    print()
    print("Where the tool gives up before the information does, the gap is")
    print("headroom. Where a run is mostly 'refused', the tool is declining")
    print("rather than guessing - which is the intended behaviour past the")
    print("point that the pixels stop carrying the answer.")
    return 0


def cmd_list_formats(args: argparse.Namespace) -> int:
    try:
        fs = FormatSet(None, args.region)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{'region':<10} {'mask':<12} {'example':<12} name")
    print(BAR)
    for f in fs.formats:
        print(f"{f.region:<10} {f.mask:<12} {f.example:<12} {f.name}")
    print(BAR)
    print("L = letter, D = digit, A = either.")
    print("Representative series only - not exhaustive, and specialist plates")
    print("(vanity, government, trade) will not match. Supply your own with --formats.")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Measure accuracy against known ground truth across a difficulty ramp."""
    levels = [
        ("clean",            dict()),
        ("rotated 12 deg",   dict(rot=12)),
        ("angled view",      dict(persp=0.35)),
        ("angled + rotated", dict(persp=0.30, rot=-14)),
        ("soft focus",       dict(blur=1.6)),
        ("motion blur",      dict(motion=9, motion_angle=0)),
        ("motion + angle",   dict(motion=8, motion_angle=20, persp=0.22, rot=8)),
        ("small (0.35x)",    dict(scale=0.35)),
        ("small + noisy",    dict(scale=0.40, noise=9, quality=45)),
        ("photo of a screen",dict(moire=0.40, scale=0.55)),
        ("screen + blur",    dict(moire=0.35, scale=0.5, blur=1.0)),
        ("heavy defocus",    dict(blur=3.0)),
        ("severe motion",    dict(motion=18, motion_angle=0)),
        ("extreme tilt 35",  dict(rot=35)),
        ("extreme tilt -42", dict(rot=-42)),
        ("tilt + motion",    dict(rot=-28, motion=14, motion_angle=20)),
        ("the hard one",     dict(persp=0.35, rot=-16, motion=10, motion_angle=15,
                                  blur=1.0, scale=0.55, noise=7, quality=50)),
        ("beyond recovery",  dict(scale=0.13, blur=2.2, noise=12, quality=25)),
    ]
    rng = random.Random(args.seed)
    engines = default_engines()
    formats = None if args.no_formats else FormatSet()
    print(f"plateread {__version__} self-test  |  engines: {', '.join(sorted(engines))}")
    print(f"{args.samples} plate(s) per level, ground truth known\n")
    print(f"{'level':<18} {'exact':>7} {'chars':>7} {'conf':>6} {'verdict':>13}")
    print(BAR)

    overall_exact = overall_chars = overall_n = 0
    for name, kw in levels:
        exact = chars = total_chars = 0
        confs, verdicts = [], []
        for _ in range(args.samples):
            truth = random_text(rng=rng)
            img = make_sample(text=truth, scene=args.scene, seed=rng.randrange(1 << 30), **kw)
            res = read_image(img, aggressive=args.aggressive, engines=engines,
                             max_candidates=3, formats=formats)
            total_chars += len(truth)
            if res:
                got = res[0].text
                confs.append(res[0].confidence)
                verdicts.append(res[0].quality.verdict)
                if got == truth:
                    exact += 1
                chars += sum(1 for a, b in zip(truth, got) if a == b)
            else:
                confs.append(0.0)
                verdicts.append("none")
        v = max(set(verdicts), key=verdicts.count) if verdicts else "none"
        print(f"{name:<18} {exact}/{args.samples:<5} {chars / max(1, total_chars):6.0%} "
              f"{np.mean(confs):6.0%} {v:>13}")
        overall_exact += exact
        overall_chars += chars
        overall_n += total_chars

    print(BAR)
    n_total = args.samples * len(levels)
    print(f"{'TOTAL':<18} {overall_exact}/{n_total:<5} "
          f"{overall_chars / max(1, overall_n):6.0%}")
    print("\nNote: 'beyond recovery' is expected to fail. It is in the ramp to")
    print("confirm the tool reports low confidence instead of inventing a plate.")
    if not tesseract_available():
        print("\nTesseract is not installed; the built-in template engine ran alone.")
        print("Installing Tesseract adds a second opinion to the vote.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plateread",
        description="Recover a license plate from a rotated, angled or blurred image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  plateread read photo.jpg
  plateread read photo.jpg --aggressive --debug-dir out/
  plateread read plate.png --pattern "[A-Z]{3}[0-9]{4}"
  plateread synth --text ABC1234 --rotate 14 --motion 9 --out test.png
  plateread selftest --samples 5
""")
    p.add_argument("--version", action="version", version=f"plateread {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("read", help="read the plate(s) in an image")
    r.add_argument("image")
    r.add_argument("--aggressive", action="store_true",
                   help="try many more restorations, including deconvolution "
                        "(slower, and more prone to inventing detail)")
    r.add_argument("--pattern", help="regex the plate must match, e.g. "
                                     "'[A-Z]{3}[0-9]{4}' - strongly improves accuracy")
    r.add_argument("--candidates", type=int, default=4,
                   help="how many plate-like regions to consider (default 4)")
    r.add_argument("--top", type=int, default=1, help="how many results to print")
    r.add_argument("--max-side", type=int, default=1600,
                   help="downscale huge inputs to this longest side (0 to disable)")
    r.add_argument("--font", action="append",
                   help="extra TTF to build character templates from (repeatable)")
    r.add_argument("--debug-dir", help="write each stage of the winning pipeline as PNGs")
    r.add_argument("--region", help="restrict to one jurisdiction's plate formats, "
                                    "e.g. uk, us-ca, fr (see list-formats)")
    r.add_argument("--formats", help="JSON file of custom plate formats "
                                     "(masks of L/D/A) to use instead of the built-ins")
    r.add_argument("--no-formats", action="store_true",
                   help="do not constrain the reading to any plate grammar")
    r.add_argument("--model", help="path to a trained CRNN checkpoint "
                                   "(default models/crnn.pt if present)")
    r.add_argument("--no-model", action="store_true",
                   help="ignore the trained model even if one is present")
    r.add_argument("--descreen", choices=["auto", "on", "off"], default="auto",
                   help="remove screen/moire patterning from a photo of a "
                        "monitor (default auto: retried only when the "
                        "first read does not settle)")
    r.add_argument("--verify-cmd",
                   help="command to look a plate up in a vehicle source you are "
                        "authorised to query; must contain {plate} and print JSON. "
                        "Used only to break ties between candidate readings.")
    r.add_argument("--verify-top", type=int, default=4,
                   help="how many candidate readings to look up (default 4)")
    r.add_argument("--verify-timeout", type=float, default=15.0)
    r.add_argument("--json", action="store_true", help="machine-readable output")
    r.add_argument("-v", "--verbose", action="store_true",
                   help="list the individual hypotheses behind the vote")
    r.set_defaults(func=cmd_read)

    s = sub.add_parser("synth", help="generate a degraded test plate")
    s.add_argument("--text", help="plate text (random if omitted)")
    s.add_argument("--out", default="sample.png")
    s.add_argument("--style", default="us", choices=["us", "eu", "dark"])
    s.add_argument("--rotate", type=float, default=0.0, help="degrees")
    s.add_argument("--perspective", type=float, default=0.0, help="0..0.9")
    s.add_argument("--motion", type=float, default=0.0, help="blur length in px")
    s.add_argument("--motion-angle", type=float, default=0.0, help="degrees")
    s.add_argument("--blur", type=float, default=0.0, help="defocus sigma")
    s.add_argument("--scale", type=float, default=1.0, help="downscale factor")
    s.add_argument("--noise", type=float, default=0.0, help="gaussian sigma")
    s.add_argument("--moire", type=float, default=0.0,
                   help="simulate a photo of a screen (0..0.6)")
    s.add_argument("--jpeg", type=int, default=0, help="jpeg quality, 0 = none")
    s.add_argument("--scene", action="store_true", help="embed in a cluttered scene")
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_synth)

    sc = sub.add_parser("scenario", help="render realistic test images "
                                         "(vehicle, plate, real degradations)")
    sc.add_argument("--name", default="all",
                    help="scenario name, or 'all' (default). See --list")
    sc.add_argument("--text", help="override the plate text")
    sc.add_argument("--out-dir", default="samples")
    sc.add_argument("--seed", type=int, default=7)
    sc.add_argument("--read", action="store_true",
                    help="read each one back and check against ground truth")
    sc.set_defaults(func=cmd_scenario)

    tr = sub.add_parser("train", help="train the sequence recogniser")
    tr.add_argument("--steps", type=int, default=4000)
    tr.add_argument("--batch", type=int, default=48)
    tr.add_argument("--lr", type=float, default=2e-3)
    tr.add_argument("--real-dir", help="directory of labelled plate crops "
                                       "to mix in and validate on")
    tr.add_argument("--real-ratio", type=float, default=0.35,
                    help="share of each batch drawn from real data")
    tr.add_argument("--val-every", type=int, default=250)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", default="models/crnn.pt")
    tr.set_defaults(func=cmd_train)

    lb = sub.add_parser("labels", help="cut plate crops out of photos, "
                                       "ready for you to label")
    lb.add_argument("image_dir")
    lb.add_argument("--out-dir", default="crops")
    lb.add_argument("--candidates", type=int, default=2)
    lb.set_defaults(func=cmd_labels)

    cm = sub.add_parser("cameras", help="render a plate from real ALPR "
                                        "camera positions and read it back")
    cm.add_argument("--camera", default="all",
                    help="one preset, or 'all' (default)")
    cm.add_argument("--text", default="TXR4821")
    cm.add_argument("--out-dir", default="cameras")
    cm.add_argument("--plate-frac", type=float, default=0.22,
                    help="plate width as a fraction of the vehicle scene")
    cm.add_argument("--blur", type=float, default=0.6)
    cm.add_argument("--motion", type=float, default=0.0,
                    help="motion blur length in px (traffic is typically 5-20)")
    cm.add_argument("--motion-angle", type=float, default=None,
                    help="blur direction; defaults to the camera geometry")
    cm.add_argument("--noise", type=float, default=4.0)
    cm.add_argument("--jpeg", type=int, default=80)
    cm.add_argument("--scale", type=float, default=1.0)
    cm.add_argument("--seed", type=int, default=7)
    cm.add_argument("--read", action="store_true")
    cm.add_argument("--no-model", action="store_true")
    cm.add_argument("--region", help="restrict to one jurisdiction's formats")
    cm.set_defaults(func=cmd_cameras)

    lm = sub.add_parser("limits", help="find where each degradation breaks it")
    lm.add_argument("--axis", default="all",
                    help="scale, motion, blur, noise, quality, or all")
    lm.add_argument("--samples", type=int, default=6)
    lm.add_argument("--threshold", type=float, default=0.5)
    lm.add_argument("--fast", action="store_true",
                    help="skip --aggressive restorations (much quicker)")
    lm.add_argument("--no-model", action="store_true")
    lm.set_defaults(func=cmd_limits)

    lf = sub.add_parser("list-formats", help="show the built-in plate grammars")
    lf.add_argument("--region", help="only this region")
    lf.set_defaults(func=cmd_list_formats)

    t = sub.add_parser("selftest", help="accuracy across a difficulty ramp")
    t.add_argument("--samples", type=int, default=3)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--scene", action="store_true")
    t.add_argument("--aggressive", action="store_true")
    t.add_argument("--no-formats", action="store_true",
                   help="do not constrain readings to plate grammars")
    t.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
