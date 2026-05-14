"""
Static site generator for docs/index.html.

Reads generated/full_library/full_library_data.json produced by run_full_analysis.py
and regenerates docs/index.html with real data from the full 83-track library.

Run after analysis:
    python3 generate_site.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Auto-discover the latest analysis run so regeneration doesn't need manual path updates.
def _find_latest_json():
    base = Path("generated/full_library")
    candidates = sorted(base.glob("audio_analysis_*/reports/full_library_data.json"), reverse=True)
    return candidates[0] if candidates else base / "full_library_data.json"

JSON_PATH = _find_latest_json()
OUT_PATH  = Path("docs/index.html")

# --- Mood color palette (matches existing CSS vars) ---
MOOD_COLORS = {
    "spacey":       ("#9B8FD4", "#9B8FD4"),
    "organic":      ("#27AE60", "#5ecc8a"),
    "synthetic":    ("#4FD1C5", "#4FD1C5"),
    "oozy":         ("#607D8B", "#90AAB8"),
    "pensive":      ("#B0C4DE", "#B0C4DE"),
    "tense":        ("#F4C471", "#F4C471"),
    "exuberant":    ("#EC96B8", "#F2BDD0"),
    "glitchy":      ("#FF6B6B", "#FF9999"),
    "chaos":        ("#EC96B8", "#EC96B8"),
    "ethereal":     ("#81E6D9", "#B2F0EC"),
    "atmospheric":  ("#4FD1C5", "#81E6D9"),
    "crystalline":  ("#4FD1C5", "#81E6D9"),
    "warm":         ("#F4C471", "#F9D99A"),
    "melodic":      ("#27AE60", "#5ecc8a"),
    "driving":      ("#F4C471", "#F4C471"),
    "percussive":   ("#FF6B6B", "#FF9999"),
    "droning":      ("#607D8B", "#90AAB8"),
}
DEFAULT_MOOD_COLOR = ("#4a6070", "#607D8B")

# Content-type character tags worth surfacing on cards
CONTENT_CHAR_TAGS = {
    "spoken_word": ("#EC96B8", "#F2BDD0"),
    "guitar":      ("#E6B55E", "#F0CF8E"),
    "live_drums":  ("#FF6B6B", "#FF9999"),
    "laughter":    ("#EC96B8", "#F2BDD0"),
    "wide_stereo": ("#81E6D9", "#B2F0EC"),
}

TENSION_COLORS = {
    (0.0, 0.15): "#607D8B",
    (0.15, 0.30): "#4FD1C5",
    (0.30, 0.60): "#4FD1C5",
    (0.60, 1.01): "#F4C471",
}

def tension_color(v):
    for (lo, hi), c in TENSION_COLORS.items():
        if lo <= v < hi:
            return c
    return "#4FD1C5"

def slugify(name):
    stem = Path(name).stem if "." in name else name
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")

def fmt_duration(secs):
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def clean_name(filename):
    stem = Path(filename).stem
    # Remove album prefix (Artist___) if present
    if "___" in stem:
        stem = stem.split("___", 1)[1]
    # Remove leading track numbers like "01 ", "02 - "
    stem = re.sub(r"^\d+[\s\-_.]+", "", stem)
    return stem.strip()

def sparkline_svg(energy_values, grad_id, width=200, height=40):
    """SVG polyline of energy values, y-inverted so high energy = tall peak."""
    if not energy_values or len(energy_values) < 2:
        return ""
    mn, mx = min(energy_values), max(energy_values)
    rng = mx - mn or 1e-6
    margin = 4
    h = height - margin * 2
    w = width

    pts = []
    for i, v in enumerate(energy_values):
        x = i / (len(energy_values) - 1) * w
        y = margin + h * (1.0 - (v - mn) / rng)
        pts.append(f"{x:.1f},{y:.1f}")

    path = "M " + " L ".join(pts)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" aria-hidden="true">\n'
        f'<defs><linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#4FD1C5" stop-opacity="0.9"/>'
        f'<stop offset="100%" stop-color="#4FD1C5" stop-opacity="0.15"/>'
        f'</linearGradient></defs>\n'
        f'<path d="{path}" fill="none" stroke="#4FD1C5" stroke-width="1.5" '
        f'stroke-linejoin="round" opacity="0.7"/>\n'
        f'</svg>'
    )


def build_cluster_map(cluster_data):
    """Return dict: filename → list of filenames in the same cluster."""
    if not cluster_data:
        return {}
    track_to_cluster = {}
    cluster_tracks = {}
    for cname, cinfo in cluster_data.items():
        files = cinfo.get("track_files", [])
        cluster_tracks[cname] = files
        for f in files:
            track_to_cluster[f] = cname

    similar = {}
    for fname, cname in track_to_cluster.items():
        peers = [f for f in cluster_tracks[cname] if f != fname]
        similar[fname] = peers[:4]
    return similar


def render_track_card(track, phases_for_file, similar_files):
    filename = track.get("filename", "")
    name = clean_name(filename)
    slug = slugify(name)
    dur = fmt_duration(float(track.get("duration", 0)))

    # Parse content-type character tags from the track-level field
    char_tags_raw = track.get("character_tags", "") or ""
    if isinstance(char_tags_raw, list):
        char_tags_set = set(char_tags_raw)
    else:
        char_tags_set = {t.strip() for t in char_tags_raw.split(",") if t.strip()}
    present_content_tags = [t for t in CONTENT_CHAR_TAGS if t in char_tags_set]

    # Mood tags from phases
    mood_counts = Counter()
    energy_vals = []
    narrative_parts = []

    for ph in phases_for_file:
        moods = ph.get("mood_descriptors", [])
        for m in moods:
            mood_counts[m] += 1
        chars = ph.get("characteristics", {})
        e = chars.get("avg_energy", 0.0)
        energy_vals.append(float(e))

        # Simple narrative from first few phases
        if len(narrative_parts) < 4 and moods:
            t = ph.get("start_time", 0)
            pt = ph.get("phase_type", "")
            t_str = fmt_duration(t)
            m0 = moods[0]
            if pt in ("intro", "exposition"):
                narrative_parts.append(f"Opens at {t_str} with a {m0} character")
            elif pt == "climax":
                narrative_parts.append(f"Peaks at {t_str} — {m0}")
            elif pt == "breakdown":
                narrative_parts.append(f"Breaks down at {t_str}: {m0}")
            else:
                narrative_parts.append(f"Settles into {m0} around {t_str}")

    narrative = ". ".join(narrative_parts[:3]) + ("." if narrative_parts else "")
    if not narrative.strip():
        narrative = f"An instrumental piece with {len(phases_for_file)} detected sections."

    # Tension = mean energy normalised to [0,1]
    max_possible_energy = 0.3
    if energy_vals:
        tension = min(1.0, sum(energy_vals) / len(energy_vals) / max_possible_energy)
    else:
        tension = 0.2
    tcol = tension_color(tension)

    # Tags — content character tags first (spoken_word, guitar, etc.), then top moods
    tags_html = []
    for ct in present_content_tags:
        bc, fc = CONTENT_CHAR_TAGS[ct]
        label = ct.replace("_", " ")
        tags_html.append(f'<span class="tag" style="border-color:{bc};color:{fc}">{label}</span>')
    for mood, cnt in mood_counts.most_common(5):
        bc, fc = MOOD_COLORS.get(mood, DEFAULT_MOOD_COLOR)
        label = f"{mood} ×{cnt}" if cnt > 1 else mood
        tags_html.append(f'<span class="tag" style="border-color:{bc};color:{fc}">{label}</span>')

    # Similar chips
    similar_html = []
    for sf in similar_files[:4]:
        sname = clean_name(sf)
        sslug = slugify(sname)
        similar_html.append(f'<a class="similar-chip" href="#{sslug}">{sname}</a>')

    # Sparkline
    grad_id = f"sg-{slug}"
    sparkline = sparkline_svg(energy_vals, grad_id)

    num_sections = len(phases_for_file)

    # BPM + key from track features
    bpm = track.get("tempo")
    track_key = track.get("key", "")
    bpm_str = f"{int(round(float(bpm)))} BPM" if bpm else ""
    key_str = track_key or ""
    meta_parts = [dur]
    if key_str:
        meta_parts.append(key_str)
    if bpm_str:
        meta_parts.append(bpm_str)
    meta_line = " · ".join(meta_parts)

    # Structural markers
    struct_badges = []
    if track.get("has_climax"):
        struct_badges.append('<span class="struct-badge climax">↑ climax</span>')
    if track.get("has_breakdown"):
        struct_badges.append('<span class="struct-badge breakdown">↓ breakdown</span>')
    if track.get("has_build_up"):
        struct_badges.append('<span class="struct-badge buildup">→ build-up</span>')
    struct_html = "".join(struct_badges)

    return f"""    <div class="track-card" id="{slug}">
      <div class="track-left">
        <div class="track-header">
          <span class="track-title">{name}</span>
          <span class="track-meta">{meta_line}</span>
        </div>
        <p class="track-narrative">{narrative}</p>
        <div class="track-tags">{''.join(tags_html)}{struct_html}</div>
        <div class="track-similar">
          <span class="similar-label">from same cluster &rarr;</span>
          {''.join(similar_html) or '<span class="similar-chip">unique</span>'}
        </div>
      </div>
      <div class="track-right">
        <div class="track-stats">
          <span><span class="sections-count">{num_sections}</span> sections</span>
          <div class="tension-meter">
            <span class="tlabel">tension</span>
            <div class="tension-track">
              <div class="tension-fill" style="width:{int(tension*100)}%;background:{tcol}"></div>
            </div>
            <span class="tval" style="color:{tcol}">{tension:.2f}</span>
          </div>
        </div>
        <div class="sparkline-wrap">
          {sparkline}
          <div class="tension-label">energy arc &rarr; time</div>
        </div>
      </div>
    </div>"""


def generate(data):
    tracks = data.get("tracks", [])
    phases = data.get("phase_analysis", [])
    cluster_data = data.get("cluster_analysis") or {}
    seq = data.get("sequence_recommendations") or []
    summary = data.get("collection_summary", {})

    # Build phase lookup
    phase_by_file = {}
    for item in phases:
        fn = item.get("filename", "")
        phase_by_file[fn] = item.get("phases", [])

    # Build similarity map
    similar_map = build_cluster_map(cluster_data)

    # Collection stats
    n_tracks = len(tracks)
    total_dur_s = float(summary.get("total_duration_seconds", sum(float(t.get("duration", 0)) for t in tracks)))
    total_dur_h = total_dur_s / 3600
    total_phases = int(summary.get("total_phases", sum(len(v) for v in phase_by_file.values())))
    n_clusters = len(cluster_data) if cluster_data else "?"

    # Unwrap sequence_recommendations if it's a dict with a "sequence" key
    if isinstance(seq, dict):
        seq = seq.get("sequence", [])

    # Sort tracks by sequence if available, else alphabetically
    seq_order = {item["filename"]: item.get("position", i) for i, item in enumerate(seq)} if seq else {}
    sorted_tracks = sorted(
        tracks,
        key=lambda t: seq_order.get(t.get("filename", ""), 9999)
    )

    # Render track cards
    cards = []
    for t in sorted_tracks:
        fn = t.get("filename", "")
        phases_for = phase_by_file.get(fn, [])
        similar = similar_map.get(fn, [])
        cards.append(render_track_card(t, phases_for, similar))

    cards_html = "\n".join(cards)

    # Hero duration string
    if total_dur_h >= 1:
        dur_str = f"{total_dur_h:.1f}h"
    else:
        dur_str = f"{int(total_dur_s/60)}&thinsp;min"

    # Collection analytics for the stats section
    mood_dist = summary.get("mood_distribution", {})
    key_dist = summary.get("key_distribution", {})
    # Compute top moods from tracks if the summary gives aggregated dict
    mood_counter = Counter()
    key_counter = Counter()
    bpm_vals = []
    for t in tracks:
        pm = t.get("primary_mood") or ""
        if pm:
            mood_counter[pm] += 1
        k = t.get("key") or ""
        if k:
            key_counter[k] += 1
        bpm = t.get("tempo")
        if bpm:
            bpm_vals.append(float(bpm))
    top_moods_html = "".join(
        f'<span class="stat-pill" style="border-color:{MOOD_COLORS.get(m, DEFAULT_MOOD_COLOR)[0]};color:{MOOD_COLORS.get(m, DEFAULT_MOOD_COLOR)[1]}">{m} <em>{cnt}</em></span>'
        for m, cnt in mood_counter.most_common(6)
    )
    top_keys_html = "".join(
        f'<span class="stat-pill">{k} <em>{cnt}</em></span>'
        for k, cnt in key_counter.most_common(6)
    )
    avg_bpm = f"{sum(bpm_vals)/len(bpm_vals):.0f}" if bpm_vals else "—"
    bpm_range = f"{min(bpm_vals):.0f}–{max(bpm_vals):.0f}" if bpm_vals else "—"

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>analyze_synths &mdash; a composer&rsquo;s hardware experiment</title>
<style>
  :root {{
    --bg:        #09171f;
    --bg-card:   #0f2130;
    --bg-raised: #152b3a;
    --border:    #1e3a4a;
    --teal:      #4FD1C5;
    --teal-dim:  #2a7a74;
    --teal-text: #81E6D9;
    --pink:      #EC96B8;
    --yellow:    #F4C471;
    --green:     #27AE60;
    --red:       #FF6B6B;
    --text:      #E8F0F2;
    --text-dim:  #7a9aaa;
    --text-mute: #3d6070;
    --mono:      'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    --sans:      -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 15px; line-height: 1.65; }}
  a {{ color: var(--teal); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  nav {{
    position: sticky; top: 0; z-index: 100;
    background: rgba(9,23,31,0.93); backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 2rem; padding: 0.75rem 2rem;
  }}
  .nav-brand {{ font-family: var(--mono); font-size: 0.95rem; color: var(--teal); font-weight: 600; letter-spacing: -0.02em; }}
  nav ul {{ list-style: none; display: flex; gap: 1.5rem; margin-left: auto; }}
  nav ul a {{ font-size: 0.82rem; color: var(--text-dim); letter-spacing: 0.04em; text-transform: uppercase; }}
  nav ul a:hover {{ color: var(--teal); text-decoration: none; }}

  .container {{ max-width: 1060px; margin: 0 auto; padding: 0 2rem; }}
  section {{ padding: 4.5rem 0; }}
  section + section {{ border-top: 1px solid var(--border); }}
  h2 {{ font-size: 1.55rem; font-weight: 600; color: var(--teal-text); margin-bottom: 0.4rem; letter-spacing: -0.02em; }}
  .section-sub {{ color: var(--text-dim); margin-bottom: 2.5rem; font-size: 0.9rem; }}

  #hero {{ padding: 6rem 0 5rem; border-bottom: 1px solid var(--border); }}
  .hero-eyebrow {{ font-family: var(--mono); font-size: 0.75rem; color: var(--teal-dim); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 1.2rem; }}
  .hero-title {{ font-size: clamp(2.4rem,5vw,3.8rem); font-weight: 700; line-height: 1.1; letter-spacing: -0.04em; margin-bottom: 1.4rem; }}
  .hero-title span {{ color: var(--teal); }}
  .hero-desc {{ font-size: 1.1rem; color: var(--text-dim); max-width: 580px; line-height: 1.7; margin-bottom: 2.5rem; }}
  .hero-stats {{ display: flex; gap: 2.5rem; flex-wrap: wrap; align-items: center; }}
  .hero-stat-value {{ font-family: var(--mono); font-size: 2rem; font-weight: 700; color: var(--teal); line-height: 1; margin-bottom: 0.25rem; }}
  .hero-stat-label {{ font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; }}
  .hero-divider {{ width: 1px; background: var(--border); align-self: stretch; min-height: 3rem; }}

  .story-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3rem 4rem; }}
  @media (max-width: 700px) {{ .story-grid {{ grid-template-columns: 1fr; }} }}
  .story-block h3 {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--teal); margin-bottom: 0.7rem; font-weight: 600; }}
  .story-block p {{ color: var(--text-dim); font-size: 0.92rem; line-height: 1.75; }}
  .story-block p + p {{ margin-top: 0.8rem; }}
  .story-block code {{ font-family: var(--mono); font-size: 0.78rem; color: var(--teal-text); background: var(--bg-raised); padding: 0.1em 0.35em; border-radius: 3px; }}

  .bench-table {{ width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.82rem; margin-bottom: 2rem; }}
  .bench-table th {{ text-align: left; color: var(--text-dim); font-weight: 500; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; padding: 0.6rem 1rem 0.6rem 0; border-bottom: 1px solid var(--border); }}
  .bench-table td {{ padding: 0.55rem 1rem 0.55rem 0; border-bottom: 1px solid rgba(30,58,74,0.5); }}
  .bench-table tr:last-child td {{ border-bottom: none; }}
  .filename {{ color: var(--text-dim); }}
  .tdim {{ color: var(--text-dim) !important; }}
  .hw-time {{ color: var(--text); }}
  .rtf {{ color: var(--teal); font-weight: 600; }}
  .rtf-bar-row {{ display: flex; align-items: center; gap: 0.7rem; }}
  .rtf-bar {{ height: 4px; border-radius: 2px; background: var(--teal); opacity: 0.65; }}
  .bench-total td {{ color: var(--teal-text) !important; font-weight: 600; border-top: 1px solid var(--teal-dim) !important; }}

  .bench-cmp {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin-top: 1.5rem; }}
  @media (max-width: 600px) {{ .bench-cmp {{ grid-template-columns: 1fr; }} }}
  .cmp-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem 1.4rem; }}
  .cmp-card.hl {{ border-color: var(--teal-dim); }}
  .cmp-label {{ font-size: 0.72rem; color: var(--text-mute); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }}
  .cmp-value {{ font-family: var(--mono); font-size: 1.8rem; font-weight: 700; color: var(--teal); line-height: 1; margin-bottom: 0.2rem; }}
  .cmp-card:not(.hl) .cmp-value {{ color: var(--text-dim); }}
  .cmp-sub {{ font-size: 0.78rem; color: var(--text-mute); }}

  #stats {{ padding: 4rem 0; border-bottom: 1px solid var(--border); }}
  .stats-rows {{ display: flex; flex-direction: column; gap: 1.2rem; margin-top: 1.5rem; }}
  .stats-row {{ display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }}
  .stats-label {{ font-family: var(--mono); font-size: 0.72rem; color: var(--text-mute); text-transform: uppercase; letter-spacing: 0.06em; min-width: 8rem; }}
  .stats-value {{ font-family: var(--mono); font-size: 1rem; color: var(--teal); font-weight: 600; }}
  .stats-range {{ font-family: var(--mono); font-size: 0.78rem; color: var(--text-dim); }}
  .stats-pills {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
  .stat-pill {{ font-size: 0.68rem; padding: 0.2em 0.6em; border-radius: 3px; border: 1px solid var(--border); color: var(--text-dim); background: transparent; }}
  .stat-pill em {{ font-style: normal; opacity: 0.65; }}
  .library-grid {{ display: grid; gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
  .track-card {{ background: var(--bg-card); padding: 1.5rem 1.8rem; display: grid; grid-template-columns: 1fr auto; gap: 1rem 2rem; align-items: start; transition: background 0.15s; }}
  .track-card:target {{ background: var(--bg-raised); outline: 1px solid var(--teal-dim); }}
  .track-header {{ display: flex; align-items: baseline; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 0.5rem; }}
  .track-title {{ font-weight: 600; font-size: 0.95rem; letter-spacing: -0.01em; }}
  .track-meta {{ font-family: var(--mono); font-size: 0.72rem; color: var(--text-mute); }}
  .track-narrative {{ font-size: 0.83rem; color: var(--text-dim); line-height: 1.6; margin-bottom: 0.9rem; }}
  .track-tags {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.75rem; }}
  .tag {{ font-size: 0.68rem; padding: 0.2em 0.55em; border-radius: 3px; font-weight: 500; border: 1px solid; background: transparent; }}
  .struct-badge {{ font-size: 0.65rem; padding: 0.15em 0.45em; border-radius: 3px; background: transparent; border: 1px solid; opacity: 0.7; }}
  .struct-badge.climax {{ border-color: #F4C471; color: #F4C471; }}
  .struct-badge.breakdown {{ border-color: #607D8B; color: #607D8B; }}
  .struct-badge.buildup {{ border-color: #27AE60; color: #27AE60; }}
  .track-similar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem; font-size: 0.72rem; }}
  .similar-label {{ color: var(--text-mute); }}
  .similar-chip {{ background: var(--bg-raised); border: 1px solid var(--border); border-radius: 3px; padding: 0.15em 0.5em; color: var(--text-dim); font-size: 0.68rem; display: inline-block; }}
  .similar-chip:hover {{ border-color: var(--teal-dim); color: var(--teal-text); text-decoration: none; }}
  .track-right {{ display: flex; flex-direction: column; align-items: flex-end; gap: 0.6rem; }}
  .track-stats {{ display: flex; flex-direction: column; align-items: flex-end; gap: 0.25rem; font-family: var(--mono); font-size: 0.72rem; color: var(--text-mute); white-space: nowrap; }}
  .sections-count {{ color: var(--teal); font-weight: 600; font-size: 0.85rem; }}
  .tension-meter {{ display: flex; align-items: center; gap: 0.4rem; }}
  .tlabel {{ font-size: 0.7rem; color: var(--text-mute); }}
  .tension-track {{ width: 48px; height: 4px; background: var(--bg-raised); border-radius: 2px; overflow: hidden; }}
  .tension-fill {{ height: 100%; border-radius: 2px; }}
  .tval {{ font-family: var(--mono); font-size: 0.7rem; }}
  .sparkline-wrap {{ width: 200px; }}
  .sparkline-wrap svg {{ display: block; }}
  .tension-label {{ font-family: var(--mono); font-size: 0.62rem; color: var(--text-mute); text-align: right; margin-top: 0.2rem; }}
  @media (max-width: 720px) {{
    .track-card {{ grid-template-columns: 1fr; }}
    .track-right {{ align-items: flex-start; flex-direction: row; flex-wrap: wrap; }}
    .sparkline-wrap {{ width: 100%; }}
    .sparkline-wrap svg {{ width: 100%; }}
  }}

  .next-grid {{ display: grid; grid-template-columns: repeat(2,1fr); gap: 1rem; }}
  @media (max-width: 600px) {{ .next-grid {{ grid-template-columns: 1fr; }} }}
  .next-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1.3rem 1.5rem; }}
  .next-card h3 {{ font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }}
  .next-card h3::before {{ content: ''; display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--teal); flex-shrink: 0; }}
  .next-card p {{ font-size: 0.82rem; color: var(--text-dim); line-height: 1.65; }}

  footer {{ border-top: 1px solid var(--border); padding: 2rem 0; }}
  .foot-inner {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; font-size: 0.78rem; color: var(--text-mute); }}
  .foot-mono {{ font-family: var(--mono); font-size: 0.72rem; }}
</style>
</head>
<body>

<nav>
  <span class="nav-brand">analyze_synths</span>
  <ul>
    <li><a href="#story">story</a></li>
    <li><a href="#benchmarks">benchmarks</a></li>
    <li><a href="#stats">stats</a></li>
    <li><a href="#library">library</a></li>
    <li><a href="#next">next</a></li>
    <li><a href="https://github.com/tsingletaryTT/analyze_synths">github</a></li>
  </ul>
</nav>

<!-- hero -->
<section id="hero">
<div class="container">
  <p class="hero-eyebrow">Tenstorrent Blackhole &nbsp;/&nbsp; JAX PJRT &nbsp;/&nbsp; temporal narrative analysis</p>
  <h1 class="hero-title">Teaching hardware<br>to <span>hear music</span></h1>
  <p class="hero-desc">A composer&rsquo;s toolkit for understanding a personal synthesizer library. {dur_str} of ambient pieces, spoken word, and electronic music analyzed on custom silicon. Every track gets sections, tension arcs, mood vocabulary, and cluster groupings.</p>
  <div class="hero-stats">
    <div>
      <div class="hero-stat-value">3,931x</div>
      <div class="hero-stat-label">real-time STFT</div>
    </div>
    <div class="hero-divider"></div>
    <div>
      <div class="hero-stat-value">{n_tracks}</div>
      <div class="hero-stat-label">tracks analyzed</div>
    </div>
    <div class="hero-divider"></div>
    <div>
      <div class="hero-stat-value">{dur_str}</div>
      <div class="hero-stat-label">total audio</div>
    </div>
    <div class="hero-divider"></div>
    <div>
      <div class="hero-stat-value">{total_phases:,}</div>
      <div class="hero-stat-label">sections found</div>
    </div>
  </div>
</div>
</section>

<!-- story -->
<section id="story">
<div class="container">
  <h2>The story</h2>
  <p class="section-sub">From a library management problem to Blackhole silicon</p>
  <div class="story-grid">
    <div class="story-block">
      <h3>The problem</h3>
      <p>A growing library of synthesizer pieces &mdash; long-form ambient tracks, generative drones, evolving textures &mdash; with no good way to understand what was in it. Standard tools give BPM and key. Not useful when the track has no tempo and the key is ambiguous.</p>
      <p>The goal was a creative vocabulary: <em>spacey, crystalline, droning, organic, oozy</em>. Descriptors that mean something to a synthesizer composer.</p>
    </div>
    <div class="story-block">
      <h3>v1 &mdash; Feature extraction</h3>
      <p>A librosa-based pipeline extracting 80+ features per track mapped through empirically-tuned thresholds to 176 descriptors: 9 core moods, 8 extended, 100 advanced, 59 character tags. K-means clustering, phase detection, and sequence recommendation on top.</p>
      <p>It worked. The results felt right. The problem was speed on long files.</p>
    </div>
    <div class="story-block">
      <h3>The insight</h3>
      <p>STFT is just three matrix multiplications. Given pre-computed static bases:</p>
      <p><code>cos_proj = frame @ cos_basis</code><br>
         <code>sin_proj = frame @ sin_basis</code><br>
         <code>mag = sqrt(cos&sup2; + sin&sup2; + &epsilon;)</code><br>
         <code>mel = mag @ mel_filter</code></p>
      <p>That&rsquo;s a kernel the hardware was built for.</p>
    </div>
    <div class="story-block">
      <h3>TT-Lang kernel</h3>
      <p>The first hardware target was TT-Lang (ttnn), Tenstorrent&rsquo;s Python-like compute language. Prototype ran. Then JAX PJRT arrived: compile the same matmuls with <code>jax.jit</code>, let the XLA backend lower to Blackhole Tensix cores. Cleaner, faster, portable.</p>
    </div>
    <div class="story-block">
      <h3>The hardware journey</h3>
      <p>Four P300C Blackhole chips in a P150X4 mesh. Device reset between runs (stuck ethernet cores after a killed process). Single-process dispatch &mdash; multi-threading causes lock contention on chip init. JIT compiles once per kernel config, shape-padded to 1376 frames.</p>
    </div>
    <div class="story-block">
      <h3>Full library ({n_tracks} tracks)</h3>
      <p>Material from 1990s through present: analog synthesizers, drum machines, spoken word, generative pieces, ambient drones. {n_clusters} clusters by spectral similarity. {total_phases:,} musical sections across {dur_str} of audio. Every track sequenced by the recommended listening arc.</p>
    </div>
  </div>
</div>
</section>

<!-- benchmarks -->
<section id="benchmarks">
<div class="container">
  <h2>Benchmarks</h2>
  <p class="section-sub">P300C Blackhole, 4-chip mesh, JAX PJRT, <code>jax.jit</code> fused STFT kernel</p>
  <table class="bench-table">
    <thead>
      <tr>
        <th>track</th><th>duration</th><th>hw time</th><th>rtf &nbsp;&nbsp;</th><th>vs numpy</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="filename">2022 09 17 caving in</td>
        <td class="tdim">9:28</td>
        <td class="hw-time">0.144&thinsp;s</td>
        <td class="rtf"><div class="rtf-bar-row"><div class="rtf-bar" style="width:calc(3941/39.41 * 1px)"></div>3,941x</div></td>
        <td class="tdim">3.2x</td>
      </tr>
      <tr>
        <td class="filename">2020 cricketeering</td>
        <td class="tdim">17:52</td>
        <td class="hw-time">0.271&thinsp;s</td>
        <td class="rtf"><div class="rtf-bar-row"><div class="rtf-bar" style="width:calc(3956/39.41 * 1px)"></div>3,956x</div></td>
        <td class="tdim">3.1x</td>
      </tr>
      <tr>
        <td class="filename">2022-09-25 while madeline sleeps</td>
        <td class="tdim">11:47</td>
        <td class="hw-time">0.179&thinsp;s</td>
        <td class="rtf"><div class="rtf-bar-row"><div class="rtf-bar" style="width:calc(3956/39.41 * 1px)"></div>3,956x</div></td>
        <td class="tdim">3.2x</td>
      </tr>
    </tbody>
    <tfoot>
      <tr class="bench-total">
        <td>3-track sample</td>
        <td>39:07</td>
        <td>0.594&thinsp;s</td>
        <td>3,931x avg</td>
        <td>3.2x</td>
      </tr>
    </tfoot>
  </table>
  <div class="bench-cmp">
    <div class="cmp-card">
      <div class="cmp-label">NumPy baseline</div>
      <div class="cmp-value" style="color:var(--text-dim)">1.9&thinsp;s</div>
      <div class="cmp-sub">3-track sample, float64</div>
    </div>
    <div class="cmp-card hl">
      <div class="cmp-label">Blackhole hardware</div>
      <div class="cmp-value">0.59&thinsp;s</div>
      <div class="cmp-sub">float32, JIT-compiled, 4 chips</div>
    </div>
    <div class="cmp-card">
      <div class="cmp-label">Speedup</div>
      <div class="cmp-value" style="color:var(--yellow)">3.2x</div>
      <div class="cmp-sub">hardware vs NumPy reference</div>
    </div>
  </div>
</div>
</section>

<!-- collection stats -->
<section id="stats">
<div class="container">
  <h2>Collection at a glance</h2>
  <p class="section-sub">Tempo, mood, and key across {n_tracks} tracks</p>
  <div class="stats-rows">
    <div class="stats-row">
      <span class="stats-label">Avg tempo</span>
      <span class="stats-value mono">{avg_bpm} BPM</span>
      <span class="stats-range mono">range {bpm_range}</span>
    </div>
    <div class="stats-row">
      <span class="stats-label">Top moods</span>
      <span class="stats-pills">{top_moods_html}</span>
    </div>
    <div class="stats-row">
      <span class="stats-label">Key distribution</span>
      <span class="stats-pills">{top_keys_html}</span>
    </div>
  </div>
</div>
</section>

<!-- library -->
<section id="library">
<div class="container">
  <h2>The library</h2>
  <p class="section-sub">{n_tracks} tracks &mdash; {dur_str} &mdash; {n_clusters} clusters &mdash; ordered by recommended listening sequence</p>
  <div class="library-grid">
{cards_html}
  </div>
</div>
</section>

<!-- next -->
<section id="next">
<div class="container">
  <h2>What&rsquo;s next</h2>
  <p class="section-sub">Directions from here</p>
  <div class="next-grid">
    <div class="next-card">
      <h3>Narrative generation</h3>
      <p>Phase-level prose descriptions beyond tag lists. A language model reading the energy arc and writing what it hears &mdash; not a transcript, a listener&rsquo;s impression.</p>
    </div>
    <div class="next-card">
      <h3>Cross-piece similarity</h3>
      <p>Cluster membership gives broad groupings. Next step: pairwise spectral distance for a proper similarity graph &mdash; &ldquo;this piece sounds like that one because of X.&rdquo;</p>
    </div>
    <div class="next-card">
      <h3>Parallel dispatch</h3>
      <p>The single-process constraint on TT hardware is a workaround for JAX init lock contention. The right fix is a single shared device context with queue-based dispatch.</p>
    </div>
    <div class="next-card">
      <h3>Expanded vocabulary</h3>
      <p>176 descriptors are a start. The advanced mood library has 100 more. Calibrating them against spoken word, 1990s drum machines, and acoustic instruments requires different thresholds.</p>
    </div>
    <div class="next-card">
      <h3>Real-time analysis</h3>
      <p>The kernel already operates at 3,931x real-time. A streaming mode that reads from a DAW or audio interface and builds sections on the fly is architecturally feasible.</p>
    </div>
    <div class="next-card">
      <h3>Export to DAW</h3>
      <p>Phase boundaries and mood tags as DAW markers. Import a 25-minute generative piece and get section markers automatically placed with creative labels.</p>
    </div>
  </div>
</div>
</section>

<footer>
<div class="container">
  <div class="foot-inner">
    <span>analyze_synths &mdash; a composer&rsquo;s hardware experiment</span>
    <span class="foot-mono">Tenstorrent Blackhole P300C &nbsp;&middot;&nbsp; JAX PJRT 0.7.1 &nbsp;&middot;&nbsp; {n_tracks} tracks &nbsp;&middot;&nbsp; {total_phases:,} sections</span>
    <a href="https://github.com/tsingletaryTT/analyze_synths">github</a>
  </div>
</div>
</footer>

</body>
</html>"""
    return html


def main():
    if not JSON_PATH.exists():
        print(f"ERROR: {JSON_PATH} not found. Run run_full_analysis.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {JSON_PATH} ...")
    with open(JSON_PATH) as f:
        data = json.load(f)

    html = generate(data)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Written {OUT_PATH} ({len(html):,} bytes)")
    print(f"  Tracks: {len(data.get('tracks', []))}")
    print(f"  Phases: {sum(len(p.get('phases',[])) for p in data.get('phase_analysis',[]))}")


if __name__ == "__main__":
    main()
