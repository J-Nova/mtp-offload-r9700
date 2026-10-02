#!/usr/bin/env python3
"""Generate the Radiance MXFP4 Grafana dashboard (consolidated layout).

Run:  python3 aijuus/grafana/gen-dashboard.py > aijuus/grafana/vllm-dashboard.json
Keeps the same uid (radiance-mxfp4) so the Grafana file provider + API update in place.
"""
import json

DS = {"type": "prometheus", "uid": "prometheus"}

_next_id = [1000]
def nid():
    _next_id[0] += 1
    return _next_id[0]


def row(title, y, collapsed=False, panels=None):
    r = {
        "id": nid(),
        "type": "row",
        "title": title,
        "collapsed": collapsed,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
    }
    if collapsed and panels:
        r["panels"] = panels
    return r


def stat(title, x, y, w, h, expr, unit=None, desc=None, legend=None, decimals=None, steps=None, color_mode=None):
    p = {
        "id": nid(),
        "type": "stat",
        "title": title,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": DS,
        "options": {"textMode": "value"},
        "targets": [{"refId": "A", "expr": expr}],
    }
    if legend:
        p["targets"][0]["legendFormat"] = legend
    if color_mode:
        p["options"]["colorMode"] = color_mode
    if desc:
        p["description"] = desc
    fc = {}
    if unit:
        fc["defaults"] = {"unit": unit}
    if decimals is not None:
        fc.setdefault("defaults", {})["decimals"] = decimals
    if steps:
        fc.setdefault("defaults", {})["thresholds"] = {"mode": "absolute", "steps": steps}
    if fc:
        p["fieldConfig"] = fc
    return p


def ts(title, x, y, w, h, targets, unit=None, desc=None, right_unit=None, right_refs=None, minv=None, maxv=None):
    """targets: list of (refId, expr, legend). right_unit + right_refs put the named
    targets on a right-hand axis (with their own unit); the rest stay on the left."""
    p = {
        "id": nid(),
        "type": "timeseries",
        "title": title,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": DS,
        "targets": [{"refId": r, "expr": e, "legendFormat": lg} for (r, e, lg) in targets],
        "fieldConfig": {
            "defaults": {
                "unit": unit or "short",
                "custom": {"drawStyle": "line", "lineWidth": 1, "spanNulls": True, "fillOpacity": 5},
            }
        },
    }
    if desc:
        p["description"] = desc
    if minv is not None:
        p["fieldConfig"]["defaults"]["min"] = minv
    if maxv is not None:
        p["fieldConfig"]["defaults"]["max"] = maxv
    if right_unit:
        refs = right_refs if right_refs is not None else [targets[-1][0]]
        p["fieldConfig"]["overrides"] = [
            {
                "matcher": {"id": "byFrameRefID", "options": r},
                "properties": [
                    {"id": "unit", "value": right_unit},
                    {"id": "custom.axisPlacement", "value": "right"},
                ],
            }
            for r in refs
        ]
    return p


def table(title, x, y, w, h, expr, desc=None, fields=None, rename=None, mappings=None):
    p = {
        "id": nid(),
        "type": "table",
        "title": title,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": DS,
        "options": {"showHeader": True, "orientation": "column", "footer": {"show": False}},
        "targets": [{"refId": "A", "expr": expr, "instant": True, "format": "table"}],
        "fieldConfig": {"defaults": {}},
    }
    if desc:
        p["description"] = desc
    transforms = []
    if fields:
        transforms.append({"id": "filterFieldsByName", "options": {"include": {"names": fields}}})
    if rename:
        transforms.append({"id": "organize", "options": {"excludeByName": {}, "renameByName": rename}})
    if transforms:
        p["transformations"] = transforms
    if mappings:
        p["fieldConfig"]["defaults"]["mappings"] = mappings
    return p


panels = []

# ============================================================ ROW 0: model state + health
panels.append(row("Model state + health", 0))
panels.append(table("Model state (live)", 0, 1, 8, 6, "radiance_model_state_ready",
                   desc="Which model is loaded on each instance + whether it is ready, from the state-collector textfile metric.",
                   fields=["model", "exported_instance", "Value"],
                   rename={"model": "Model", "exported_instance": "Instance", "Value": "Status"},
                   mappings=[{"type": "value", "options": {
                       "0": {"text": "NOT READY", "color": "red", "index": 0},
                       "1": {"text": "ready", "color": "green", "index": 1}}}]))
panels.append(stat("Spec acceptance %", 8, 1, 4, 6,
                   "100 * sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m])) / clamp_min(sum(rate(vllm:spec_decode_num_draft_tokens_total[5m])), 1e-9)",
                   unit="percent", decimals=1,
                   desc="Accepted / draft tokens over 5m. The headline spec-decode efficiency number. Green >60%, amber 45-60%, red <45%.",
                   steps=[{"color": "red", "value": None}, {"color": "orange", "value": 45}, {"color": "green", "value": 60}],
                   color_mode="value"))
panels.append(stat("Hottest GPU temp (C)", 12, 1, 4, 6, "max(amd_gpu_temp_junction_celsius)", unit="celsius",
                   desc="Max junction temp across cards. R9700 throttles near 95-100C."))
panels.append(stat("Prefill (tok/s)", 16, 1, 4, 6,
                   "sum(rate(vllm:prompt_tokens_by_source_total[1m]))",
                   desc="Total prefill tokens/s across all sources (compute + cache hit + offload), 1m rate."))
panels.append(stat("Decode (tok/s)", 20, 1, 4, 6,
                   "sum(rate(vllm:generation_tokens_total[1m]))",
                   desc="Decode (generation) tokens/s, 1m rate."))

# ============================================================ ROW 1: serving throughput + spec + latency
panels.append(row("Serving: throughput, spec & latency", 7))
panels.append(ts("Throughput (tok/s)", 0, 8, 8, 8, [
    ("A", "sum(rate(vllm:generation_tokens_total[1m]))", "decode"),
    ("B", "sum(rate(vllm:prompt_tokens_by_source_total{source=\"local_compute\"}[1m]))", "prefill compute"),
    ("C", "sum(rate(vllm:prompt_tokens_by_source_total{source=\"local_cache_hit\"}[1m]))", "prefill cache hit"),
    ("D", "sum(rate(vllm:prompt_tokens_by_source_total{source=\"external_kv_transfer\"}[1m]))", "prefill offload"),
], unit="short", desc="Decode + prefill tokens/s by source (1m rate)."))
panels.append(ts("Spec acceptance (%)", 8, 8, 8, 8, [
    ("A", "100 * sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m])) / clamp_min(sum(rate(vllm:spec_decode_num_draft_tokens_total[5m])), 1e-9)", "acceptance (total)"),
    ("B", "100 * sum by (instance) (rate(vllm:spec_decode_num_accepted_tokens_total[5m])) / clamp_min(sum by (instance) (rate(vllm:spec_decode_num_draft_tokens_total[5m])), 1e-9)", "{{instance}}"),
], unit="percent", minv=0, maxv=100,
   desc="Accepted / draft tokens (5m). Total + per instance. The core spec-decode efficiency number."))
panels.append(ts("Latency (s)", 16, 8, 8, 8, [
    ("A", "histogram_quantile(0.50, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))", "TTFT p50"),
    ("B", "histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))", "TTFT p95"),
    ("C", "histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))", "TTFT p99"),
    ("D", "histogram_quantile(0.50, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[5m])))", "ITL p50"),
], unit="s", desc="Time-to-first-token percentiles + inter-token latency p50."))
panels.append(ts("Per-position accepted (tok/s)", 0, 16, 8, 8, [
    ("A", "sum by (position) (rate(vllm:spec_decode_num_accepted_tokens_per_pos_total[5m]))", "pos {{position}}"),
], unit="short", desc="Accepted tokens/s per spec position -- the draft-depth decay curve (which positions are productive)."))
panels.append(ts("Concurrency", 8, 16, 8, 8, [
    ("A", "vllm:num_requests_running", "{{instance}} running"),
    ("B", "vllm:num_requests_waiting", "{{instance}} waiting"),
], unit="short", desc="Requests running + waiting per instance."))
panels.append(ts("KV cache usage (%)", 16, 16, 8, 8, [
    ("A", "(vllm:gpu_cache_usage_perc or vllm:kv_cache_usage_perc) * 100", "{{instance}}"),
], unit="percent", minv=0, maxv=100, desc="GPU KV cache occupancy per instance."))

# ============================================================ ROW 2: GPU per card
panels.append(row("GPU (per card)", 24))
panels.append(ts("GPU utilization (%)", 0, 25, 8, 8, [
    ("A", "amd_gpu_busy_percent", "{{gpu}}"),
], unit="percent", minv=0, maxv=100, desc="GPU busy % per card."))
panels.append(ts("VRAM (GiB + %)", 8, 25, 8, 8, [
    ("A", "amd_vram_used_bytes", "{{gpu}} used"),
    ("B", "amd_vram_used_bytes / amd_vram_total_bytes * 100", "{{gpu}} %"),
], unit="bytes", right_unit="percent", desc="VRAM used (GiB, left) + % of total (right)."))
panels.append(ts("GPU power (W)", 16, 25, 8, 8, [
    ("A", "amd_gpu_power_watts", "{{gpu}}"),
    ("B", "amd_gpu_power_cap_watts", "{{gpu}} cap"),
], unit="watt", desc="Drawn power vs cap per card."))
panels.append(ts("GPU temps (C)", 0, 33, 8, 8, [
    ("A", "amd_gpu_temp_edge_celsius", "{{gpu}} edge"),
    ("B", "amd_gpu_temp_junction_celsius", "{{gpu}} junction"),
    ("C", "amd_gpu_temp_mem_celsius", "{{gpu}} mem"),
], unit="celsius", desc="Edge / junction / memory temps per card."))
panels.append(ts("GPU clocks (Hz)", 8, 33, 8, 8, [
    ("A", "amd_gpu_sclk_hz", "{{gpu}} shader"),
    ("B", "amd_gpu_mclk_hz", "{{gpu}} memory"),
], unit="hz", desc="Shader + memory clock per card (auto-scaled to MHz)."))
panels.append(ts("GPU fan", 16, 33, 8, 8, [
    ("A", "amd_gpu_fan_rpm", "{{gpu}} rpm"),
], unit="rpm", desc="Fan speed per card."))

# ============================================================ ROW 3: GPU throttle / limits / PCIe / faults (collapsed)
row3_panels = [
    ts("Throttle & limits", 0, 0, 8, 8, [
        ("A", "amd_gpu_power_watts / clamp_min(amd_gpu_power_cap_watts, 1)", "{{gpu}} power-limit"),
        ("B", "amd_gpu_temp_junction_crit_celsius - amd_gpu_temp_junction_celsius", "{{gpu}} thermal headroom"),
    ], unit="short", right_unit="celsius",
     desc="Power-limit ratio (0-1, left) + thermal headroom to crit (C, right)."),
    ts("PCIe link", 8, 0, 8, 8, [
        ("A", "amd_pcie_current_link_width", "{{gpu}} width"),
        ("B", "amd_pcie_max_link_width", "{{gpu}} max width"),
        ("C", "amd_pcie_current_link_speed_gt_s", "{{gpu}} speed"),
    ], unit="short", right_unit="short",
     desc="PCIe link width (lanes, left) + speed (GT/s, right). Card1 is x1."),
    ts("GPU mem & volt", 16, 0, 8, 8, [
        ("A", "amd_mem_busy_percent", "{{gpu}} UMC busy"),
        ("B", "amd_gtt_used_percent", "{{gpu}} GTT"),
        ("C", "amd_gpu_voltage_mv", "{{gpu}} volt"),
    ], unit="percent", right_unit="voltm",
     desc="UMC busy % + GTT % (left) + core voltage (right)."),
    stat("GPU ECC / AER / RAS faults", 0, 8, 8, 6,
         "max(amd_gpu_aer_fatal_total) + max(amd_gpu_ras_fatal_total) + max(amd_gpu_vram_bad_pages)",
         desc="Sum of fatal AER + fatal RAS + bad VRAM pages. 0 = clean; any increase = investigate.", decimals=0),
    ts("Throttle reason flags (derived)", 8, 8, 8, 6, [
        ("A", "(amd_gpu_power_watts / clamp_min(amd_gpu_power_cap_watts, 1)) > 0.98", "{{gpu}} POWER_LIMIT"),
        ("B", "(amd_gpu_temp_junction_crit_celsius - amd_gpu_temp_junction_celsius) < 5", "{{gpu}} THERMAL"),
    ], unit="short", desc="1 when the card is power- or thermally throttling."),
    ts("GPU throttle status (raw)", 16, 8, 8, 6, [
        ("A", "amd_gpu_throttle_status", "{{gpu}} status"),
        ("B", "amd_gpu_indep_throttle_status", "{{gpu}} indep"),
    ], unit="short", desc="Raw throttle status bitfield per card."),
]
panels.append(row("GPU throttle / limits / PCIe / faults", 41, collapsed=True, panels=row3_panels))

# ============================================================ ROW 4: CPU / host
panels.append(row("CPU / host", 42))
panels.append(ts("CPU (usage % + load)", 0, 43, 8, 8, [
    ("A", "100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])))", "usage"),
    ("B", "node_load1", "load 1m"),
    ("C", "node_load5", "load 5m"),
], unit="percent", right_unit="short", right_refs=["B", "C"], desc="CPU usage % (left) + load average (right)."))
panels.append(ts("CPU power (W)", 8, 43, 8, 8, [
    ("A", "max(amd_cpu_power_watts)", "package"),
], unit="watt", desc="CPU package power (RAPL)."))
panels.append(ts("Host memory", 16, 43, 8, 8, [
    ("A", "node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes", "used"),
    ("B", "node_memory_SwapTotal_bytes - node_memory_SwapFree_bytes", "swap"),
], unit="bytes", desc="RAM used + swap used."))
panels.append(ts("Disk", 0, 51, 8, 8, [
    ("A", "100 * (1 - node_filesystem_avail_bytes{mountpoint=\"/\"} / node_filesystem_size_bytes{mountpoint=\"/\"})", "used %"),
    ("B", "sum(rate(node_disk_read_bytes_total[5m]))", "read"),
    ("C", "sum(rate(node_disk_written_bytes_total[5m]))", "write"),
], unit="percent", right_unit="Bps", right_refs=["B", "C"], desc="Root disk used % (left) + I/O read/write (right)."))
panels.append(ts("Network", 8, 51, 8, 8, [
    ("A", "sum(rate(node_network_receive_bytes_total{device!~\"lo\"}[5m]))", "rx"),
    ("B", "sum(rate(node_network_transmit_bytes_total{device!~\"lo\"}[5m]))", "tx"),
], unit="Bps", desc="Network rx/tx (excl. loopback)."))

# ============================================================ ROW 5: KV Offload summary
panels.append(row("KV Offload (summary)", 59))
panels.append(stat("RAM tier size", 0, 60, 4, 4, "sum(radiance_kvram_allocated_bytes)", unit="bytes", desc="Pinned CPU KV tier total."))
panels.append(stat("Disk tier used %", 4, 60, 4, 4, "radiance_kvcache_bytes / radiance_kvcache_max_bytes * 100", unit="percent", desc="Shared fs tier fill vs cap."))
panels.append(stat("Offload hit rate %", 8, 60, 4, 4,
                   "100 * sum(rate(vllm:external_prefix_cache_hits_total[5m])) / clamp_min(sum(rate(vllm:external_prefix_cache_queries_total[5m])), 1e-9)",
                   unit="percent", desc="External (offload) prefix-cache hit rate."))
panels.append(stat("Disk tier blocks", 12, 60, 4, 4, "radiance_kvcache_blocks", unit="short", desc="Blocks in the fs tier."))
panels.append(ts("RAM tier allocated (per engine)", 0, 64, 8, 8, [
    ("A", "radiance_kvram_allocated_bytes", "{{engine}}"),
], unit="bytes", desc="Pinned RAM tier per engine."))
panels.append(ts("Disk tier bytes vs cap", 8, 64, 8, 8, [
    ("A", "radiance_kvcache_bytes", "bytes"),
    ("B", "radiance_kvcache_max_bytes", "cap"),
], unit="bytes", desc="fs tier used vs hard cap."))
panels.append(ts("Offload transfer rate (B/s)", 16, 64, 8, 8, [
    ("A", "sum(rate(vllm:kv_offload_total_bytes_total{transfer_type=\"CPU_to_GPU\"}[5m]))", "restore (CPU->GPU)"),
    ("B", "sum(rate(vllm:kv_offload_total_bytes_total{transfer_type=\"GPU_to_CPU\"}[5m]))", "store (GPU->CPU)"),
], unit="Bps", desc="Offload restore/store throughput."))

# ============================================================ ROW 6: KV Offload deep diagnostics (collapsed)
row6_panels = [
    ts("Lookup terminal outcomes (/s)", 0, 0, 8, 8, [
        ("A", "sum(rate(vllm:kv_offload_lookup_calls_total[5m]))", "calls"),
        ("B", "sum(rate(vllm:kv_offload_lookup_served_total[5m]))", "served"),
        ("C", "sum(rate(vllm:kv_offload_lookup_skip_short_window_total[5m])) or vector(0)", "skip short_window"),
        ("D", "sum(rate(vllm:kv_offload_lookup_skip_zero_hit_total[5m])) or vector(0)", "skip zero_hit"),
        ("E", "sum(rate(vllm:kv_offload_lookup_deferred_backend_total[5m])) or vector(0)", "deferred backend"),
        ("F", "sum(rate(vllm:kv_offload_lookup_deferred_loading_total[5m])) or vector(0)", "deferred loading"),
    ], unit="short", desc="Offload lookup terminal outcomes per second."),
    ts("Lookup chunk outcomes (/s)", 8, 0, 8, 8, [
        ("A", "sum(rate(vllm:kv_offload_lookup_chunk_hit_total[5m]))", "hit"),
        ("B", "sum(rate(vllm:kv_offload_lookup_chunk_hit_pending_total[5m])) or vector(0)", "hit_pending"),
        ("C", "sum(rate(vllm:kv_offload_lookup_chunk_retry_total[5m])) or vector(0)", "retry"),
        ("D", "sum(rate(vllm:kv_offload_lookup_chunk_miss_total[5m]))", "miss"),
    ], unit="short", desc="Offload chunk-level lookup outcomes per second."),
    ts("Lookup served tokens/s", 16, 0, 8, 8, [
        ("A", "sum(rate(vllm:kv_offload_lookup_served_tokens_total[5m]))", "served tokens/s"),
    ], unit="short", desc="Tokens served from the offload tier per second."),
    ts("Tier capacity vs used (bytes)", 0, 8, 8, 8, [
        ("A", "vllm:kv_offload_tier_capacity_bytes", "{{tier}} capacity"),
        ("B", "vllm:kv_offload_tier_used_bytes", "{{tier}} used"),
    ], unit="bytes", desc="Per-tier capacity vs used bytes."),
    ts("Tier hit blocks/tokens", 8, 8, 8, 8, [
        ("A", "sum by (tier) (rate(vllm:kv_offload_tier_hit_blocks_total[5m]))", "{{tier}} hit blocks/s"),
        ("B", "sum by (tier) (rate(vllm:kv_offload_tier_hit_tokens_total[5m]))", "{{tier}} hit tokens/s"),
    ], unit="short", desc="Per-tier hit blocks + tokens per second."),
    ts("Tier stall / evictions (/s)", 16, 8, 8, 8, [
        ("A", "sum by (tier) (rate(vllm:kv_offload_tier_stall_seconds_total[5m]))", "{{tier}} stall s/s"),
        ("B", "sum by (tier) (rate(vllm:kv_offload_tier_evictions_total[5m]))", "{{tier}} evictions/s"),
        ("C", "sum by (tier) (rate(vllm:kv_offload_tier_lookup_miss_evicted_total[5m])) or vector(0)", "{{tier}} miss-evicted/s"),
    ], unit="short", desc="Per-tier stall seconds + evictions + miss-evictions."),
    ts("Promotion initiated / refused (/s)", 0, 16, 8, 8, [
        ("A", "sum by (tier) (rate(vllm:kv_offload_promotion_initiated_total[5m]))", "{{tier}} initiated/s"),
        ("B", "sum by (tier) (rate(vllm:kv_offload_promotion_refused_total[5m])) or vector(0)", "{{tier}} refused/s"),
        ("C", "sum by (tier) (rate(vllm:kv_offload_promotion_refused_no_evictable_total[5m])) or vector(0)", "{{tier}} no_evictable/s"),
    ], unit="short", desc="Offload promotion initiated vs refused (with no-evictable reason)."),
    ts("Deferral lifecycle (/s) + depth", 8, 16, 8, 8, [
        ("A", "sum(rate(vllm:kv_offload_lookup_deferral_total[5m]))", "deferral total"),
        ("B", "sum(rate(vllm:kv_offload_lookup_deferral_served_total[5m]))", "served"),
        ("C", "sum(rate(vllm:kv_offload_lookup_deferral_gave_up_total[5m]))", "gave_up"),
        ("D", "sum(rate(vllm:kv_offload_lookup_deferral_depth_sum[5m])) / clamp_min(sum(rate(vllm:kv_offload_lookup_deferral_depth_count[5m])), 1e-9)", "avg depth"),
    ], unit="short", desc="Lookup deferral lifecycle per second + average deferral depth (steps)."),
    ts("CPU tier headroom / FS backlog", 16, 16, 8, 8, [
        ("A", "max(vllm:kv_offload_cpu_cache_evictable_perc) * 100", "evictable %"),
        ("B", "max(vllm:kv_offload_cpu_cache_free_perc) * 100", "free %"),
        ("C", "sum(vllm:kv_offload_fs_inflight_jobs)", "FS inflight jobs"),
    ], unit="percent", right_unit="short", right_refs=["C"],
       desc="CPU tier admission headroom (left) + FS tier cascade backlog (right)."),
]
panels.append(row("KV Offload Deep Diagnostics", 72, collapsed=True, panels=row6_panels))

dashboard = {
    "title": "Radiance MXFP4 (2x GPU, data-parallel) -- full hardware",
    "uid": "radiance-mxfp4",
    "schemaVersion": 39,
    "version": 3,
    "tags": ["vllm", "gpu", "cpu", "hardware", "spec-decode"],
    "timezone": "browser",
    "refresh": "30s",
    "time": {"from": "now-1h", "to": "now"},
    "editable": True,
    "panels": panels,
}

print(json.dumps(dashboard, indent=2))
