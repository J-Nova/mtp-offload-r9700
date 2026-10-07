# GPU Host Configuration Map

Where everything for the two AMD GPUs is defined on this host, what the effective
settings are, and how to change/verify/revert them.

## Hardware

- 2× **AMD Radeon AI PRO R9700** (RDNA4 / Navi 48), PCI vendor:device `1002:7551`,
  subsystem `1043:0626`, 32 GB GDDR6, vbios `115-G287BP00-100`.
- Stock TDP **300 W**; spec Boost **2920 MHz** / Game **2350 MHz** (marketing figures,
  not hard caps — see Findings).
- Device mapping (verified live via `readlink -f /sys/class/drm/cardX/device`):
  - `card0` = PCI `0000:09:00.0`  (bridge `0000:08:00.0`)
  - `card1` = PCI `0000:0c:00.0`  (bridge `0000:0b:00.0`)
- LACT keys GPUs by the full string `1002:7551-1043:0626-0000:09:00.0` /
  `1002:7551-1043:0626-0000:0c:00.0`.
- **Do not assume these PCI IDs are stable.** An earlier revision of this doc used
  `05:00.0` / `09:00.0`; only `09:00.0` still matched, so the `-450`/`-65mV` block keyed
  to `05:00.0` silently matched nothing and one GPU ran stock (the other undervolted).
  Always re-check `readlink -f /sys/class/drm/cardX/device` and the
  `could not find GPU with id ...` warning in `journalctl -u lactd` after any hardware change.
- OS/kernel: `6.17.0-42-generic`; LACT package `lact 0.10.1-0`.
- **Quirk:** `card0` is enumerated at **PCIe 8.0GT/s x1** while `card1` is **x16**.
  Link-width anomaly, independent of the clock/power/fan config below.

## Where GPU settings are defined (the map)

| Concern | Location | Notes |
|---|---|---|
| Overdrive enable (kernel) | `/etc/default/grub` line 10 | `amdgpu.ppfeaturemask=0xfffd7fff` (sets the 0x4000 overdrive bit). Required for `pp_od_clk_voltage` and `gpu_od/fan_ctrl`. Takes effect at boot; `update-grub` after edits. |
| Persistent per-GPU settings | `/etc/lact/config.yaml` | Main source of truth. Managed by LACT. |
| LACT daemon | `lactd.service` | enabled + running. **Watches `config.yaml` and reloads/reapplies on any file change** (no restart needed). Config backups: `config.yaml.bak`, `.bak2`, `.bak3`. |
| Power cap (script) | `/usr/local/bin/r9700-power-cap` | Writes `power1_cap = 215000000` µW for every `0x7551` device. |
| Power cap (service) | `/etc/systemd/system/r9700-power-cap.service` | enabled oneshot; runs the script above. Redundant with the drop-in. |
| Power cap (drop-in) | `/etc/systemd/system/lactd.service.d/powercap.conf` | `ExecStartPost=/usr/local/bin/r9700-power-cap` — **second, duplicate invocation** of the same script. |
| Fan regulator (lm-sensors) | `fancontrol.service` + `/etc/fancontrol` | Service is `enabled` but **inactive/dead**: `ConditionFileNotEmpty=/etc/fancontrol` fails because `/etc/fancontrol` does not exist. No lm-sensors fan control is running. |
| Runtime GPU knobs | amdgpu sysfs (see below) | Volatile; LACT re-asserts them on start/reload. |

## Effective per-GPU settings (both GPUs are identical, 1:1)

From `/etc/lact/config.yaml`:

```yaml
gpus:
  1002:7551-1043:0626-0000:09:00.0:   # card0
    fan_control_enabled: true
    fan_control_settings:
      mode: curve
      static_speed: 0.5
      temperature_key: junction       # hotspot, not edge
      interval_ms: 500
      curve:                          # must be exactly 5 points on RDNA3+
        45: 0.12
        55: 0.18
        65: 0.28
        75: 0.45
        85: 1.00
      spindown_delay_ms: 3000
      change_threshold: 3
    pmfw_options:
      minimum_pwm: 12
      acoustic_target: 2000
      acoustic_limit: 6500            # was 2600 (this was the cap limiting cooling)
      target_temperature: 85
    power_cap: 210.0
    performance_level: auto
    voltage_offset: -15
    gpu_clock_offsets:
      0: -500
  1002:7551-1043:0626-0000:0c:00.0:   # card1
    # ... identical ...
```

- **`gpu_clock_offsets: {0: -500}`** — RDNA4 SCLK offset (MHz), applied to both cards.
  This is the deliberate cap at the OD floor for stability. (Was `-450`.)
- `voltage_offset: -15` — VDDGFX undervolt (mV); progressively softened `-65 → -50 → -15`
  to maximise voltage headroom while chasing the crash (see Findings). Note `power1_cap_min`
  is **210 W**, so the power cap cannot go below ~210 W per card.
- `power_cap: 210.0` — W, at the **hardware floor** (`power1_cap_min` = 210 W). Also enforced
  by `r9700-power-cap`.
- `performance_level: auto` — SMU picks DPM states; nothing is pinned.
- **`fan_control_enabled: true` + `fan_control_settings`** — a **quiet-below-85 °C** manual
  curve driven off the **junction (hotspot)** sensor: `45→12% … 75→45%`, then full 100% at
  **85 °C**. This replaced the firmware auto curve, which with `acoustic_limit: 2600` capped
  the fans at ~2300 RPM and let the card thermally throttle (TEMP_HOTSPOT) at ~97 °C under
  load. Equilibrium under 100% load is ~74–78 °C junction at only ~2.8–3.1k RPM (much
  quieter than the first 85 °C curve, which held ~68–77 °C at ~4.5–5.7k RPM).
  `pmfw_options` are mostly ignored while a custom curve is active.
- `minimum_pwm: 12` — the hardware minimum (range 12–100). Must not be 0 (see Findings).

## Runtime sysfs interfaces (per card)

Base: `/sys/class/drm/cardX/device/`

### Clock / voltage — `pp_od_clk_voltage`
Exposes only:
```
OD_SCLK_OFFSET:  SCLK_OFFSET: -500Mhz  1000Mhz      # GFX clock offset
OD_MCLK:         MCLK: 97Mhz  1500Mhz               # VRAM clock (absolute)
OD_VDDGFX_OFFSET: VDDGFX_OFFSET: -200mv  0mv        # core voltage offset
```
- Write the value, then write `c` to **commit**; write `r` to **reset to default**.
- **There is no absolute GFX-max-SCLK knob** on this ASIC (see Findings).

### Fan / thermal — `gpu_od/fan_ctrl/`
```
fan_minimum_pwm                 (12..100)
acoustic_limit_rpm_threshold    (500..6500)
acoustic_target_rpm_threshold   (500..6500)
fan_target_temperature          (25..105)
fan_curve                       (anchor points, manual mode)
fan_zero_rpm_enable             (0/1)
fan_zero_rpm_stop_temperature
```
- Write value + `c` to commit; `r` to reset. Acoustic/min-pwm/target-temp act in **auto**
  fan mode; `fan_curve` switches to **manual**. Do not drive `pwm1` and `fan1_target` at
  the same time.

### DPM / power / sensors
```
pp_dpm_sclk / pp_dpm_mclk / pp_dpm_socclk / pp_dpm_fclk / pp_dpm_pcie
power_dpm_force_performance_level           (auto/low/high/manual/...)
hwmon/hwmon*/pwm1                          (0..255)
hwmon/hwmon*/fan1_input, fan1_target
hwmon/hwmon*/temp1_input (edge), temp2_input (junction), temp3_input (mem)
hwmon/hwmon*/power1_cap, power1_cap_default (300 W), power1_cap_max (330 W)
hwmon/hwmon*/power1_average
hwmon/hwmon*/freq1_input (GFX Hz), freq2_input (mem Hz)
```

## Findings / quirks (as of 2026-10-02)

- **Fans: LACT manual curve (since 2026-10-05), hotspot-driven.** `fan_control_enabled: true`
  with a 5-point curve on the `junction` sensor. The old firmware auto mode (with
  `acoustic_limit 2600`) capped fans at ~2300 RPM, which caused hotspot throttling at ~97 °C
  under load. `fancontrol.service` (lm-sensors) remains dead and unused.
  A LACT custom curve is a **runtime** setting: fans keep this curve until LACT reapplies it,
  and it is defined entirely in `/etc/lact/config.yaml` (no scripts).
- **`minimum_pwm` must be 12–100.** It was set to `0`, which the kernel rejects
  (`Value 0 is out of range, should be between 12 and 100`), which aborted `apply_config`
  on **every** apply and skipped its commit step. Fixed to `12` (the already-effective
  hardware minimum, so no behavior change) so the config now applies cleanly.
- **Zero-RPM reset gap in LACT 0.10.1.** `reset_pmfw_settings()` resets target-temp,
  acoustic target/limit and min-pwm, but **not** `zero_rpm`/`zero_rpm_threshold`; the
  `amdgpu-sysfs` crate exposes no reset functions for them and LACT `master` still omits
  them. Not triggered here (zero-RPM is unset). Workaround/proper fix documented separately.
- **Why the GPU boosted to ~3.3 GHz.** With `performance_level: auto` under sustained
  ~100% load, the SMU runs its top **dynamic** boost state. The `-65 mV` undervolt gives
  frequency headroom, so it sat above the 2920 MHz marketing boost (3.24–3.31 GHz on
  card0). The spec number is not a hard cap.
- **SCLK offset cap floor ~3.03 GHz.** `OD_SCLK_OFFSET` bottoms out at `-500` → ~3.03 GHz
  measured. Absolute max-SCLK writes (`s 1 <clk>`) are **rejected** (`EINVAL`), and
  `rocm-smi --setperfdeterminism` is **unsupported** on this card. Therefore a precise
  2.8 GHz cap is **not reachable** through the overdrive interface. `-500` was chosen
  as the stability setting; it lands ~2.9–3.0 GHz under load.
- Observed effect of `-500` / `-50mV`: downclocked further and with more voltage headroom
  than `-450`/`-65`, used while diagnosing the recurring hard crashes.

## Change log

- **2026-10-05**
  - Stability tuning while diagnosing recurring hard crashes: `gpu_clock_offsets` `-450 → -500`
    (OD floor) and `voltage_offset` `-65 → -50 → -15` on both GPUs; `power_cap` `215 → 210`
    (hardware floor). Under sustained load `-500` lands ~2.78–2.84 GHz (peaks ~2.91–2.93).
    A true 2.8 GHz cap is not selectable: `max_core_clock` is rejected (`GPU does not report
    allowed OD ranges`), `rocm-smi --setperfdeterminism/--setextremum/--setsrange` are unsupported,
    and the only coarser levers are `--setsclk 0 2` (~2.30 GHz) / `profile_standard` (~1.58 GHz).
    Backups: `config.yaml.pre-500-50-<ts>`, `config.yaml.pre-15mv-<ts>`, `config.yaml.pre-210w-<ts>`.
  - Rewrote fan control: enabled a **manual LACT curve** (`fan_control_enabled: true`,
    `temperature_key: junction`, `spindown_delay_ms: 3000`, `change_threshold: 3`) on both
    GPUs, targeting **85 °C**, and raised `acoustic_limit` `2600 → 6500`. Final curve is
    **quiet below 85 °C**: `45→12% / 55→18% / 65→28% / 75→45% / 85→100%`. Under 100% load
    junction settles ~74–78 °C at ~2.8–3.1k RPM with no throttling.
    Pre-change backups: `config.yaml.pre-fancurve-20261005-181219` (firmware auto),
    `config.yaml.pre-fan85-20261005-181816` (first 85 °C curve), and
    `config.yaml.pre-fanquiet-<ts>` (pre-this-curve).
  - Fixed stale PCI keys: `05:00.0`/`09:00.0` → `09:00.0`/`0c:00.0`, so the
    `voltage_offset: -65` / `gpu_clock_offsets: {0: -450}` block now applies to
    **both** GPUs (previously only the GPU enumerated at `09:00.0` got it).
  - Pre-change backup: `/etc/lact/config.yaml.pre-od-disable-20261005-174058`.
  - Applied live by LACT's config watcher; no restart performed. Verified both cards
    report `-450Mhz` / `-65mV`, and no `Failed to upload overdrive table` errors.
  - Undervolt retained deliberately (host needs it).
- **2026-10-02**
  - Added `gpu_clock_offsets: {0: -450}` to **both** GPUs (identical, 1:1).
  - Changed `minimum_pwm: 0 → 12` on both (fixes the apply error; idempotent).
  - Pre-change backup: `/etc/lact/config.yaml.bak3`.
  - Applied live by LACT's config watcher; no restart performed.

## Verify

```bash
# clock offset actually applied
cat /sys/class/drm/card0/device/pp_od_clk_voltage | sed -n '2p'   # -> -450Mhz
cat /sys/class/drm/card1/device/pp_od_clk_voltage | sed -n '2p'   # -> -450Mhz

# live GFX clock (Hz) + busy + junction temp
for c in 0 1; do
  d=/sys/class/drm/card$c/device
  echo "card$c sclk=$(cat $d/hwmon/hwmon*/freq1_input) busy=$(cat $d/gpu_busy_percent) junc=$(cat $d/hwmon/hwmon*/temp2_input)"
done

# power cap + min pwm
for c in 0 1; do
  echo "card$c cap=$(cat /sys/class/drm/card$c/device/hwmon/hwmon*/power1_cap) minpwm=$(sed -n '2p' /sys/class/drm/card$c/device/gpu_od/fan_ctrl/fan_minimum_pwm)"
done

# fan curve (5 points) + live mode/speed
for c in 0 1; do
  echo "-- card$c --"
  cat /sys/class/drm/card$c/device/gpu_od/fan_ctrl/fan_curve
  echo "fan=$(cat /sys/class/drm/card$c/device/hwmon/hwmon*/fan1_input) rpm"
done
lact cli -g 1002:7551-1043:0626-0000:09:00.0 stats | grep -iE 'temp|fan|throttl'

# LACT apply status (should have no errors after a reload)
journalctl -u lactd --no-pager -n 20 | grep -iE 'error|reload'
```

## Change it

- Preferred: edit `/etc/lact/config.yaml` (LACT reloads automatically), or the LACT GUI
  (OC page → RDNA4 GPU clock offset; note the GUI's 5 s confirm-or-revert window).
- The offset is **relative per card** and shifts the whole SCLK curve; keep both GPUs
  identical if you want 1:1 behavior.
- Fan curve: edit `fan_control_settings.curve` (temp→speed, 5 points, RDNA3+), or set
  `temperature_key: edge` for a less aggressive curve. Set `fan_control_enabled: false` to
  hand control back to the firmware auto curve.

## Revert

Do **not** blindly restore the old backups: `config.yaml.bak3`,
`config.yaml.pre-od-disable-20261005-174058` etc. contain the stale `05:00.0` PCI key
and will re-introduce the partial-apply bug. Restore the current PCI keys
(`09:00.0` / `0c:00.0`) after copying.

To drop only the clock cap, remove the `gpu_clock_offsets` block (leave
`minimum_pwm: 12` — `0` will break apply). To drop the undervolt, remove `voltage_offset`.
The watcher reapplies automatically on save.
