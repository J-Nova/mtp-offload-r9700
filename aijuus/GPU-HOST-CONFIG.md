# GPU Host Configuration Map

Where everything for the two AMD GPUs is defined on this host, what the effective
settings are, and how to change/verify/revert them.

## Hardware

- 2× **AMD Radeon AI PRO R9700** (RDNA4 / Navi 48), PCI vendor:device `1002:7551`,
  subsystem `1043:0626`, 32 GB GDDR6, vbios `115-G287BP00-100`.
- Stock TDP **300 W**; spec Boost **2920 MHz** / Game **2350 MHz** (marketing figures,
  not hard caps — see Findings).
- Device mapping:
  - `card0` = PCI `0000:05:00.0`
  - `card1` = PCI `0000:09:00.0`
- LACT keys GPUs by the full string `1002:7551-1043:0626-0000:05:00.0` /
  `1002:7551-1043:0626-0000:09:00.0`.
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
  1002:7551-1043:0626-0000:05:00.0:   # card0
    fan_control_enabled: false
    pmfw_options:
      minimum_pwm: 12
      acoustic_target: 2000
      acoustic_limit: 2600
      target_temperature: 85
    power_cap: 215.0
    performance_level: auto
    voltage_offset: -65
    gpu_clock_offsets:
      0: -450
  1002:7551-1043:0626-0000:09:00.0:   # card1
    # ... identical ...
```

- **`gpu_clock_offsets: {0: -450}`** — RDNA4 SCLK offset (MHz), applied to both cards.
  This is the deliberate cap. Decision: **stick with -450.**
- `voltage_offset: -65` — VDDGFX undervolt (mV).
- `power_cap: 215.0` — W (below the 300 W default). Also enforced by `r9700-power-cap`.
- `performance_level: auto` — SMU picks DPM states; nothing is pinned.
- `fan_control_enabled: false` — custom fan curve disabled; **fans run the amdgpu firmware
  auto curve**. `pmfw_options` (acoustic/target/min-pwm) are the firmware-managed knobs
  that apply in this mode.
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

- **Fans are firmware-controlled.** LACT fan control is off, `fancontrol.service` is dead,
  so the amdgpu PMFW auto curve drives the fans, bounded by the PMFW acoustic settings
  (`acoustic_limit 2600 rpm` caps the top speed).
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
  2.8 GHz cap is **not reachable** through the overdrive interface. `-450` was chosen
  as the agreed setting; it lands ~3.05–3.09 GHz under load.
- Observed effect of `-450`: card0 GFX ~3.28–3.31 → **~3.05–3.09 GHz**, junction
  ~100 °C → **~91 °C** under load.

## Change log

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

# LACT apply status (should have no errors after a reload)
journalctl -u lactd --no-pager -n 20 | grep -iE 'error|reload'
```

## Change it

- Preferred: edit `/etc/lact/config.yaml` (LACT reloads automatically), or the LACT GUI
  (OC page → RDNA4 GPU clock offset; note the GUI's 5 s confirm-or-revert window).
- The offset is **relative per card** and shifts the whole SCLK curve; keep both GPUs
  identical if you want 1:1 behavior.

## Revert

```bash
sudo cp /etc/lact/config.yaml.bak3 /etc/lact/config.yaml   # restore pre-2026-10-02 state
```

The watcher will reapply the restored config automatically. To drop only the cap, remove
the `gpu_clock_offsets` block (leave `minimum_pwm: 12` — `0` will break apply).
