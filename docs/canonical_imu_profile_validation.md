# Canonical deck IMU profile validation

Research date: 2026-08-28

## Decision

The proposed IMU profile is physically plausible as a **canonical, mid-grade MEMS IMU**, but it must not be frozen exactly as originally written. Two corrections are required:

1. `ODR = 200 Hz` does not determine sensor bandwidth or per-sample noise. Freeze an explicit causal low-pass filter and use its equivalent noise bandwidth (ENBW).
2. Keep `5 ms` as an authored one-sample **delivery delay**, not as a claim that total sensor latency is 5 ms. The low-pass filter adds frequency-dependent group delay; with the recommended filter, nominal low-frequency end-to-end delay is about 9.87 ms.

With those corrections, the profile below can be frozen for formal V1 evaluation. It is not a digital twin of any one commercial IMU.

## Primary-source comparison

All three devices are official manufacturer examples. They span vibration-oriented consumer, precision consumer, and calibrated industrial IMUs.

| Property | Bosch BMI088 | TDK ICM-42688-P | Analog Devices ADIS16470 | Proposed value and position |
|---|---:|---:|---:|---|
| Accelerometer output/range | 16 bit; ±3/6/12/24 g | 16 bit; ±2/4/8/16 g; higher-resolution FIFO optional | 16- or 32-bit access; ±40 g | ±16 g, 16 bit exactly matches a standard TDK mode and lies within the device-class envelope |
| Gyroscope output/range | 16 bit; up to ±2000 °/s | 16 bit; up to ±2000 °/s; higher-resolution FIFO optional | 16- or 32-bit access; ±2000 °/s | ±2000 °/s, 16 bit is common across all three |
| Accel noise density | 160 µg/√Hz X/Y, 190 µg/√Hz Z | 65 µg/√Hz X/Y, 70 µg/√Hz Z | 100 µg/√Hz | 150 µg/√Hz is inside the 65–190 µg/√Hz comparison envelope and intentionally closer to the noisier BMI088 |
| Gyro rate-noise density | 0.014 °/s/√Hz | 0.0028 °/s/√Hz | 0.008 °/s/√Hz | 0.005 °/s/√Hz is inside the 0.0028–0.014 envelope |
| Uncalibrated / initial offset evidence | accel 20 mg typical; gyro ±1 °/s | accel ±20 mg; gyro ±0.5 °/s | factory calibrated; over-temperature bias error 1σ is ±4 mg accel and 0.2 °/s gyro; in-run stability is 13 µg and 8 °/h | proposed 2.04 mg accel and 0.05 °/s gyro are credible **post-calibration residuals**, not raw offsets |
| ODR / bandwidth evidence | 200 Hz is supported; at that ODR accel has selectable 20/38/80 Hz cutoff and gyro 23/64 Hz | 200 Hz is supported; filter order, 3 dB BW, NBW and group delay are separately programmable | native 2000 SPS; averaging/decimation can produce 200 SPS | 200 Hz is supported by the class, but a separate bandwidth contract is mandatory |

Sources:

- Bosch, [BMI088 datasheet, revision 1.9](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi088-ds001.pdf), Tables 4, 5, 8 and the gyro bandwidth register table. Bosch specifies 16-bit outputs, the ranges, offsets, noise densities, and separate ODR/BW choices.
- TDK InvenSense, [ICM-42688-P datasheet, DS-000347 revision 1.8](https://invensense.tdk.com/wp-content/uploads/2020/04/ds-000347_icm-42688-p-datasheet.pdf), Tables 1–2 and the UI-filter tables. TDK specifies 65/70 µg/√Hz acceleration noise, 0.0028 °/s/√Hz rate noise, ±20 mg / ±0.5 °/s initial offsets, 16-bit standard output, and explicit NBW/group-delay tables.
- Analog Devices, [ADIS16470 datasheet, revision C](https://www.analog.com/media/en/technical-documentation/data-sheets/ADIS16470.pdf), Table 1 and the filtering/decimation sections. ADI specifies 100 µg/√Hz, 0.008 °/s/√Hz, factory calibration, Allan-variance characteristics, 2000 SPS native conversion and programmable decimation.

The comparison is order-of-magnitude evidence, not a statistical pooling exercise. For example, BMI088's `20 mg typical`, TDK's `±20 mg`, and ADIS16470's `1σ` entries have different test and statistical meanings; none may be relabeled as the standard deviation of a common Gaussian distribution.

## Parameter-by-parameter verdict

| Proposed parameter | Verdict | Reason and provenance |
|---|---|---|
| 200 Hz ODR | Accept | All relevant excitation in the inherited ShakeBench family is well below Nyquist; 200 Hz is also a normal supported ODR in BMI088 and ICM-42688-P. ODR is an authored benchmark choice, not a bandwidth specification. |
| 10 samples per 20 Hz policy step | Accept | Exactly covers one 50 ms control interval with ten new samples. Freeze oldest-to-newest ordering and timestamps. This is an interface choice. |
| 5 ms fixed delay | Accept with rename | Freeze as one 200 Hz sample of delivery delay. Do not report it as total latency. TDK's official 200 Hz, second-order filter table spans 3.3–30 ms group delay depending on bandwidth and includes a 5.1 ms configuration, so 5 ms is plausible but not universal. |
| ±16 g, 16 bit accel | Accept | Directly supported by ICM-42688-P. It is a canonical range choice, not a claim about BMI088 or ADIS16470. |
| 150 µg/√Hz accel noise | Accept | Plausible mid-grade value in the official 65–190 µg/√Hz envelope. Treat as a one-sided amplitude spectral density (ASD). |
| 0.02 m/s² initial accel bias std | Accept as authored residual | Equals 2.039 mg, about 10× smaller than the 20 mg initial-offset scale of BMI088/ICM-42688-P and within ADIS16470's ±4 mg over-temperature 1σ scale. It is a post-calibration residual distribution, not a datasheet zero-g tolerance. |
| 1e-4 m/s²/√s accel bias diffusion | Accept conditionally | Units are correct only for the Brownian update defined below. No reviewed datasheet directly supplies this coefficient. It is authored and should be used only for bounded, sub-minute episodes. |
| ±2000 °/s, 16 bit gyro | Accept | Common supported mode across all three examples. |
| 0.005 °/s/√Hz gyro noise | Accept | Plausible between TDK's 0.0028 and BMI088's 0.014; also near but below ADIS16470's 0.008. Treat as a one-sided ASD. |
| 0.05 °/s initial gyro bias std | Accept as authored residual | 10× smaller than TDK's ±0.5 °/s initial ZRO and 20× smaller than BMI088's ±1 °/s, but above the best industrial in-run scale. It credibly represents residual error after stationary zeroing. |
| 1e-4 °/s/√s gyro bias diffusion | Accept conditionally | Correct as a Brownian bias diffusion coefficient, but not derivable from noise density, angular random walk, or in-run bias stability. Authored benchmark nuisance only. |

## ENBW, discrete noise, and quantization

### Why `sqrt(fs/2)` is not the output-noise rule

For a one-sided white-noise ASD `N` and a causal filter `H`, output variance is

\[
\sigma_y^2=N^2 B_{\mathrm{ENBW}},\qquad
B_{\mathrm{ENBW}}=\int_0^{f_s/2}\frac{|H(f)|^2}{|H(0)|^2}\,df.
\]

Analog Devices' official [MT-048 noise tutorial](https://www.analog.com/media/en/training-seminars/tutorials/MT-048.pdf) states that RMS noise requires integrating noise spectral density over the bandwidth and that practical filter ENBW differs from its 3 dB cutoff. TDK goes further and publishes separate 3 dB BW, NBW and group-delay tables for ICM-42688-P.

`N * sqrt(fs/2)` is appropriate only for constructing a pre-filter, Nyquist-band iid discrete white sequence. It is not the final sample standard deviation when a digital or analog filter is present.

### Frozen filter and calculated noise

Use the same causal, second-order digital Butterworth low-pass for all six channels:

```yaml
filter:
  type: butterworth_lowpass
  order: 2
  sample_rate_hz: 200
  cutoff_3db_hz: 40
  b: [0.20657208382614792, 0.41314416765229584, 0.20657208382614792]
  a: [1.0, -0.36952737735124142, 0.19581571265583306]
  enbw_hz: 40.7618155662
```

Numerical integration of the frozen transfer function over `[0, 100 Hz]` gives `ENBW = 40.7618155662 Hz`. Its group delay is 4.866 ms at DC and 5.148 ms at 9 Hz. A 5 ms delivery delay therefore gives about 9.87 ms nominal low-frequency end-to-end delay (about 10.15 ms at 9 Hz). At 9 Hz, magnitude is 0.999265, so it retains the canonical excitation while rejecting higher-frequency noise.

The corresponding pre-quantization output-noise RMS is:

- accelerometer: `150e-6 g/√Hz * sqrt(40.7618 Hz) = 0.957675 mg = 0.00939158 m/s²`;
- gyroscope: `0.005 °/s/√Hz * sqrt(40.7618 Hz) = 0.0319225 °/s = 5.57153e-4 rad/s`.

Use the conventional standard gravity `g0 = 9.80665 m/s²`, as listed by the official [NIST Guide to the SI](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b8).

For symmetric full-span approximation, the quantization steps are:

- accel: `32 g / 2^16 = 0.48828125 mg = 0.00478840 m/s²`;
- gyro: `4000 °/s / 2^16 = 0.0610352 °/s`.

Thus accel noise RMS is about 1.96 LSB, while gyro noise RMS is about 0.52 LSB. Quantization is secondary for acceleration but material for angular rate. Apply deterministic clip-and-round after bias/noise/filtering; do not add a separate uniform "quantization noise" random variable. The often-used `LSB/sqrt(12)` approximation assumes adequate dithering and is especially questionable for the gyro here. ADIS16470's datasheet likewise compares measured RMS noise with 16-bit quantization noise before deciding whether extra output bits carry useful information.

## Bias semantics and random-walk provenance

Freeze initial residuals as per-axis 1σ Gaussian variables drawn once at episode reset:

\[
b_{a,0}\sim\mathcal N(0,(0.02\ \mathrm{m/s^2})^2),\qquad
b_{g,0}\sim\mathcal N(0,(0.05\ ^\circ/\mathrm{s})^2).
\]

These model **residual bias after a nominal stationary calibration**. They are not raw zero-g/zero-rate offsets and must never be cited as manufacturer specifications. The official ADIS16470 robot documentation illustrates that stationary averaging is an actual offset-calibration procedure, but its result depends on averaging time and motion during calibration ([Analog Devices ADIS16470 IMU calibration guide](https://wiki.analog.com/first/adis16470_imu_frc)).

Freeze bias evolution as independent Brownian diffusion per axis:

\[
b_{k+1}=b_k+q_b\sqrt{\Delta t}\,\epsilon_k,\qquad \epsilon_k\sim\mathcal N(0,1),
\]

with `q_ba = 1e-4 m/s²/√s`, `q_bg = 1e-4 °/s/√s`, and `Δt = 0.005 s`. This produces bias-increment variance `q_b² Δt` and drift standard deviation `q_b sqrt(T)` over duration `T`.

This coefficient is wholly **benchmark-authored**. The reviewed datasheets provide combinations of zero offset, temperature drift, output noise density, angular/velocity random walk, in-run stability and Allan-deviation plots. Those are not interchangeable with Brownian bias diffusion. In particular, ADIS16470's angular random walk and velocity random walk describe integrated white measurement noise, while in-run stability is an Allan-deviation stability statistic; neither directly supplies `q_b` above. Analog Devices' [AN-1041](https://www.analog.com/media/en/technical-documentation/application-notes/AN-1041.pdf) uses Allan variance to relate averaging time and bias accuracy, reinforcing that a stochastic bias process must be identified from a declared model rather than copied from a differently named datasheet field.

At a 25 s PickPlace episode, the authored diffusion contributes only `5e-4 m/s²` accel-bias std and `5e-4 °/s` gyro-bias std, much smaller than the initial residuals. Do not use this unbounded Brownian model for long-duration navigation claims without a separate validation.

## Physical measurement equations

Let `W` be the inertial world frame, `B` the deck/robot-base frame with reference point `O`, and `S` the sensor frame at point `P`. Let `R_SW` map world-frame vectors into sensor coordinates, and let `r_OP^W` point from `O` to `P`. For a sensor rigidly attached to the deck:

\[
a_P^W=a_O^W+\alpha_{WB}^W\times r_{OP}^W
+\omega_{WB}^W\times(\omega_{WB}^W\times r_{OP}^W),
\]

\[
f^S=R_{SW}(a_P^W-g^W),\qquad
\omega^S=R_{SW}\omega_{WB}^W.
\]

The rigid-point acceleration includes tangential (`alpha × r`) and centripetal (`omega × (omega × r)`) terms; a fixed point has no relative/Coriolis term. NASA gives the general translating/rotating-frame relation in its official [rigid-body dynamics derivation](https://ntrs.nasa.gov/api/citations/19940031916/downloads/19940031916.pdf). Even though the canonical sensor is initially mounted at the robot-base origin (`r = 0`), implement the general equation and test a nonzero synthetic lever arm.

The accelerometer output is **specific force**, not inertial acceleration. With world `+z` upward, `g^W = [0, 0, -g0]`; a stationary, aligned IMU reports `+g0` on sensor z, and a freely falling sensor reports zero. This is the official convention in [ROS REP-145](https://ros.org/reps/rep-0145.html) and is consistent with NASA's [accelerometer-output derivation](https://ntrs.nasa.gov/api/citations/20180007917/downloads/20180007917.pdf). Gyro output is the deck's realized angular velocity expressed in `S`; do not use Euler-angle derivatives as angular velocity.

## Final frozen canonical profile

```yaml
deck_imu:
  profile_id: canonical_midgrade_v1
  frame:
    parent: robot_base
    position_m: [0.0, 0.0, 0.0]
    quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]

  output:
    sample_rate_hz: 200
    policy_rate_hz: 20
    window_samples: 10
    window_order: oldest_to_newest
    channels: [specific_force_x, specific_force_y, specific_force_z,
               angular_velocity_x, angular_velocity_y, angular_velocity_z]
    units: [m/s2, m/s2, m/s2, rad/s, rad/s, rad/s]
    delivery_delay_samples: 1       # 5 ms, in addition to filter phase
    timestamp_semantics: acquisition_time

  lowpass:
    type: butterworth
    order: 2
    cutoff_3db_hz: 40.0
    enbw_hz: 40.7618155662
    b: [0.20657208382614792, 0.41314416765229584, 0.20657208382614792]
    a: [1.0, -0.36952737735124142, 0.19581571265583306]

  accelerometer:
    range_m_s2: 156.9064             # ±16 * 9.80665
    bits: 16
    noise_density_m_s2_sqrt_hz: 0.0014709975  # 150 µg/√Hz
    initial_residual_bias_std_m_s2: 0.02
    bias_diffusion_m_s2_sqrt_s: 0.0001

  gyroscope:
    range_rad_s: 34.9065850399       # ±2000 °/s
    bits: 16
    noise_density_rad_s_sqrt_hz: 0.0000872664626  # 0.005 °/s/√Hz
    initial_residual_bias_std_rad_s: 0.000872664626 # 0.05 °/s
    bias_diffusion_rad_s_sqrt_s: 0.00000174532925  # 1e-4 °/s/√s

  omitted_effects:
    - scale_factor_error
    - axis_misalignment
    - temperature_dependence
    - vibration_rectification_error
    - packet_loss
    - timing_jitter
```

### Provenance labels

- **Datasheet-anchored in magnitude/support:** measurement ranges, 16-bit mode, ODR feasibility, noise-density scale, plausibility of millisecond-scale filtering/readout latency.
- **Authored benchmark choices:** exact 200 Hz/20 Hz rates, ten-sample interface, 40 Hz Butterworth coefficients, 5 ms delivery delay, exact 150 µg and 0.005 °/s noise values, residual-bias distributions, both bias-diffusion coefficients, omission of temperature/misalignment/scale error, and episode/reset seeding semantics.
- **Physics/standards anchored:** specific-force sign, gravity subtraction, sensor-frame output, rigid-body lever-arm terms and SI output units.

## V0 smoke mode and formal V1 mode

`ideal_smoke` should exercise the identical physical equations, filter, 200 Hz sampling, ten-sample window and timestamp/delay plumbing, but set bias, diffusion and white noise to zero and bypass quantization. This isolates topology and timing failures. It must be labeled non-scoreable.

`canonical_noisy_v1` should enable the complete frozen pipeline:

```text
realized rigid-body motion
→ specific force / gyro in sensor frame
→ seeded residual bias + seeded Brownian evolution + pre-filter white noise
→ frozen causal 40 Hz low-pass
→ clip to full scale
→ deterministic 16-bit round-to-nearest quantization
→ one-sample delivery queue
→ oldest-to-newest [10, 6] policy window
```

Policy receives only the delayed noisy/quantized window. Clean specific force, clean angular velocity, bias, noise contribution, clipping flags, filter state and delivery timestamps are recorder-only truth. All matched V1–V3 conditions must reuse the same sensor seed and initial residual bias.

Minimum release tests should verify: stationary `+g`, free-fall zero, nonzero lever-arm tangential/centripetal cases, 40 Hz −3 dB response, measured ENBW/noise RMS after warm-up, bias-increment variance `q_b² Δt`, exact quantization bins and clipping, one-sample delivery age, ten new contiguous samples per policy step, deterministic replay, and static-history prefill without artificial zero frames.
