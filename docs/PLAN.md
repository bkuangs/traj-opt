# Plan: from prototype to professional

The goal is a planner whose safety and timing claims rest on measured data and enough statistics to back them. Every step stays as simple as possible, and complexity is added only when a metric shows it is needed.

## Where we are

| | today | professional |
|---|---|---|
| Requirements | separation 50 m / 30 m, 2 s window, cost weights: all made up | written down, with a source for each number |
| Error model | constant wind, start delay, cross-track and altitude offsets; σ values invented | fitted to flight logs; gusts and turn overshoot |
| P(loss of separation) | count over 100–200 trials, no confidence bound | upper confidence bound below a stated budget |
| Planner | bnb over a fixed menu of routes; 64-route cap | simple rules first (altitude layer, approach sector, time slot), optimizer only as a fallback |
| In flight | speed loop exists only inside the Monte Carlo replay | speed loop, single-aircraft repair, lost-link hold |
| Validation | Python replay only | Python replay → SITL → hardware-in-the-loop → flight test, with predicted vs actual compared at each stage |

## Steps

Each step lists what it delivers and whether placeholder numbers are enough.

1. **One requirements file** (`requirements.toml`)
   - Contents: separation minimums, arrival window, risk budget (e.g. P(LoS) < 1e-4 per mission), aircraft limits, error σ's, lost-link behaviour.
   - Each value is tagged `source = "placeholder" | "spec" | "measured"`.
   - Bench prints how many values are still placeholders.
   - *Placeholders: fully.*
2. **Report the metrics that matter**
   - Add to bench:
     1. P(LoS) with a Clopper–Pearson 95% upper bound
     2. p95 and p99 of the worst aircraft's arrival error per mission
     3. fraction of missions where every aircraft gets a safe plan
     4. p99 planning time
   - Drop "cost reduction vs greedy" as a headline number.
   - *Placeholders: fully.*
3. **Better-shaped error model**
   - Replace constant offsets with slowly varying ones (first-order Markov gusts and cross-track drift), extra error in turns, and a common altimeter bias.
   - Read the σ values from step 1.
   - *Placeholders: the model's form can be finished; its numbers can't.*
4. **Sensitivity sweep**
   - Vary each placeholder ×0.5 to ×3 and record which ones move P(LoS) and the arrival tail.
   - For each one, find the largest value that still meets requirements (the "allowable envelope").
   - Output: a ranked list of which real numbers to measure first, and the threshold each must beat.
   - *Placeholders: this is exactly what they're for.*
5. **Rare-event estimation**
   - Importance sampling: draw errors from widened distributions and reweight each trial by its likelihood ratio.
   - Check against brute force on a case with P ≈ 1e-2, then use it to reach 1e-4 with a few thousand trials instead of about 30,000.
   - *Placeholders: fully, since correctness is checked against brute force.*
6. **Procedural baseline planner**
   - Rules: assign each aircraft a unique altitude layer away from the arrival point; inside a merge radius, separate by approach sector and a time-slot offset.
   - Run the same verifier and metrics.
   - Keep bnb only for scenarios where the rules can't fit everyone, and report how often that happens.
   - *Placeholders: mostly. Layer spacing will change once real altimeter error is known.*
7. **In-flight logic in the replay**
   - Make the existing speed loop the documented controller.
   - Add the repair ladder: retime → replan one aircraft against the others' reserved cells → full replan.
   - Add the lost-link rule: hold in the aircraft's own layer.
   - Measure how each changes metrics 1–3.
   - *Placeholders: fully for the logic; gains and latencies need real values.*
8. **Simulator cross-check (SITL)**
   - Fly 2–3 plans in ArduPilot or PX4 SITL with a fixed-wing model.
   - Compare SITL arrival and tracking errors with the Python replay.
   - This is a dry run of the sim-to-real comparison, so the pipeline is ready when flight logs arrive.
   - *Placeholders: fully. The SITL airframe model is generic.*
9. **Fit to real data** *(needs real numbers)*
   - Write a log parser (`.bin`/`.ulg` → speed, cross-track and altitude errors, latency).
   - Fit the step 3 model and re-run steps 4–6 with `source = "measured"`.
10. **Flight test** *(needs real aircraft)*
    - 2–3 aircraft, generous separation.
    - Pass criterion: predicted and measured arrival and tracking error distributions agree. If they don't, go back to step 9.

## How far placeholders get us

About **80% of the work** (steps 1–8) can be done now. With placeholders we can:

- build and test all the machinery: metrics, confidence bounds, rare-event sampler, baseline planner, in-flight logic, SITL harness;
- prove the machinery is *correct*, using known-answer tests and brute-force cross-checks;
- answer *"how good must the aircraft be?"* The sensitivity sweep turns into requirements such as "cross-track σ must be < 25 m for P(LoS) < 1e-4 with 50 m separation".
- know which measurements matter most before asking for them.

What placeholders **cannot** give:

- any absolute claim such as "P(LoS) is 3e-5" or "the arrival window is met 99% of the time";
- final separation minimums, layer spacing or speed margin;
- evidence that the error model's *shape* is right, which needs logs (step 9).

**Rule:** while any input is still `placeholder`, results are reported as "meets requirements if X < threshold", never as absolute numbers.

## Real data to request, in likely priority order

Step 4 will confirm the order.

1. Flight logs of the aircraft holding speed and following curved paths: cross-track and speed-tracking error.
2. Altitude error between aircraft flying side by side (baro vs GPS).
3. Wind and gust statistics for the operating area.
4. Command-to-response latency, and datalink latency and loss.
5. Measured speed envelope and turn radius at operating weight.

## Order of work

`1 → 2 → 3 → 4` first. Together they answer "which numbers matter, and how good must they be". After that, 5, 6 and 7 are independent and can be done in any order. Step 8 can start any time. Steps 9 and 10 wait for data.
