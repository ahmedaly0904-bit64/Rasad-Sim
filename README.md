# Rasad · رَصَد

**Measure how much of your simulation's result is a property of the model, and how much is the random seed.**

Stochastic simulations are usually reported without error bars. A paper says *"the civilisation
collapsed in year 240"* and never answers two questions:

1. If the random seed changes, is it still year 240?
2. Is this a property of the model, or an accident of one run?

Rasad answers both by measurement.

---

## Why this exists

This tool was not designed in the abstract. It was built because another project needed it.

**Omran** is a simulation of Ibn Khaldun's theory of *asabiyyah*. Ibn Khaldun (1332–1406) was
a historian who argued in the *Muqaddimah* that civilisations rise and fall on a single
force: **asabiyyah**, the cohesion that lets a group act as one. A group with strong
cohesion overtakes a settled, comfortable one; then comfort erodes its own cohesion over
generations, and it is overtaken in turn. The cycle repeats. He called the study of this
*ʿilm al-ʿumran* — the science of human social organisation — which is where the project
takes its name.

Omran models that computationally: nations on a grid, populations that grow and starve,
ideas that spread between neighbours like an infection, wars along contested borders. Run it
and you get numbers — a final population, a count of wars, a year the collapse happened.

The problem was that nobody knew whether those numbers meant anything.

Change the random seed and they change. Was a collapse in year 240 a property of the model,
or an accident of one run? There was no way to answer without measuring, and no tool that
measured it — the frameworks that do sensitivity analysis are built for engineering and
physics models, and the agent-based modelling field is repeatedly criticised for publishing
results without error bars.

So Rasad was written to answer that question, under one hard constraint: **it must not touch
Omran.** It observes from the outside and modifies nothing, the way an instrument measures a
specimen without altering it. That constraint shaped the architecture — and it is why Rasad
works on any simulation, not just the one it was written for.

What it found is in [`FINDINGS.md`](https://github.com/ahmedaly0904-bit64/rasad/blob/master/FINDINGS.md), and it was not what anyone expected.

---

## Install

```bash
pip install rasad-sim
```

The distribution is `rasad-sim` because `rasad` collides with an existing name on
PyPI. The import is unaffected:

```python
import rasad
```

Python 3.12+ · numpy · plotly

## Use

Describe your simulation as one function:

```python
def run(params: dict, seed: int) -> dict:
    """One run. A number is a final result; a list of numbers is a time series."""
    ...
```

Then:

```python
import rasad

report = rasad.measure(run, params={"growth": 0.03}, runs=100)
print(report.summary())
report.plot().write_html("divergence.html")
```

You get, for every output, two lines: where the **mean** sits — with its standard error and a
90% Student-t confidence interval — and where **one run** lands — standard deviation, cv, a
p05–p95 interval, and a variability class, **low**, **moderate**, or **high**. Plus a curve
showing how far the runs drift apart over time.

---

## What it found in practice

### A simulation that could not reproduce itself

Applied to [Omran](https://github.com/ahmedaly0904-bit64/Omran), an agent-based model of
Ibn Khaldun's theory of *asabiyyah*, **without modifying a line of it**:

| Output | Mean | cv | p05–p95 | Class |
|---|---|---|---|---|
| `final_total_population` | 4,649.39 | 0.2841 | [2,704.75, 6,774.30] | high |
| `survivors` | 1.14 | 0.3059 | [1.00, 2.00] | high |
| `total_wars` | 13.44 | 0.3986 | [6.00, 22.10] | high |
| `total_famines` | 0.00 | 0.0000 | [0.00, 0.00] | low |

No numeric output of the model had low variability. Worse, the measurement exposed something the author
did not know: **the same seed produced different results in different processes.**

Six runs with seed `1`, thirty simulated years:

```
1428 · 1428 · 1512 · 1426 · 1426 · 1512
```

Comparing the population curve year by year located the split: the runs are **identical for
nineteen years**, then diverge at year twenty by **one individual** — which becomes hundreds
by year one hundred. That is error propagation, measured.

This was then confirmed with Rasad out of the path entirely: Omran's own `main.py` carries a
hard-coded `random.seed(42)`, and running that file directly eight times gave **four distinct
results**.

Full write-up: [`FINDINGS.md`](https://github.com/ahmedaly0904-bit64/rasad/blob/master/FINDINGS.md)

### Aggregates can be stable while their parts are noise

On a [SimPy](https://simpy.readthedocs.io) machine-shop simulation
([`examples/simpy_machine_shop.py`](https://github.com/ahmedaly0904-bit64/rasad/blob/master/examples/simpy_machine_shop.py)):

| Output | cv | Verdict |
|---|---|---|
| total parts produced | 0.014 | **low** |
| best machine · worst machine | 0.017 | **low** |
| **gap between best and worst** | **0.31** | **high** |

The shop's total output is stable. The gap between machines is pure noise. Anyone looking at
one run and saying *"machine 7 is underperforming, investigate it"* is chasing a random seed.

**Averages hide fragility.** That alone is a reason to measure what you publish.

---

## Verified beyond the model it was written for

| Framework | Models | Result |
|---|---|---|
| [Mesa](https://github.com/projectmesa/mesa) | Schelling, WolfSheep, Boltzmann | works; WolfSheep's sheep population has high variability (cv 2.95 — usually extinct, occasionally not) |
| [SimPy](https://simpy.readthedocs.io) | machine shop | works; see above |
| [EoN](https://epidemicsonnetworks.readthedocs.io) | SIR on a network | works; epidemic duration has high variability (7.5 → 14.7) |
| [Omran](https://github.com/ahmedaly0904-bit64/Omran) | asabiyyah model | the model it was written for; see above |

Examples: [`examples/`](https://github.com/ahmedaly0904-bit64/rasad/tree/master/examples)

A control worth stating: the SimPy and Mesa examples reproduce byte-identically across
separate processes. That establishes that Omran's non-reproducibility is a bug in Omran,
and that Rasad's own pipeline is deterministic.

---

## The class is a convention. The numbers are the result.

The default cutoffs — 0.05 and 0.20 — **are a choice, not a theory**. An output at cv 0.21
reads *high*; raise the cutoff to 0.25 and the same data reads *moderate*.

So every report prints the thresholds it used and labels them as a convention:

```
thresholds (a convention, not a rule): low < 0.0500 <= moderate <= 0.2000 < high
```

And they belong to the caller:

```python
rasad.measure(run, params={}, runs=100,
              thresholds={"low": 0.01, "moderate": 0.05})
```

**The real result is the interval.** `p05-p95 [7.46, 14.74]` says the duration may double,
without needing a word on top of it.

---

## Limitations

- **Only the seed varies.** Parameters are held fixed, so *"which parameter drives the
  result?"* is not answered yet. Sensitivity analysis is the next version.
- **Execution is sequential.** No parallelism.
- **One value per output name per run.** A model is rejected when its output keys change
  between runs, and when a key keeps its name but changes shape — a number in some runs and
  a sequence in others. Both would otherwise summarise a subset of the runs while reporting
  the full count.

Accepted outputs: real numbers only — Python numbers, numpy integer and float scalars, 1-D
numpy arrays of those, lists and tuples. **Booleans are always rejected** — alone or inside a
list — because an average of ones and zeros means nothing. **Complex numbers are rejected
too**, because converting one to a float silently keeps the real part and drops the rest.

---

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests -v
```

121 tests · 98% line coverage · 577 of 682 mutants killed · linted with ruff

The mutation figure is the weakest of the three, and the gap between it and the coverage
figure is the point: `validate_thresholds` was fully covered and still accepted two equal
cutoffs, because no test passed the one input that separates `<` from `<=`. Coverage says
a line ran. A surviving mutant says nothing checked what it did.

Of the 103 survivors, 31 are in `report` — chart labels and text layout that no test pins
character by character — and 29 are in the Student-t quantile code, almost all at its
numerical guards: `> tiny` becoming `>= tiny` at 1e-300, an iteration cap of 1000 becoming
1001. No input can tell those apart. Many of the rest change only the wording of an error
message that the tests match by substring. The Omran adapter accounts for 11.

To reproduce: `mutmut run` with `OMRAN_SRC` set to an **absolute** path. mutmut runs the
tests from inside its own `mutants/` directory, where the default relative path does not
resolve — the Omran tests then skip, and its 56 mutants count as untested instead.

The statistics are tested against models whose answers are known analytically: a constant
must give a standard deviation of exactly zero; a random walk's divergence must grow as the
square root of time. Those tests caught real bugs — including a collapsed axis that would
have produced plausible, meaningless numbers.

## Notes on the implementation

Two analyses of the code, in Arabic:

- [فئات المدخلات](https://github.com/ahmedaly0904-bit64/rasad/blob/master/docs/input-classes.md) — the five input classes `summarize()` actually
  distinguishes, each with the test that covers it, and why branch order decides which class
  a list falls into.
- [كم تشغيلة تكفي؟](https://github.com/ahmedaly0904-bit64/rasad/blob/master/docs/how-many-runs.md) — the standard error of the mean applied to Omran:
  100 runs pin the mean to 2.8%, and reaching 1% costs 784.

---

## بالعربية

**رَصَد** أداة تقيس صلابة نتائج المحاكاة: أيُّ المخرجات خاصيةٌ في النموذج، وأيُّها أثرٌ للبذرة العشوائية.

تُنشر نتائج المحاكاة العشوائية غالبًا بلا حدود خطأ. يُقال «انهارت الحضارة في السنة 240» دون
الإجابة على سؤالين: هل يظل الرقم 240 لو تغيّرت البذرة؟ وهل هذه خاصية في النموذج أم صدفة في
تشغيلة واحدة؟ يجيب رَصَد عنهما بالقياس لا بالتقدير.

يوصّف المستخدم محاكاته بدالةٍ واحدة تستقبل المعاملات والبذرة وترجّع قاموس مخرجات. يشغّلها
رَصَد مرارًا ببذورٍ مختلفة، ثم يعرض لكل مخرَج متوسطه وانحرافه ومدى تسعين بالمئة وتصنيفًا لتغايره —
**low** أو **moderate** أو **high** — مع منحنى يبيّن اتساع التباعد بين التشغيلات عبر الزمن.

طُبِّق على ثلاثة أطر محاكاة لم يُكتب لأجلها، وعلى عُمران الذي كُتب لأجله، فكشف في عُمران — محاكاة لنظرية العصبية عند ابن خلدون —
أنها **لا تعيد إنتاج نتائجها بالبذرة نفسها**: تشغيلتان متطابقتان تفترقان عند السنة العشرين
بفارق فردٍ واحد، يصير مئاتٍ بحلول السنة المئة. وهذا انتشار الخطأ في صورته المقيسة.

### لماذا كُتب

لم تُصمَّم هذه الأداة في الفراغ، بل كُتبت لأن مشروعًا آخر احتاجها.

**عُمران** محاكاةٌ لنظرية العصبية عند ابن خلدون (1332–1406)، الذي رأى في *المقدمة* أن الحضارات
تنهض وتسقط بقوةٍ واحدة: **العصبية**، أي التماسك الذي يجعل الجماعة تفعل كأنها واحد. جماعةٌ
عصبيتها قوية تغلب جماعةً مستقرةً مترفة، ثم يُفسد الترف عصبيتها هي عبر الأجيال، فتُغلَب بدورها.
وسمّى ابن خلدون دراسة ذلك **علم العمران**، ومنه أخذ المشروع اسمه.

يحاكي عُمران هذا حاسوبيًّا: دولٌ على شبكة، وسكانٌ ينمون ويجوعون، وأفكارٌ تنتقل بين الجيران
كالعدوى، وحروبٌ على الحدود المتنازعة. تشغّله فتخرج لك أرقام — سكانٌ في النهاية، وعدد حروب،
وسنةٌ وقع فيها الانهيار.

والمشكلة أن أحدًا لم يكن يعرف: هل لهذه الأرقام معنى؟ غيّر البذرة العشوائية تتغيّر. فهل الانهيار
في السنة 240 خاصيةٌ في النموذج أم صدفةُ تشغيلةٍ واحدة؟ لا سبيل إلى الجواب إلا بالقياس، ولم تكن
هناك أداةٌ تقيسه.

فكُتب رَصَد لهذا السؤال، بقيدٍ واحدٍ صارم: **ألّا يمسّ عُمران**. يراقبه من خارجه ولا يعدّل فيه
حرفًا، كما يقيس المِجهر عيّنةً دون أن يغيّرها. وهذا القيد هو الذي شكّل بنية الأداة — وهو سبب
عملها على أي محاكاة، لا على التي كُتبت لأجلها وحدها.

### التصنيف

وحدود التصنيف الافتراضية اصطلاحٌ لا قاعدة، ولذلك يعلنها كل تقرير ويتركها بيد المستخدم.
**النتيجة الحقيقية هي المدى**، لا التصنيف الذي يعلوه.

---

## How this was built — full disclosure

Most of the code here was written by AI models under explicit delegation and human review.

| Stage | Owner |
|---|---|
| Specification and architecture | Ahmed, in dialogue with Claude (Opus 5) |
| **Test authoring** | Written into each task brief **before** implementation; the implementer was forbidden from altering a character |
| Implementation | DeepSeek V4 Flash, via `opencode` + AgentRouter — seven tasks |
| Review gates | Claude (Opus 5): every diff read, tests run independently of the implementer's claim |
| Decisions and merges | Ahmed — every commit after review |

**The ordering is what matters, not the tooling: the tests came first and were the
specification.** The model was never asked to write code and then write the thing that proves
it correct. It was given a written contract and held to it.

What the gates caught during implementation: a dead condition in a type check, a deprecated
import, a missing return annotation, unreadable number formatting.

What they did not catch — the logic errors that shipped:

- An output that was constantly zero was classified as *high* variability when it was the most
  stable number in the report. Neither the reference models nor the review found it — **the
  real data did, on the first run against Omran.**
- **Version 0.2.0 went to PyPI accepting numpy complex values and dropping their imaginary
  part**, so every statistic described the real part alone. A plain Python `complex` was
  rejected; a numpy one was not. A later code review of the whole package found it, together
  with a threshold check that raised the wrong exception type, a `classify` that called `nan`
  *high*, and an adapter that left the caller's global random state reseeded.
- **The "90% confidence interval" on the mean was not 90% at small run counts.** It was a
  percentile bootstrap, which cannot reach outside the smallest and largest run, so it held
  the true mean about 50% of the time with two runs, 79% with five and 84% with ten. It is
  now a Student-t interval, which holds close to 90% at every run count, and costs
  microseconds instead of ten thousand resamples per output.

All fixed in 0.2.1, each with a test that failed before the fix.

That is the boundary. Tests prove right the arithmetic they were written for. Real data and a
second reading find the cases nobody thought to write a test for.

---

## License

MIT
