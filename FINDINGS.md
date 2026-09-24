# Measuring Omran — findings

*Full analysis in Arabic below. Summary in English first.*

**Setup:** `scripts/measure_omran.py` — 100 runs × 100 simulated years, seeds `0..99`.
Omran at commit `ccba093`, **with no line of it modified**.

## In one sentence

> **Three of Omran's four numeric outputs have high variability, and the fourth, famines, is low
> only because famines never happen.** Those numbers describe a particular run, not the behaviour
> of the model.

| Output | Mean | cv | Verdict |
|---|---|---|---|
| `final_total_population` | 4,649 | 0.28 | **high** |
| `survivors` | 1.14 | 0.31 | **high** |
| `total_wars` | 13.4 | 0.40 | **high** |
| `total_famines` | 0 | 0 | low |

Final population ranges from **2,704 to 6,774** across 90% of runs — the number can double on
the seed alone.

## Three findings

1. **Famines never happen.** Zero in all 100 runs over 100 years. Low variability by measurement, but
   an open question in the model: either the mechanism is never reached, or its condition is
   unreachable with these parameters.

2. **Omran does not reproduce its own results.** The same seed gives different answers in
   different processes. Comparing the population curve year by year, runs are identical for
   nineteen years and then split at year twenty by **one individual**, which becomes hundreds
   by year one hundred. Strongest suspect — needing confirmation, not proven — is
   `tuple(frozenset)` over `Nation` objects in `grid.py`, whose ordering follows identity
   hashes and therefore memory addresses. The timing supports it: year twenty is roughly when
   national borders first touch, which is the first time that code path runs at all.

### Confirmed without Rasad in the path

A fair objection to the above: perhaps the adapter causes it. It does not.

Omran's own `main.py` carries a hard-coded `random.seed(42)` on line 8. Running that file
directly — Omran's entry point, Omran's seed, no Rasad code involved — eight times:

```
1,938,850  ×5
1,991,944  ×1
1,938,993  ×1
1,936,239  ×1
```

**Four distinct results from eight runs of a program with a fixed seed.** The spread is about
2.9%. This rules out the adapter as the cause and confirms the finding at its source.

(Run with plotly's `show()` stubbed in a separate process, so Omran's files were not touched.)

3. **Divergence grows from 0.82 to 1,320** over one hundred years. Omran's population curve
   carries information in its first decades; after that it describes its seed, not its model.

## A bug in Rasad that this data exposed

`total_famines` was zero in every run: zero standard deviation, zero mean. Since
`cv = std / |mean|`, that gave `inf` and was classified **high** variability — the single most stable
output in the report.

The reference models missed it: the zero-mean test used `[-1, 1]`, whose standard deviation is
not zero. Fixed in `analyzer.summarize`: zero spread means `cv = 0.0` whatever the mean.

**The lesson:** reference models prove the arithmetic is right. Only real data reveals the case
nobody thought to write a test for.

---

القياس: `scripts/measure_omran.py` — 100 تشغيلة × 100 سنة، بذور `0..99`.
عُمران عند `ccba093`، **ولم يُعدَّل منه سطر واحد**.

---

## الإجابة في جملة

> **ثلاثة من مخرجات عُمران العددية الأربعة تغيّرها مرتفع، والرابع، المجاعات، منخفض فقط لأن المجاعة لا تحدث أصلًا.** أرقامه تصف تشغيلة بعينها، لا سلوك النموذج.

| المخرَج | المتوسط | معامل الاختلاف | التصنيف |
|---|---|---|---|
| `final_total_population` | 4,649 | 0.28 | **مرتفع** |
| `survivors` | 1.14 | 0.31 | **مرتفع** |
| `total_wars` | 13.4 | 0.40 | **مرتفع** |
| `total_famines` | 0 | 0 | **منخفض** |

السكان النهائيون يتراوحون بين **2,704 و6,774** في 90% من التشغيلات — أي أن الرقم قد يتضاعف تبعًا للبذرة وحدها. أي جملة من نوع «انتهت المحاكاة بـ 4,600 نسمة» بلا معنى ما لم تُذكر معها هذه السعة.

---

## 1. المجاعات لا تحدث أبدًا

`total_famines` = صفر في **كل** التشغيلات المئة، على مدى مئة سنة.

تغيّره منخفض بمعنى القياس، لكنه سؤال مفتوح في النموذج: إما أن آلية المجاعة لا تُستدعى أصلًا، أو أن شروطها لا تتحقق بهذه المعاملات. الحالتان تستحقان النظر — عدّاد لا يتحرك أبدًا إما كود ميت أو شرط مستحيل.

## 2. عُمران لا يُعيد إنتاج نتائجه بنفس البذرة

**أخطر ما وجده القياس.**

تشغيلتان بالبذرة نفسها في **عمليتين مختلفتين** تعطيان نتيجتين مختلفتين. عيّنة من ست عمليات بالبذرة `1` لثلاثين سنة:

```
1428 · 1428 · 1512 · 1426 · 1426 · 1512
```

داخل العملية الواحدة النتيجة ثابتة. `PYTHONHASHSEED=0` لا يصلحها.

### أين يبدأ الانحراف

بمقارنة منحنى السكان سنةً سنة بين تشغيلات بنفس البذرة:

| المقارنة | أول سنة تختلف | الفارق وقتها |
|---|---|---|
| 1 مقابل 2 | السنة 20 | 1328 ↔ 1327 — **فرد واحد** |
| 1 مقابل 3 | لا تختلف | — |
| 1 مقابل 4 | السنة 23 | 1409 ↔ 1378 |

**فرد واحد في السنة 20 يصير مئات بحلول السنة المئة.** هذا انتشار الخطأ حرفيًا، مقيسًا.

### المرشّح الأقوى للسبب — يحتاج تأكيدًا

في `grid.py` ضمن `spread()`:

```python
for pair, length in border_lengths.items():
    n1, n2 = tuple(pair)
```

`pair` هو `frozenset` من كائني `Nation`. ترتيب `tuple(frozenset)` يتبع تجزئة العناصر، وكائنات `Nation` تُجزَّأ بالتجزئة الافتراضية المشتقة من `id()` — أي من عنوانها في الذاكرة، وهو يختلف بين العمليات.

ويدعم هذا التوقيت: الانحراف يبدأ حوالي السنة 20، وهي تقريبًا اللحظة التي تتلامس فيها حدود الدول لأول مرة بعد توسّع الشبكة — أي أول استدعاء فعلي لهذا المسار.

**لماذا «مرشّح» لا «سبب مؤكد»:** المعالجة داخل الحلقة تبدو متماثلة بين `n1` و`n2`، فلا يظهر من القراءة وحدها كيف يغيّر الترتيب النتيجة. تأكيد السبب يحتاج تتبّعًا فعليًا، لا استنتاجًا من الشكل.

هذا يتقاطع مع بندين مفتوحين في دفتر ديون عُمران: `spread_idea()` المعتمد على ترتيب اللوب، و`random.shuffle(self.nations)` الذي يعدّل قائمة يملكها المستدعي.

### تأكيدٌ من خارج رَصَد

اعتراضٌ وجيه على ما سبق: لعلّ المحوِّل هو سبب التذبذب. وليس كذلك.

يحمل ملف `main.py` في عُمران سطرًا مثبّتًا: `random.seed(42)`. وبتشغيل هذا الملف مباشرةً —
مدخل عُمران نفسه، وبذرته هو، دون أي سطرٍ من رَصَد في الطريق — ثماني مرات:

```
1,938,850  ×5
1,991,944  ×1
1,938,993  ×1
1,936,239  ×1
```

**أربع نتائج مختلفة من ثماني تشغيلات لبرنامجٍ بذرته مثبّتة في كوده.** والفارق نحو 2.9%.

هذا يُخرج المحوِّل من دائرة الاتهام، ويؤكّد النتيجة عند مصدرها. (شُغِّل في عملية منفصلة مع
تعطيل عرض الرسوم في الذاكرة، فلم يُمَسّ أي ملف في عُمران.)

### الأثر على هذا التقرير

الأرقام أعلاه من **قياس واحد**. قياس آخر بنفس البذور أعطى متوسطًا 4,675 بدل 4,649.

**التصنيفات لم تتغيّر** — المرتفع بقي مرتفعًا والمنخفض منخفضًا. الاتجاه موثوق؛ الخانات العشرية ليست.

## 3. منحنى التباعد

التباعد بين التشغيلات ينمو من **0.82** في السنة الأولى إلى **1,320** في السنة المئة.

الرسم: `omran_divergence.html`

عمليًا: منحنى السكان في عُمران يحمل معلومة في عقوده الأولى، وبعدها يصف بذرته لا نموذجه.

---

## بق في رَصَد كشفته هذه البيانات

`total_famines` كان صفرًا في كل تشغيلة: انحرافه صفر، ومتوسطه صفر. وبما أن `cv = std / |mean|`، أعطى `inf` فصُنّف **مرتفع التغيّر** — وهو أثبت مخرَج في التقرير.

النماذج المرجعية لم تمسك هذا: اختبار المتوسط-الصفر فيها كان `[-1, 1]`، وانحرافه ليس صفرًا.

أُصلح في `analyzer.summarize`: انحراف صفر يعني `cv = 0.0` مهما كان المتوسط. أُضيف اختبار `test_cv_is_zero_when_the_value_never_varies_at_zero`.

**الدرس:** النماذج المرجعية تثبت أن الحساب صحيح؛ البيانات الحقيقية وحدها تكشف الحالات التي لم يخطر ببالك أن تكتب لها نموذجًا.

---

## ما لا تقوله هذه النتائج

- **لا تقول إن عُمران خاطئ.** تقول إن مخرجاته الحالية لا تُقاس عليها استنتاجات دون ذكر سعة التغيّر.
- **لا تقيس حساسية المعاملات.** المعاملات ثابتة هنا والبذرة وحدها تتغيّر. سؤال «أي معامل يحكم النتيجة؟» يحتاج النسخة التالية.
- **لم تُختبر نظرية العصبية.** `asabiyyah.py` فارغ، والقياس جرى على ما هو منفَّذ فعلًا.
