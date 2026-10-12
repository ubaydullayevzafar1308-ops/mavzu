# Error analysis

Model: TahrirchiBERT-small fine-tuned on 20k samples for one epoch, served as ONNX INT8.
Evaluation set: the 4,999-article stratified subsample of the deduplicated test split used by
`make eval`. The model gets 932 of them wrong (accuracy 0.814, macro-F1 0.672).

Sources: confusion rates come from `top_confusions` in `onnx_model.int8/metrics.json`, quoted
headlines and confidences from `onnx_model.int8/errors.csv`. Raw counts (for example "184 articles")
come from re-running the INT8 model on the same subsample; they are not stored in this directory.

## 1. One third of all errors is a single pair: O‘zbekiston ↔ Jamiyat

310 of the 932 errors (33%) are swaps between these two classes: 184 *Jamiyat* articles predicted
as *O‘zbekiston* (34% of the class) and 126 the other way round. Both are general sections for
domestic news, and the label records the section an outlet chose, not a property of the text.

- A weather forecast, "Bugun O‘zbekiston bo‘ylab yog‘ingarchilik kutilmaydi…", is labelled
  *Jamiyat*; the model says *O‘zbekiston* with 0.90 confidence.
- A road accident report, "Jizzaxda sodir bo‘lgan YTHda velosiped haydovchisi vafot etdi", is
  labelled *O‘zbekiston*; the model says *Jamiyat* with 0.87.

This looks like label noise rather than a weak model. The TF-IDF baseline fails on the same
articles: of the 623 articles both models get wrong, they pick the same wrong label in 528 (85%).
In 70% of the transformer's errors the true label is its second choice.

## 2. Topic classes lose to the big "where it happened" classes

79% of all errors are predictions of one of the three largest classes: *O‘zbekiston* (42%),
*Jamiyat* (21%) and *Dunyo* (16%). An article about the economy of Uzbekistan or about a foreign
tech company has two defensible labels, a topic and a geography, and the model picks the geography.

| True class | Predicted as | Rate | Typical example from `errors.csv` |
|---|---|---:|---|
| Iqtisodiyot | O‘zbekiston | 45% | "2017-yilda xorijiy investitsiyalar hajmi 4,5 mlrd dollarga yetishi mumkin" (0.83) |
| Siyosat | O‘zbekiston | 29% | "Prezident Qoraqalpog‘istonga tashrif buyurdi" (0.91) |
| Avto | O‘zbekiston | 32% | "Toshkent chorrahalariga o‘rnatilayotgan kuzatuv kameralari…" (0.87) |
| Texnologiya | Dunyo | 28% | "Sud Uber’ning fransuz taksichilariga 1,2 mln yevro to‘lashi haqida qaror chiqardi" (0.93) |
| Foto | Dunyo | 23% | "Donald Trampning inauguratsiyasi (fotogalereya)" (0.89) |

The class prior pushes in the same direction: *O‘zbekiston* and *Dunyo* are 56% of the training
data, and with a 20k-sample training set the topic classes are seen only a few hundred times each
(about 475 *Iqtisodiyot* articles).

## 3. Jinoyat and Ayollar almost disappear into a neighbouring class

These are the two worst classes, and both are small and have a larger class with the same content.

- *Jinoyat* (crime): 7 of 41 correct, 28 predicted as *Jamiyat* (68%). Most outlets publish crime
  news under their society section, so the same kind of story carries different labels: one
  stabbing report, "Peshku tuman hokimligi hovlisida erkak ayolini pichoqladi", is labelled
  *Jinoyat*, another, "Zarafshonlik erkak ichkilik oqibatida 3 kishini pichoqladi", is labelled
  *O‘zbekiston*. The model predicts *Jamiyat* for both.
- *Ayollar* (women): 6 of 26 correct, 13 predicted as *Salomatlik* (50%). The section is mostly
  lifestyle and health advice, for example "Ertalab nima uchun oyoq og‘riydi va uni qanday davolash
  mumkin?", which is health content by any reading.

Precision on both classes is fine (0.78 and 0.67); the model simply rarely predicts them. The
training subsample holds only about 163 *Jinoyat* and 104 *Ayollar* articles. The baseline, trained
on 50k samples, shows the same confusions at lower rates (47% and 13% on the full test split), so
more data helps here but does not remove the overlap. With 41 and 26 test articles these two rates
are noisy.

## Checked and ruled out: short texts

Short articles are not the problem. The error rate is 15–16% for articles under 1,000 characters
and 22% for those between 2,000 and 3,000. The shortest class, *Pazandachilik* (median 350
characters), is one of the best. Long articles are harder because they belong to the overlapping
general classes, and the model reads only their first 256 tokens.
