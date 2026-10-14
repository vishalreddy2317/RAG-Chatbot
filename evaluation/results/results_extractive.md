# Evaluation results — mode `extractive`

Generated 2026-10-14T10:32:14+00:00 · extractive sentence ranking (sentence-transformers/all-MiniLM-L6-v2) · embeddings `sentence-transformers/all-MiniLM-L6-v2` · chunk 800/150 · threshold 0.3

## Summary

| Metric | Value |
|---|---|
| Answerable questions | 30 |
| Recall@1 / @3 / @5 (expected page retrieved) | 86.7% / 96.7% / 100.0% |
| MRR | 0.918 |
| Correct document at rank 1 | 100.0% |
| Answered (evidence found) | 100.0% |
| Keyword hit (answer contains expected fact) | 100.0% |
| Mean grounding score / fully grounded answers | 1.0 / 100.0% |
| Abstained on unanswerable: overall / off-topic / in-domain | 87.5% / 100.0% / 75.0% |
| Follow-up Recall@3 with / without history | 100.0% / 75.0% |
| Follow-up keyword hit | 100.0% |
| Mean latency per question | 0.079 s |

## Answerable questions

| id | question | expected | top-1 (score) | R@1 | R@3 | keyword | grounding |
|---|---|---|---|---|---|---|---|
| q01 | What is the community solar credit rate for 2024? | orion_solar_handbook.pdf p.3 | orion_solar_handbook.pdf p.3 (0.57) | ✓ | ✓ | ✓ | 1.00 |
| q02 | Is there an enrollment fee? | orion_solar_handbook.pdf p.2 | orion_solar_handbook.pdf p.2 (0.39) | ✓ | ✓ | ✓ | 1.00 |
| q03 | How much notice is required to cancel a subscription? | orion_solar_handbook.pdf p.2 | orion_solar_handbook.pdf p.2 (0.58) | ✓ | ✓ | ✓ | 1.00 |
| q04 | What is the maximum residential share size? | orion_solar_handbook.pdf p.4 | orion_solar_handbook.pdf p.4 (0.55) | ✓ | ✓ | ✓ | 1.00 |
| q05 | When is planned maintenance on the array scheduled? | orion_solar_handbook.pdf p.5 | orion_solar_handbook.pdf p.5 (0.59) | ✓ | ✓ | ✓ | 1.00 |
| q06 | What happens if the array produces less than 90 percent of its expected output? | orion_solar_handbook.pdf p.5 | orion_solar_handbook.pdf p.5 (0.38) | ✓ | ✓ | ✓ | 1.00 |
| q07 | Can I keep my subscription if I move to a new address? | orion_solar_handbook.pdf p.6 | orion_solar_handbook.pdf p.6 (0.58) | ✓ | ✓ | ✓ | 1.00 |
| q08 | What are the phone support hours? | orion_solar_handbook.pdf p.7 | orion_solar_handbook.pdf p.7 (0.35) | ✓ | ✓ | ✓ | 1.00 |
| q09 | How was the solar array funded? | orion_solar_handbook.pdf p.8 | orion_solar_handbook.pdf p.8 (0.51) | ✓ | ✓ | ✓ | 1.00 |
| q10 | How many active members does the program have? | orion_solar_handbook.pdf p.1 | orion_solar_handbook.pdf p.1 (0.46) | ✓ | ✓ | ✓ | 1.00 |
| q11 | Within how many days must a billing dispute be submitted? | orion_solar_handbook.pdf p.7 | orion_solar_handbook.pdf p.7 (0.67) | ✓ | ✓ | ✓ | 1.00 |
| q12 | What is the total capacity of the Meadowbrook array? | orion_solar_handbook.pdf p.1 | orion_solar_handbook.pdf p.5 (0.60) | ✗ | ✗ | ✓ | 1.00 |
| q13 | What processing fee applies to credit card payments? | orion_solar_handbook.pdf p.3 | orion_solar_handbook.pdf p.3 (0.56) | ✓ | ✓ | ✓ | 1.00 |
| q14 | Who maintains the Meadowbrook array? | orion_solar_handbook.pdf p.5 | orion_solar_handbook.pdf p.5 (0.66) | ✓ | ✓ | ✓ | 1.00 |
| q15 | Are renters eligible to join the solar program? | orion_solar_handbook.pdf p.1 | orion_solar_handbook.pdf p.1 (0.61) | ✓ | ✓ | ✓ | 1.00 |
| q16 | How long was the typical waiting period in 2023? | orion_solar_handbook.pdf p.2 | orion_solar_handbook.pdf p.2 (0.34) | ✓ | ✓ | ✓ | 1.00 |
| q17 | How much does a non-resident library card cost? | lumen_library_policy.pdf p.1 | lumen_library_policy.pdf p.1 (0.63) | ✓ | ✓ | ✓ | 1.00 |
| q18 | How long can I borrow a DVD? | lumen_library_policy.pdf p.2 | lumen_library_policy.pdf p.2 (0.60) | ✓ | ✓ | ✓ | 1.00 |
| q19 | Does the library charge overdue fines? | lumen_library_policy.pdf p.3 | lumen_library_policy.pdf p.3 (0.76) | ✓ | ✓ | ✓ | 1.00 |
| q20 | How many holds can I place at one time? | lumen_library_policy.pdf p.4 | lumen_library_policy.pdf p.2 (0.31) | ✗ | ✓ | ✓ | 1.00 |
| q21 | How much does 3D printing cost at the library? | lumen_library_policy.pdf p.5 | lumen_library_policy.pdf p.5 (0.67) | ✓ | ✓ | ✓ | 1.00 |
| q22 | How much does color printing cost per page? | lumen_library_policy.pdf p.6 | lumen_library_policy.pdf p.5 (0.51) | ✗ | ✓ | ✓ | 1.00 |
| q23 | What is the interlibrary loan fee? | lumen_library_policy.pdf p.4 | lumen_library_policy.pdf p.4 (0.55) | ✓ | ✓ | ✓ | 1.00 |
| q24 | How long are public computer sessions? | lumen_library_policy.pdf p.6 | lumen_library_policy.pdf p.2 (0.31) | ✗ | ✓ | ✓ | 1.00 |
| q25 | How much does a replacement library card cost? | lumen_library_policy.pdf p.1 | lumen_library_policy.pdf p.1 (0.60) | ✓ | ✓ | ✓ | 1.00 |
| q26 | What is the processing fee for a lost item? | lumen_library_policy.pdf p.3 | lumen_library_policy.pdf p.3 (0.63) | ✓ | ✓ | ✓ | 1.00 |
| q27 | How many branches does the library operate? | lumen_library_policy.pdf p.1 | lumen_library_policy.pdf p.1 (0.64) | ✓ | ✓ | ✓ | 1.00 |
| q28 | How much do meeting rooms cost for for-profit businesses? | lumen_library_policy.pdf p.5 | lumen_library_policy.pdf p.5 (0.65) | ✓ | ✓ | ✓ | 1.00 |
| q29 | How long is a hold kept at the pickup branch? | lumen_library_policy.pdf p.4 | lumen_library_policy.pdf p.4 (0.41) | ✓ | ✓ | ✓ | 1.00 |
| q30 | How many digital items can I borrow per month? | lumen_library_policy.pdf p.6 | lumen_library_policy.pdf p.6 (0.63) | ✓ | ✓ | ✓ | 1.00 |

## Unanswerable questions (should abstain)

| id | category | question | best score | abstained |
|---|---|---|---|---|
| u01 | off-topic | What is the capital of France? | 0.01 | ✓ |
| u02 | off-topic | How do I reset my router password? | 0.19 | ✓ |
| u03 | off-topic | Who won the 2022 FIFA World Cup? | 0.14 | ✓ |
| u04 | off-topic | Write a Python function that sorts a list. | 0.04 | ✓ |
| u05 | in-domain | What is the library's policy on bringing pets? | 0.38 | ✓ |
| u06 | in-domain | Who is the CEO of Verdant Power & Light? | 0.35 | ✓ |
| u07 | in-domain | How many solar panels are installed at the Meadowbrook site? | 0.57 | ✗ |
| u08 | in-domain | What is the library's Wi-Fi password? | 0.50 | ✓ |

## Follow-up questions

| id | turns | retrieval query | R@3 with history | R@3 without | keyword |
|---|---|---|---|---|---|
| f01 | What is the community solar credit rate for 2024? → When are they reconciled each year? | What is the community solar credit rate for 2024? When are they reconciled each year? | ✓ | ✗ | ✓ |
| f02 | How long can I borrow a DVD? → How many of them can I have at once? | How long can I borrow a DVD? How many of them can I have at once? | ✓ | ✓ | ✓ |
| f03 | Is there an enrollment fee? → What about a transfer fee? | What about a transfer fee? | ✓ | ✓ | ✓ |
| f04 | What is the interlibrary loan fee? → How long does it take to arrive? | What is the interlibrary loan fee? How long does it take to arrive? | ✓ | ✓ | ✓ |
