# MJSA release policy

The three completed reviewer workbooks contain `Evidence_Context`, `Conditional_Claim`, and `Short_Reason`. Evidence contexts and copyright-sensitive claims may reproduce or closely preserve scientific-paper language; reviewer reasons may contain copied or paraphrased source material. These columns and the original Excel files are therefore not released.

Released fields are limited to anonymous sample IDs, frozen audit/sample membership, release-hashed paper IDs, section label, judge labels for four structured dimensions, overall judgment, confidence, judge identity/run, prompt version, review status, deterministic consensus statistics, and aggregate agreement metrics.

The three source workbooks had identical A–D membership and row order. A content audit found 16 workbook-pair cell mismatches in the sensitive Evidence_Context column, corresponding to eight rows where the ChatGPT workbook differed by one character from the Kimi/Grok copies. Conditional_Claim matched. No sensitive text or hashes of those text cells are published; the discrepancy does not affect released membership or labels.

Formal audit facts retained in the release:

- Maximum frozen T3 pool: 2,286 instances.
- Audit sample: 150 instances from 139 papers.
- Sampling seed: 20260819.
- Judges: GPT-5.6 Sol, Kimi-K3, Grok-4.5.
- Strict majority Accept: at least two Accept and no Reject.
- Original historical bootstrap RNG seed: not recorded in frozen artifacts; the release statistics script requires an explicit reproduction seed and labels it accordingly.
