# The prediction file

`ar_134naclh_tww_bag3_wit_zf3.parquet`: taxi-out time in seconds for every departure of `final_submitting.parquet` (January, February,
June and July 2026), the two template columns in the template's order. It is the final-phase chain (README.md, stage 3d), averaged over 3 draws of the chain `ar_134naclh_tww` (`make final-bag`: draw 1 is `make final`, draws 2 and 3 are `pipeline/45_bag_draw.sh`, the average and the witness step `pipeline/46_bag_file.sh`), then the plan-window step (`zf3`, window 3,606 s, `scripts/witness_post.py`); `SHA256` holds its hash, `ar_134naclh_tww_bag3.bag.json` the sha256 of the 3 recorded draws it was averaged from (the draw files themselves are not in this repository).

Check it (template ids and order, finite non-negative seconds, every step of its recipe chain in recipes/, the hash):

```
EC_PHASE=final python scripts/check_final.py final/ar_134naclh_tww_bag3_wit_zf3.parquet --id ar_134naclh_tww_bag3_wit_zf3 --sha256_file final/SHA256
```
