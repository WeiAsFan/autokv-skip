# AutoKV-Skip v4.0 数据构造报告

运行：`v4-766ad175b56aab95`；状态：`construction_not_found`。
构造阶段完成：`True`；两批确认通过：`False`。

每批独立要求 BF16 正确率 ≥ 0.8，P32−P0 严格 > 0.1（绝对分差）。
错误答案保留在完整分母内；通信失败表示未完成。没有按单题模型对错筛选数据。
确认通过后才冻结一个生成条件；正式数据全部重新生成。

| 条件 | 状态 | 发现题数 | BF16 | FP8 | 分差 |
|---|---|---:|---:|---:|---:|
| single_lookup-l8192-r16 | discovery_complete | 64 | 0.8594 | 0.8594 | 0.0000 |
| single_lookup-l8192-r64 | discovery_complete | 64 | 1.0000 | 1.0000 | 0.0000 |
| single_lookup-l8192-r128 | confirmation_failed | 64 | 0.8906 | 0.8438 | 0.0469 |
| single_lookup-l8192-r256 | discovery_complete | 64 | 0.7812 | 0.7188 | 0.0625 |
| single_lookup-l16384-r16 | discovery_complete | 64 | 0.9062 | 0.9062 | 0.0000 |
| single_lookup-l16384-r64 | discovery_complete | 64 | 0.8125 | 0.8125 | 0.0000 |
| single_lookup-l16384-r128 | discovery_complete | 64 | 0.7656 | 0.7500 | 0.0156 |
| single_lookup-l16384-r256 | discovery_complete | 64 | 0.7031 | 0.6562 | 0.0469 |
| single_lookup-l24576-r16 | discovery_complete | 64 | 0.8281 | 0.8281 | 0.0000 |
| single_lookup-l24576-r64 | confirmation_failed | 64 | 0.8438 | 0.8281 | 0.0156 |
| single_lookup-l24576-r128 | discovery_complete | 64 | 0.7188 | 0.6719 | 0.0469 |
| single_lookup-l24576-r256 | discovery_complete | 64 | 0.7188 | 0.7344 | -0.0156 |
| two_hop_lookup-l8192-r16 | discovery_complete | 64 | 0.1875 | 0.2031 | -0.0156 |
| two_hop_lookup-l8192-r64 | discovery_complete | 64 | 0.0312 | 0.0312 | 0.0000 |
| two_hop_lookup-l8192-r128 | discovery_complete | 64 | 0.0156 | 0.0156 | 0.0000 |
| two_hop_lookup-l8192-r256 | unconstructible | — | — | — | — |
| two_hop_lookup-l16384-r16 | discovery_complete | 64 | 0.0781 | 0.0938 | -0.0156 |
| two_hop_lookup-l16384-r64 | discovery_complete | 64 | 0.0312 | 0.0156 | 0.0156 |
| two_hop_lookup-l16384-r128 | discovery_complete | 64 | 0.0156 | 0.0000 | 0.0156 |
| two_hop_lookup-l16384-r256 | discovery_complete | 64 | 0.0156 | 0.0312 | -0.0156 |
| two_hop_lookup-l24576-r16 | discovery_complete | 64 | 0.0469 | 0.0469 | 0.0000 |
| two_hop_lookup-l24576-r64 | discovery_complete | 64 | 0.0312 | 0.0156 | 0.0156 |
| two_hop_lookup-l24576-r128 | discovery_complete | 64 | 0.0000 | 0.0000 | 0.0000 |
| two_hop_lookup-l24576-r256 | discovery_complete | 64 | 0.0000 | 0.0000 | 0.0000 |

## single_lookup-l8192-r16：发现

题数 64；BF16 0.859375；FP8 0.859375；分差 0.000000。
BF16 95% Wilson 区间：`[0.7538371825842297, 0.9242142744396914]`；配对分差区间：`[-0.046875, 0.046875]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 9, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 9, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r64：发现

题数 64；BF16 1.000000；FP8 1.000000；分差 0.000000。
BF16 95% Wilson 区间：`[0.9433759402071946, 1.0]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 0, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 0, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r128：发现

题数 64；BF16 0.890625；FP8 0.843750；分差 0.046875。
BF16 95% Wilson 区间：`[0.7910135867310387, 0.945998866555832]`；配对分差区间：`[0.0, 0.109375]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 7, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 10, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r128：确认 batch-1

题数 512；BF16 0.876953；FP8 0.878906；分差 -0.001953。
BF16 95% Wilson 区间：`[0.8456598964509836, 0.9026320320079368]`；配对分差区间：`[-0.01953125, 0.015625]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 63, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 62, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r128：确认 batch-2

题数 512；BF16 0.875000；FP8 0.853516；分差 0.021484。
BF16 95% Wilson 区间：`[0.8435314068192065, 0.9008833613886315]`；配对分差区间：`[0.001953125, 0.039111328124999734]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 64, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 75, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r128：确认合并（仅供描述，不代替逐批判据）

题数 1024；BF16 0.875977；FP8 0.866211；分差 0.009766。
BF16 95% Wilson 区间：`[0.8543720751327486, 0.8947706972199725]`；配对分差区间：`[-0.0029296875, 0.0224609375]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 127, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 137, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r256：发现

题数 64；BF16 0.781250；FP8 0.718750；分差 0.062500。
BF16 95% Wilson 区间：`[0.6656721604337418, 0.8649768059328052]`；配对分差区间：`[0.0, 0.140625]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 14, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 18, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l16384-r16：发现

题数 64；BF16 0.906250；FP8 0.906250；分差 0.000000。
BF16 95% Wilson 区间：`[0.810171204003544, 0.9563217474148016]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 6, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 6, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l16384-r64：发现

题数 64；BF16 0.812500；FP8 0.812500；分差 0.000000。
BF16 95% Wilson 区间：`[0.7002563943589847, 0.8893535682705119]`；配对分差区间：`[-0.078125, 0.078125]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 12, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 12, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l16384-r128：发现

题数 64；BF16 0.765625；FP8 0.750000；分差 0.015625。
BF16 95% Wilson 区间：`[0.6486674241743136, 0.8525010440607586]`；配对分差区间：`[-0.046875, 0.078125]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 15, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 16, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l16384-r256：发现

题数 64；BF16 0.703125；FP8 0.656250；分差 0.046875。
BF16 95% Wilson 区间：`[0.5822979889645707, 0.8009484867446022]`；配对分差区间：`[-0.03125, 0.125]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 19, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 22, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r16：发现

题数 64；BF16 0.828125；FP8 0.828125；分差 0.000000。
BF16 95% Wilson 区间：`[0.7178678837734179, 0.9012225769875536]`；配对分差区间：`[-0.078125, 0.078125]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 11, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 11, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r64：发现

题数 64；BF16 0.843750；FP8 0.828125；分差 0.015625。
BF16 95% Wilson 区间：`[0.7357193822745122, 0.912851576617934]`；配对分差区间：`[-0.046875, 0.078125]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 10, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 11, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r64：确认 batch-1

题数 512；BF16 0.810547；FP8 0.794922；分差 0.015625。
BF16 95% Wilson 区间：`[0.7743386557769042, 0.8421298241452116]`；配对分差区间：`[-0.003955078124999997, 0.037109375]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 97, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 105, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r64：确认 batch-2

题数 512；BF16 0.830078；FP8 0.800781；分差 0.029297。
BF16 95% Wilson 区间：`[0.7951173271721453, 0.8601227552607954]`；配对分差区间：`[0.0078125, 0.05078125]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 87, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 102, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r64：确认合并（仅供描述，不代替逐批判据）

题数 1024；BF16 0.820312；FP8 0.797852；分差 0.022461。
BF16 95% Wilson 区间：`[0.795613743078101, 0.8426169824587628]`；配对分差区间：`[0.0078125, 0.0380859375]`（`estimated`）。
BF16 可解：`True`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 184, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 207, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r128：发现

题数 64；BF16 0.718750；FP8 0.671875；分差 0.046875。
BF16 95% Wilson 区间：`[0.5986606971779259, 0.8140662766627217]`；配对分差区间：`[-0.015625, 0.125]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 18, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 21, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r256：发现

题数 64；BF16 0.718750；FP8 0.734375；分差 -0.015625。
BF16 95% Wilson 区间：`[0.5986606971779259, 0.8140662766627217]`；配对分差区间：`[-0.09375, 0.046875]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 18, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 17, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l8192-r16：发现

题数 64；BF16 0.187500；FP8 0.203125；分差 -0.015625。
BF16 95% Wilson 区间：`[0.11064643172948806, 0.29974360564101526]`；配对分差区间：`[-0.06289062499999998, 0.03125]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 52, 'no_code': 26, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP8：`{'incorrect': 51, 'no_code': 26, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`。

## two_hop_lookup-l8192-r64：发现

题数 64；BF16 0.031250；FP8 0.031250；分差 0.000000。
BF16 95% Wilson 区间：`[0.008612138346171874, 0.10697291770958313]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 62, 'no_code': 32, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 62, 'no_code': 33, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l8192-r128：发现

题数 64；BF16 0.015625；FP8 0.015625；分差 0.000000。
BF16 95% Wilson 区间：`[0.002763541923337505, 0.08334101600094265]`；配对分差区间：`[-0.046875, 0.046875]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 63, 'no_code': 24, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 2}`；FP8：`{'incorrect': 63, 'no_code': 25, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 2}`。

`two_hop_lookup-l8192-r256`：完整记录/问题/模板至少需要 15435 tokens，目标 8192±128。

## two_hop_lookup-l16384-r16：发现

题数 64；BF16 0.078125；FP8 0.093750；分差 -0.015625。
BF16 95% Wilson 区间：`[0.03383117690855046, 0.17019537354162906]`；配对分差区间：`[-0.046875, 0.0]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 59, 'no_code': 33, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP8：`{'incorrect': 58, 'no_code': 33, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 2}`。

## two_hop_lookup-l16384-r64：发现

题数 64；BF16 0.031250；FP8 0.015625；分差 0.015625。
BF16 95% Wilson 区间：`[0.008612138346171874, 0.10697291770958313]`；配对分差区间：`[0.0, 0.046875]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 62, 'no_code': 41, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 63, 'no_code': 41, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l16384-r128：发现

题数 64；BF16 0.015625；FP8 0.000000；分差 0.015625。
BF16 95% Wilson 区间：`[0.002763541923337505, 0.08334101600094265]`；配对分差区间：`[0.0, 0.046875]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 63, 'no_code': 36, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 64, 'no_code': 34, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l16384-r256：发现

题数 64；BF16 0.015625；FP8 0.031250；分差 -0.015625。
BF16 95% Wilson 区间：`[0.002763541923337505, 0.08334101600094265]`；配对分差区间：`[-0.046875, 0.0]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 63, 'no_code': 29, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP8：`{'incorrect': 62, 'no_code': 31, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l24576-r16：发现

题数 64；BF16 0.046875；FP8 0.046875；分差 0.000000。
BF16 95% Wilson 区间：`[0.016069016786292974, 0.12899653740093686]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 61, 'no_code': 27, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP8：`{'incorrect': 61, 'no_code': 24, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l24576-r64：发现

题数 64；BF16 0.031250；FP8 0.015625；分差 0.015625。
BF16 95% Wilson 区间：`[0.008612138346171874, 0.10697291770958313]`；配对分差区间：`[0.0, 0.046875]`（`estimated`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 62, 'no_code': 48, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP8：`{'incorrect': 63, 'no_code': 46, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`。

## two_hop_lookup-l24576-r128：发现

题数 64；BF16 0.000000；FP8 0.000000；分差 0.000000。
BF16 95% Wilson 区间：`[0.0, 0.05662405979280534]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 64, 'no_code': 51, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP8：`{'incorrect': 64, 'no_code': 49, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l24576-r256：发现

题数 64；BF16 0.000000；FP8 0.000000；分差 0.000000。
BF16 95% Wilson 区间：`[0.0, 0.05662405979280534]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`False`；FP8 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 64, 'no_code': 40, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP8：`{'incorrect': 64, 'no_code': 39, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

当前有限范围与预算内未确认满足条件的数据；未完成时只能等待续跑，不能解释为负结论。
完整扫描仍未找到时，可据各条件的 BF16 可解性与分差另立更低位宽方案；本轮不自动改用 FP4。

## 成本与解释边界

实际累计开销（含正式阶段，如已运行）：`{'phases': {'development': {'requests': 0, 'retries': 0, 'server_starts': 0, 'seconds': 0}, 'endpoints': {'requests': 0, 'retries': 0, 'server_starts': 0, 'seconds': 0}, 'search': {'requests': 0, 'retries': 0, 'server_starts': 0, 'seconds': 0}, 'test': {'requests': 0, 'retries': 0, 'server_starts': 0, 'seconds': 0}, 'confirmation': {'requests': 4096, 'retries': 0, 'server_starts': 2, 'seconds': 13718.916357412934}, 'discovery': {'requests': 2944, 'retries': 0, 'server_starts': 2, 'seconds': 10047.740822246997}}, 'timing_incomplete': False, 'total': {'requests': 7040, 'retries': 0, 'server_starts': 4, 'seconds': 23766.65717965993}}`。
背景复用统计：`{'discovery': {'document_uses': 140886, 'unique_documents': 25292, 'reuse_fraction': 0.8204789688116634, 'maximum_samples_per_document': 21}, 'confirmation': {'document_uses': 224083, 'unique_documents': 25366, 'reuse_fraction': 0.8868008728908485, 'maximum_samples_per_document': 23}}`。
同阶段可复用背景；区间以新映射实例为抽样单位，只解释固定背景池条件下的分布。
发现集用于选择条件，确认的两个批次分别判断；没有将发现/确认并入独立测试。
区间未经同时覆盖校正，点判据通过不等于已证明总体性质；退化 bootstrap 不解释为误差为零。
