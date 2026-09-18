# AutoKV-Skip v4.1 数据构造报告

运行：`v41-40ef467cef6a32d2`；状态：`construction_passed`。
构造阶段完成：`True`；两批确认通过：`True`。

每批独立要求 BF16 正确率 ≥ 0.8，P32−P0 严格 > 0.1（绝对分差）。
错误答案保留在完整分母内；通信失败表示未完成。没有按单题模型对错筛选数据。
确认通过后才冻结一个生成条件；正式数据全部重新生成。

| 条件 | 状态 | 发现题数 | BF16 | FP4 | 分差 |
|---|---|---:|---:|---:|---:|
| single_lookup-l8192-r16 | discovery_complete | 64 | 0.9688 | 0.8750 | 0.0938 |
| single_lookup-l8192-r64 | confirmation_failed | 64 | 0.9688 | 0.8281 | 0.1406 |
| single_lookup-l8192-r128 | confirmation_passed | 64 | 0.8906 | 0.6406 | 0.2500 |
| single_lookup-l8192-r256 | discovery_complete | 64 | 0.7656 | 0.4844 | 0.2812 |
| single_lookup-l16384-r16 | discovery_complete | 64 | 0.8906 | 0.7969 | 0.0938 |
| single_lookup-l16384-r64 | discovery_complete | 64 | 0.8125 | 0.6719 | 0.1406 |
| single_lookup-l16384-r128 | discovery_complete | 64 | 0.7969 | 0.6094 | 0.1875 |
| single_lookup-l16384-r256 | discovery_complete | 64 | 0.6406 | 0.4375 | 0.2031 |
| single_lookup-l24576-r16 | discovery_complete | 64 | 0.7812 | 0.5938 | 0.1875 |
| single_lookup-l24576-r64 | discovery_complete | 64 | 0.7812 | 0.6719 | 0.1094 |
| single_lookup-l24576-r128 | discovery_complete | 64 | 0.7969 | 0.5156 | 0.2812 |
| single_lookup-l24576-r256 | discovery_complete | 64 | 0.7344 | 0.5156 | 0.2188 |
| two_hop_lookup-l8192-r16 | discovery_complete | 64 | 0.1406 | 0.0781 | 0.0625 |
| two_hop_lookup-l8192-r64 | discovery_complete | 64 | 0.0156 | 0.0156 | 0.0000 |
| two_hop_lookup-l8192-r128 | discovery_complete | 64 | 0.0000 | 0.0000 | 0.0000 |
| two_hop_lookup-l8192-r256 | unconstructible | — | — | — | — |
| two_hop_lookup-l16384-r16 | discovery_complete | 64 | 0.0938 | 0.0469 | 0.0469 |
| two_hop_lookup-l16384-r64 | discovery_complete | 64 | 0.0156 | 0.0000 | 0.0156 |
| two_hop_lookup-l16384-r128 | discovery_complete | 64 | 0.0156 | 0.0000 | 0.0156 |
| two_hop_lookup-l16384-r256 | discovery_complete | 64 | 0.0000 | 0.0000 | 0.0000 |
| two_hop_lookup-l24576-r16 | discovery_complete | 64 | 0.1406 | 0.0625 | 0.0781 |
| two_hop_lookup-l24576-r64 | discovery_complete | 64 | 0.0156 | 0.0312 | -0.0156 |
| two_hop_lookup-l24576-r128 | discovery_complete | 64 | 0.0000 | 0.0000 | 0.0000 |
| two_hop_lookup-l24576-r256 | discovery_complete | 64 | 0.0000 | 0.0000 | 0.0000 |

## single_lookup-l8192-r16：发现

题数 64；BF16 0.968750；FP4 0.875000；分差 0.093750。
BF16 95% Wilson 区间：`[0.8930270822904169, 0.9913878616538281]`；配对分差区间：`[0.03125, 0.171875]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 2, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 8, 'no_code': 1, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r64：发现

题数 64；BF16 0.968750；FP4 0.828125；分差 0.140625。
BF16 95% Wilson 区间：`[0.8930270822904169, 0.9913878616538281]`；配对分差区间：`[0.0625, 0.234375]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`True`；两项合并：`True`。
BF16 错误/格式/截断：`{'incorrect': 2, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 11, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r64：确认 batch-1

题数 512；BF16 0.947266；FP4 0.822266；分差 0.125000。
BF16 95% Wilson 区间：`[0.9243620032866211, 0.963507694211269]`；配对分差区间：`[0.09375, 0.16015625]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`True`；两项合并：`True`。
BF16 错误/格式/截断：`{'incorrect': 27, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 91, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r64：确认 batch-2

题数 512；BF16 0.941406；FP4 0.847656；分差 0.093750。
BF16 95% Wilson 区间：`[0.9175865885145797, 0.9586516282300628]`；配对分差区间：`[0.068359375, 0.123046875]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 30, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 78, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r64：确认合并（仅供描述，不代替逐批判据）

题数 1024；BF16 0.944336；FP4 0.834961；分差 0.109375。
BF16 95% Wilson 区间：`[0.928560863765797, 0.9567896853783279]`；配对分差区间：`[0.087890625, 0.1298828125]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`True`；两项合并：`True`。
BF16 错误/格式/截断：`{'incorrect': 57, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 169, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r128：发现

题数 64；BF16 0.890625；FP4 0.640625；分差 0.250000。
BF16 95% Wilson 区间：`[0.7910135867310387, 0.945998866555832]`；配对分差区间：`[0.140625, 0.35976562499999787]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`True`；两项合并：`True`。
BF16 错误/格式/截断：`{'incorrect': 7, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 23, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r128：确认 batch-1

题数 512；BF16 0.869141；FP4 0.705078；分差 0.164062。
BF16 95% Wilson 区间：`[0.8371591247862454, 0.8956241626683451]`；配对分差区间：`[0.123046875, 0.205078125]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`True`；两项合并：`True`。
BF16 错误/格式/截断：`{'incorrect': 67, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 151, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r128：确认 batch-2

题数 512；BF16 0.886719；FP4 0.736328；分差 0.150391。
BF16 95% Wilson 区间：`[0.8563373724144565, 0.9113403572998764]`；配对分差区间：`[0.11328125, 0.185546875]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`True`；两项合并：`True`。
BF16 错误/格式/截断：`{'incorrect': 58, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 135, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r128：确认合并（仅供描述，不代替逐批判据）

题数 1024；BF16 0.877930；FP4 0.720703；分差 0.157227。
BF16 95% Wilson 区间：`[0.8564540735358316, 0.8965803495823582]`；配对分差区间：`[0.1318359375, 0.1865234375]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`True`；两项合并：`True`。
BF16 错误/格式/截断：`{'incorrect': 125, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 286, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l8192-r256：发现

题数 64；BF16 0.765625；FP4 0.484375；分差 0.281250。
BF16 95% Wilson 区间：`[0.6486674241743136, 0.8525010440607586]`；配对分差区间：`[0.171875, 0.390625]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`True`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 15, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 33, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l16384-r16：发现

题数 64；BF16 0.890625；FP4 0.796875；分差 0.093750。
BF16 95% Wilson 区间：`[0.7910135867310387, 0.945998866555832]`；配对分差区间：`[0.015625, 0.171875]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 7, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 13, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l16384-r64：发现

题数 64；BF16 0.812500；FP4 0.671875；分差 0.140625。
BF16 95% Wilson 区间：`[0.7002563943589847, 0.8893535682705119]`；配对分差区间：`[0.015234375000000022, 0.265625]`（`estimated`）。
BF16 可解：`True`；FP4 恢复需求：`True`；两项合并：`True`。
BF16 错误/格式/截断：`{'incorrect': 12, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 21, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l16384-r128：发现

题数 64；BF16 0.796875；FP4 0.609375；分差 0.187500。
BF16 95% Wilson 区间：`[0.6828636426898583, 0.8772658218081634]`；配对分差区间：`[0.09335937500000002, 0.296875]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`True`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 13, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 25, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l16384-r256：发现

题数 64；BF16 0.640625；FP4 0.437500；分差 0.203125。
BF16 95% Wilson 区间：`[0.518208506097791, 0.7471159770854825]`；配对分差区间：`[0.09375, 0.3125]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`True`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 23, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 36, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r16：发现

题数 64；BF16 0.781250；FP4 0.593750；分差 0.187500。
BF16 95% Wilson 区间：`[0.6656721604337418, 0.8649768059328052]`；配对分差区间：`[0.09375, 0.296875]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`True`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 14, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 26, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r64：发现

题数 64；BF16 0.781250；FP4 0.671875；分差 0.109375。
BF16 95% Wilson 区间：`[0.6656721604337418, 0.8649768059328052]`；配对分差区间：`[0.03125, 0.203125]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`True`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 14, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 21, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r128：发现

题数 64；BF16 0.796875；FP4 0.515625；分差 0.281250。
BF16 95% Wilson 区间：`[0.6828636426898583, 0.8772658218081634]`；配对分差区间：`[0.171875, 0.40625]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`True`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 13, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 31, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## single_lookup-l24576-r256：发现

题数 64；BF16 0.734375；FP4 0.515625；分差 0.218750。
BF16 95% Wilson 区间：`[0.6151712627143486, 0.8270362092577739]`；配对分差区间：`[0.109375, 0.328125]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`True`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 17, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 31, 'no_code': 0, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l8192-r16：发现

题数 64；BF16 0.140625；FP4 0.078125；分差 0.062500。
BF16 95% Wilson 区间：`[0.0757857255603086, 0.24616281741577023]`；配对分差区间：`[-0.015625, 0.140625]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 55, 'no_code': 30, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP4：`{'incorrect': 59, 'no_code': 28, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l8192-r64：发现

题数 64；BF16 0.015625；FP4 0.015625；分差 0.000000。
BF16 95% Wilson 区间：`[0.002763541923337505, 0.08334101600094265]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 63, 'no_code': 37, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP4：`{'incorrect': 63, 'no_code': 40, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`。

## two_hop_lookup-l8192-r128：发现

题数 64；BF16 0.000000；FP4 0.000000；分差 0.000000。
BF16 95% Wilson 区间：`[0.0, 0.05662405979280534]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 64, 'no_code': 29, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 2}`；FP4：`{'incorrect': 64, 'no_code': 25, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`。

`two_hop_lookup-l8192-r256`：完整记录/问题/模板至少需要 15435 tokens，目标 8192±128。

## two_hop_lookup-l16384-r16：发现

题数 64；BF16 0.093750；FP4 0.046875；分差 0.046875。
BF16 95% Wilson 区间：`[0.04367825258519838, 0.18982879599645597]`；配对分差区间：`[-0.03125, 0.125]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 58, 'no_code': 30, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 61, 'no_code': 33, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l16384-r64：发现

题数 64；BF16 0.015625；FP4 0.000000；分差 0.015625。
BF16 95% Wilson 区间：`[0.002763541923337505, 0.08334101600094265]`；配对分差区间：`[0.0, 0.046875]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 63, 'no_code': 42, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP4：`{'incorrect': 64, 'no_code': 42, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l16384-r128：发现

题数 64；BF16 0.015625；FP4 0.000000；分差 0.015625。
BF16 95% Wilson 区间：`[0.002763541923337505, 0.08334101600094265]`；配对分差区间：`[0.0, 0.046875]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 63, 'no_code': 42, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP4：`{'incorrect': 64, 'no_code': 37, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l16384-r256：发现

题数 64；BF16 0.000000；FP4 0.000000；分差 0.000000。
BF16 95% Wilson 区间：`[0.0, 0.05662405979280534]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 64, 'no_code': 32, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 64, 'no_code': 31, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l24576-r16：发现

题数 64；BF16 0.140625；FP4 0.062500；分差 0.078125。
BF16 95% Wilson 区间：`[0.0757857255603086, 0.24616281741577023]`；配对分差区间：`[0.0, 0.15625]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 55, 'no_code': 15, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 60, 'no_code': 25, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l24576-r64：发现

题数 64；BF16 0.015625；FP4 0.031250；分差 -0.015625。
BF16 95% Wilson 区间：`[0.002763541923337505, 0.08334101600094265]`；配对分差区间：`[-0.046875, 0.0]`（`estimated`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 63, 'no_code': 51, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 62, 'no_code': 49, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`。

## two_hop_lookup-l24576-r128：发现

题数 64；BF16 0.000000；FP4 0.000000；分差 0.000000。
BF16 95% Wilson 区间：`[0.0, 0.05662405979280534]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 64, 'no_code': 50, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`；FP4：`{'incorrect': 64, 'no_code': 43, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 0}`。

## two_hop_lookup-l24576-r256：发现

题数 64；BF16 0.000000；FP4 0.000000；分差 0.000000。
BF16 95% Wilson 区间：`[0.0, 0.05662405979280534]`；配对分差区间：`None`（`degenerate`）。
BF16 可解：`False`；FP4 恢复需求：`False`；两项合并：`False`。
BF16 错误/格式/截断：`{'incorrect': 64, 'no_code': 46, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`；FP4：`{'incorrect': 64, 'no_code': 47, 'multiple_codes': 0, 'repeated_code': 0, 'extra_code_occurrences': 0, 'empty': 0, 'length_finished': 1}`。

## 冻结规则

条件：`{'condition_id': 'single_lookup-l8192-r128', 'task': 'single_lookup', 'length_bucket': 8192, 'record_count': 128}`。
完整词表、模板、位置规则、种子、来源与两批确认统计见 `construction/selected-rule.json`。

## 成本与解释边界

实际累计开销（含正式阶段，如已运行）：`{'phases': {'development': {'requests': 0, 'retries': 0, 'server_starts': 0, 'seconds': 0}, 'endpoints': {'requests': 1024, 'retries': 0, 'server_starts': 2, 'seconds': 1470.7652548018377}, 'search': {'requests': 24576, 'retries': 0, 'server_starts': 216, 'seconds': 42961.527863507625}, 'test': {'requests': 0, 'retries': 0, 'server_starts': 0, 'seconds': 0}, 'confirmation': {'requests': 4096, 'retries': 0, 'server_starts': 2, 'seconds': 5680.5724268069025}, 'discovery': {'requests': 2944, 'retries': 0, 'server_starts': 2, 'seconds': 10060.84297956014}}, 'timing_incomplete': False, 'total': {'requests': 32640, 'retries': 0, 'server_starts': 222, 'seconds': 60173.708524676505}}`。
背景复用统计：`{'discovery': {'document_uses': 141157, 'unique_documents': 25275, 'reuse_fraction': 0.8209440552009465, 'maximum_samples_per_document': 17}, 'confirmation': {'document_uses': 92603, 'unique_documents': 24720, 'reuse_fraction': 0.7330540047298684, 'maximum_samples_per_document': 13}}`。
同阶段可复用背景；区间以新映射实例为抽样单位，只解释固定背景池条件下的分布。
发现集用于选择条件，确认的两个批次分别判断；没有将发现/确认并入独立测试。
区间未经同时覆盖校正，点判据通过不等于已证明总体性质；退化 bootstrap 不解释为误差为零。
