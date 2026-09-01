# AutoKV-Skip v1.0 对应源码发布要求

状态：未满足

适用运行：`8181c9a332ef6e9c`

要求冻结日期：2026-09-02

## 1. 为什么必须发布

本次结果归档记录的源码提交是：

```text
2c6a4e1ac96e9835a6af2ad450cb9dfa269e4953
```

归档记录的源码树 SHA-256 是：

```text
64ad8dcaf4f36cf9f3b9575a0a324b656d3ed06f314907226ca9709154542c70
```

归档同时记录 `git_dirty=false`，说明实验声称使用了一个干净提交。但是当前 GitHub 仓库中的结果提交 `096697c...` 只新增了结果归档、校验文件和说明，没有包含或连接到 `2c6a4e1...`；该源码提交目前不能从 GitHub 取得。

因此，v1.0 当前只能证明“这份结果归档内部自洽”，不能证明“面试官或其他开发者可以从公开仓库取得同一源码并复现”。发布该精确提交是 v1.0 证据闭环的硬要求。

## 2. 发布对象

必须发布实验实际使用的精确 Git 对象 `2c6a4e1...`，不能用以下内容替代：

- 当前 `main` 上看起来相似的旧源码；
- 根据结果反向重写的一份“近似修复版”；
- 在 `2c6a4e1...` 之后继续修改过、但没有重新运行实验的代码；
- 只有压缩包、补丁截图或零散文件而没有可检出的 Git 提交。

如果服务器已经丢失该 Git 对象，就不能把重建代码冒充本次运行源码；此时应先公开新源码，再以新提交重跑并产生新的 Run ID。

## 3. 在实验服务器上的发布步骤

以下命令必须在保存本次运行源码 Git 对象的服务器仓库中执行。先核对路径，不要在当前本地工作区执行，也不要 force-push。

```bash
cd /mnt_d/huangxiaoyuan/autokv-skip
git status --short --branch
git cat-file -t 2c6a4e1ac96e9835a6af2ad450cb9dfa269e4953
git show --stat --oneline 2c6a4e1ac96e9835a6af2ad450cb9dfa269e4953
git remote -v
```

成功判据：`git cat-file` 输出 `commit`，`git show` 能展示精确提交，`origin` 指向 `WeiAsFan/autokv-skip`。然后先获取远端状态，再把该对象发布到独立、稳定的证据分支：

```bash
git fetch origin
git push origin 2c6a4e1ac96e9835a6af2ad450cb9dfa269e4953:refs/heads/v1.0/run-8181c9-source
```

该操作不会切换服务器工作区，也不要求改写 `main`。上传后，在 GitHub 上确认提交页面可访问，再通过普通 Pull Request 将确有必要的修复合入 `main`；不要为了制造线性历史而 rebase、squash 或覆盖证据分支。

## 4. 结果说明需要补充的链接

源码分支可访问后，更新 `results/full-20260901/README.zh-CN.md`，至少加入：

```md
- 运行源码：[2c6a4e1](https://github.com/WeiAsFan/autokv-skip/commit/2c6a4e1ac96e9835a6af2ad450cb9dfa269e4953)
- 源码证据分支：`v1.0/run-8181c9-source`
- 源码树 SHA-256：`64ad8dcaf4f36cf9f3b9575a0a324b656d3ed06f314907226ca9709154542c70`
```

如果 `main` 后续包含该提交，也保留证据分支和精确 commit 链接；不要只链接一个会继续移动的分支名。

## 5. 验收清单

- [ ] GitHub 的 `2c6a4e1...` commit 页面能够匿名访问。
- [ ] `v1.0/run-8181c9-source` 分支直接包含该提交，且没有被 force-push 改写。
- [ ] 新克隆仓库后可以执行 `git checkout 2c6a4e1...`。
- [ ] 检出后，manifest 中记录的每个 `source.files` 路径及 SHA-256 都与仓库文件一致。
- [ ] 按 v1.0 源码身份算法复算得到 `source_tree_sha256=64ad8dca...`。
- [ ] `results/full-20260901/README.zh-CN.md` 已链接精确 commit、证据分支和源码树 SHA-256。
- [ ] 没有把当前 `main`、近似修复代码或脏工作区错误标记为本次运行源码。

只有全部项目满足后，本文状态才可改为“已满足”。单纯上传结果压缩包不满足源码发布要求。
