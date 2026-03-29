# 在 Linux 服务器上将你 Fork 并修改后的 Ultralytics 应用到项目（新手版）

> 适用场景：你已经在服务器上装好官方 Ultralytics 环境，现在想改用你自己 GitHub Fork 后的版本。

## 0) 先理解两种常见用法

Ultralytics 通常有两种使用方式：

1. **命令行方式**：例如 `yolo detect train ...`
2. **Python 方式**：例如 `from ultralytics import YOLO`

无论哪种方式，关键点都一样：**让当前 Python 环境安装的是你 Fork 的代码，而不是官方 pip 上的旧版本**。

---

## 1) 激活你原来的环境

先登录服务器并进入你原来的虚拟环境（以下二选一）：

### Conda

```bash
conda activate your_env_name
```

### venv

```bash
source /path/to/venv/bin/activate
```

然后确认 Python/Pip 指向当前环境：

```bash
which python
which pip
python -V
```

---

## 2) 拉取你自己的 Fork 仓库

进入你想放代码的位置（示例用 `~/projects`）：

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/<你的GitHub用户名>/ultralytics.git
cd ultralytics
```

> 如果你已经 clone 过，就用：

```bash
cd ~/projects/ultralytics
git pull
```

---

## 3) 切到你修改所在分支

```bash
git branch -a
git checkout <你的分支名>
git pull origin <你的分支名>
```

> 如果你的修改在 `main` 分支，就把 `<你的分支名>` 换成 `main`。

---

## 4) 卸载已安装的官方 ultralytics（关键）

```bash
pip uninstall -y ultralytics
```

如果之前装过多次，执行 2 次也可以，直到提示未安装。

---

## 5) 用“可编辑安装”安装你 Fork 的代码（最推荐）

在仓库根目录（有 `pyproject.toml` 的地方）执行：

```bash
pip install -e .
```

说明：

- `-e`（editable）表示源码改了立即生效，不用每次重新 `pip install`。
- 开发调试阶段建议一直用这种方式。

---

## 6) 验证当前用的是不是你 Fork 的版本

### 方法 A：看导入路径

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

输出路径应当指向你服务器上的仓库目录，例如：

`/home/xxx/projects/ultralytics/ultralytics/__init__.py`

### 方法 B：看 pip 记录

```bash
pip show ultralytics
```

如果是 editable 安装，通常能看到安装位置与源码路径相关信息。

### 方法 C：跑一个最小命令

```bash
yolo help
```

如果命令正常，说明入口脚本可用。

---

## 7) 在你的项目里使用（两种方式）

### 方式 1：直接在项目环境里用

只要你的项目运行时激活的是这个环境，就会自动使用你 Fork 的 ultralytics。

### 方式 2：项目里固定依赖到你的 GitHub 分支（可选）

在 `requirements.txt` 写：

```txt
git+https://github.com/<你的GitHub用户名>/ultralytics.git@<分支名>
```

然后：

```bash
pip install -r requirements.txt
```

> 注意：这种方式适合部署固定版本；但本地频繁改代码时，还是 `pip install -e .` 更方便。

---

## 8) 你后续更新 Fork 代码的标准流程

```bash
cd ~/projects/ultralytics
git pull
```

如果你是 `pip install -e .`，通常拉最新代码后就可直接生效。

---

## 9) 常见坑与解决

1. **同一台机器多个 Python 环境混用**
   - 现象：你改了代码但运行没变化。
   - 解决：每次先 `which python && which pip` 确认环境。

2. **系统里残留多个 ultralytics 安装**
   - 解决：`pip uninstall ultralytics` 多执行一次，然后重新 `pip install -e .`。

3. **用了 sudo pip 导致权限/路径混乱**
   - 解决：尽量不要 `sudo pip`，用虚拟环境安装。

4. **yolo 命令指向旧环境**
   - 解决：先激活环境，再执行 `which yolo` 检查路径。

---

## 10) 一套可直接复制的“最短指令”

把下面替换成你自己的信息后直接执行：

```bash
# 1) 激活环境
conda activate your_env_name

# 2) 拉代码
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/<你的GitHub用户名>/ultralytics.git
cd ultralytics
git checkout <你的分支名>

# 3) 安装你 Fork 版本
pip uninstall -y ultralytics
pip install -e .

# 4) 验证
python -c "import ultralytics; print(ultralytics.__file__)"
which yolo
yolo help
```

如果你愿意，我下一步可以按你当前服务器实际情况（Conda/venv、仓库路径、分支名）给你生成一份**一键可执行的定制命令清单**。
