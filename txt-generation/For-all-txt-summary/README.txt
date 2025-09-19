本代码实现了一个 DeepSeek Reasoner 批量润色工具，用于批量处理不同类别的医学文本描述。
主要功能包括：

按类别自动选择 Prompt：支持 ADI、DEB、LYM、MUC、MUS、NOR、STR、TUM 等类别。

Prompt 文件集中管理：从 Deepseek-refine.txt 读取多类别的指令模板。

自动识别类别：根据输入路径中的目录名识别文本类别。

批量并发处理：使用多线程并行调用 API，加快处理速度。

自动重试机制：调用失败时会自动重试，避免因网络波动中断。

编码容错读取：对输入文本与 prompt 文件支持多种编码（UTF-8、GBK、BIG5 等）。

结果保存：生成两类文件：

_opt.txt：优化后的结果

_reason.txt：模型的推理过程（可选，默认保存）



This script implements a DeepSeek Reasoner batch refinement tool, designed to process and refine medical text descriptions across multiple categories.

Key features:

Category-specific prompts: Supports categories such as ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM.

Centralized prompt management: Prompts are stored in a single file Deepseek-refine.txt.

Automatic category detection: The script infers the category from directory names in the input path.

Batch parallel processing: Uses multi-threading to call the API concurrently for efficiency.

Auto-retry mechanism: Retries on failures to ensure robustness.

Encoding fallback: Supports multiple encodings (UTF-8, GBK, BIG5, etc.) to handle diverse text files.

Result saving:

_opt.txt: Optimized result

_reason.txt: Model’s reasoning trace (optional, enabled by default)