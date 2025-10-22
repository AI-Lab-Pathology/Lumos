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
