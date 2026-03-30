## Getting Started

1. **Download the Model**

   * Inside `code/model/`, there is a `codegen-2B` folder.
   * Download the model weights from [Salesforce/codegen-2B-multi](https://huggingface.co/Salesforce/codegen-2B-multi) and place them in the `codegen-2B` directory.

2. **Install Dependencies**

   * Make sure you have Python installed, then install the required Python packages:

     ```
     torch accelerate deepspeed peft
     ```
   * **MLIR Requirement:**
     FLEX4IR requires a working MLIR installation (with `mlir-opt` and related tools) to compile and test generated programs.

   * **LLVM Requirement:**
     FLEX4IR requires a working LLVM installation (with `opt` and related tools) to compile and test generated programs.
     
    Please ensure MLIR/LLVM is properly installed and available in your system path before running experiments. You may need to configure MLIR/LLVM in the parent directory or according to your environment.

3. **Merge Model Files**

   * After downloading, run the following command to merge the model files:

     ```bash
     bash merge.sh
     ```