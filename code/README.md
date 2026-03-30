## 1. Clone the LLVM Project

git clone https://github.com/llvm/llvm-project.git

## 2 Build the project
cd llvm-project
mkdir build
cd build
cmake ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_BUILD_EXAMPLES=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_CCACHE_BUILD=On -DCMAKE_INSTALL_PREFIX="../install"

## 3. Run mlir-opt with Seed
./build/bin/mlir-opt your_program.mlir


## 4. Run opt with Seed
./build/bin/opt your_program.ll -S -o your_program_opt.ll