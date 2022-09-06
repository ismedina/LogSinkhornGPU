#include <string>

#include <torch/extension.h>

#include "pybind/extern.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("LogSumExpGPU", &LogSumExpGPU<float>); 
  //m.def("LogSumExpGPU_accesor", &LogSumExpGPU_accesor);
}
