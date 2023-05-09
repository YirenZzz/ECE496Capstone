#ifndef __NeuralNetwork__
#define __NeuralNetwork__

#include <stdint.h>

namespace tflite
{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class AllOpsResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} // namespace tflite

struct TfLiteTensor;

class NeuralNetwork
{
private:
    //tflite::MicroMutableOpResolver<4> *resolver;
    tflite::AllOpsResolver *resolver;
    tflite::ErrorReporter *error_reporter;
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    uint8_t *tensor_arena;

public:
    float getInputScale();
    int getInputZeroPoint();
    float getOutputScale();
    int getOutputZeroPoint();
    int8_t *getInputBuffer();
    float *getInputBuffer_f();
    int getInputSize();
    int getOutputSize();
    int *getInputData();
    int *getOutputData();
    NeuralNetwork();
    int8_t *predict();
    float *predict_f();
    void invoke();
};

#endif