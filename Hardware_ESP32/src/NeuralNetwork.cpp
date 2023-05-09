// Include model headers
#include "NeuralNetwork.h"
#include "model_data.h"

// Include library headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

/*#include "esp32-hal-psram.h"
extern "C"
{
#include "esp32/himem.h"
#include "esp32/spiram.h"
}*/

const int kArenaSize = 90 * 1024;

NeuralNetwork::NeuralNetwork()
{
    error_reporter = new tflite::MicroErrorReporter();

    // Load model from model_data files
    model = tflite::GetModel(best_doubleCNNmodel_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Load operators needed by the model
    resolver = new tflite::AllOpsResolver();
    /*resolver = new tflite::MicroMutableOpResolver<4>();
    resolver->AddDepthwiseConv2D();
    resolver->AddFullyConnected();
    resolver->AddReshape();
    resolver->AddSoftmax();*/

    // Memory allocation
    //tensor_arena = (uint8_t *)ps_malloc(kArenaSize);
    tensor_arena = (uint8_t *)malloc(kArenaSize);
    //esp_spiram_init();
    if (!tensor_arena)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
        return;
    }

    // Build an interpreter to run the model with.
    interpreter = new tflite::MicroInterpreter(
        model, *resolver, tensor_arena, kArenaSize);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    size_t used_bytes = interpreter->arena_used_bytes();
    TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
    TF_LITE_REPORT_ERROR(error_reporter, "INPUT %d\n", input->dims->size);
}

float NeuralNetwork::getInputScale()
{
    return input->params.scale;
}

int NeuralNetwork::getInputZeroPoint()
{
    return input->params.zero_point;
}

float NeuralNetwork::getOutputScale()
{
    return output->params.scale;
}

int NeuralNetwork::getOutputZeroPoint()
{
    return output->params.zero_point;
}

int NeuralNetwork::getInputSize()
{
    return input->dims->size;
}

int NeuralNetwork::getOutputSize()
{
    return output->dims->size;
}

int *NeuralNetwork::getInputData()
{
    return input->dims->data;
}

int *NeuralNetwork::getOutputData()
{
    return output->dims->data;
}

int8_t *NeuralNetwork::getInputBuffer()
{
    //return input->data.f;
    return input->data.int8;
}

float *NeuralNetwork::getInputBuffer_f()
{
    return input->data.f;
    //return input->data.int8;
}

int8_t *NeuralNetwork::predict()
{
    //interpreter->Invoke();
    //return output->data.f;
    return output->data.int8;
}

float *NeuralNetwork::predict_f()
{
    //interpreter->Invoke();
    return output->data.f;
    //return output->data.int8;
}

void NeuralNetwork::invoke()
{
    interpreter->Invoke();
}