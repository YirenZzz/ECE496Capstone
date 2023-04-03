/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "constants.h"
#include "main_functions.h"
#include "model.h"
#include "eeg_data.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 100 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.


void setup() {
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
  MicroPrintf("SETUP Model running...\n");

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;

  Serial.print("Model size: ");
  Serial.println(g_model_len);
  Serial.print("First element: ");
  Serial.println(g_model[0]);
}

// The name of this function is important for Arduino compatibility.
void loop() {
//   float position = static_cast<float>(inference_count) /
//                    static_cast<float>(kInferencesPerCycle);
//   float x  = 0.14;
//     // Generate random input data
//   for (int i = 0; i < 7680; i++) {
//     // input->data.int8[i] = x / input->params.scale + input->params.zero_point;
//     input->data.f[i] = sleep_array[i] ;
//   }
//   // MicroPrintf("input->data.f[1]: %f\n", input->data.f[1]);
//   // MicroPrintf("input->data.f[10]: %f\n", input->data.f[10]);
//   // MicroPrintf("input->data.f[100]: %f\n", input->data.f[100]);
//   // MicroPrintf("input->data.f[1000]: %f\n", input->data.f[1000]);
//   MicroPrintf("input->data.f[1]:");
//   Serial.println(input->data.f[1]);
//   MicroPrintf("input->data.f[10]:");
//   Serial.println(input->data.f[10]);
//   MicroPrintf("input->data.f[10] 666666?:");
//   Serial.println(input->data.f[10],6);
//   MicroPrintf("x:");
//   Serial.println(x);
//    MicroPrintf("x: hhhhh");
//   Serial.println(x,6);
  
//   // Run the inference
//   TfLiteStatus invoke_status = interpreter->Invoke();
//   if (invoke_status != kTfLiteOk) {
//     MicroPrintf("Invoke failed");
//     return;
//   }
//   // int8_t y_quantized;
// float y;
// for (int i = 0; i < 5; i++) {
//   int8_t y_quantized = output->data.int8[i];
//   float y_float = (y_quantized - output->params.zero_point) * output->params.scale;
//   y = y_float + 1.0;
//   MicroPrintf("i %d\n", i);
//   MicroPrintf("output->data.int8[i]:");
//   Serial.println(output->data.int8[i]);
//   MicroPrintf("output->data.f[i]:");
//   Serial.println(output->data.f[i],6);

//   MicroPrintf("y_quantized:");
//   Serial.println(y_quantized);
//   MicroPrintf("y_float:");
//   Serial.println(y_float,6);
//   MicroPrintf("y:");
//   Serial.println(y,6);
// }
  
  
}







// //--------------------------------------------------------------------------
// /* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*/

// #include <TensorFlowLite.h>

// #include "constants.h"
// #include "main_functions.h"
// #include "model.h"
// #include "eeg_data.h"
// #include "output_handler.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_log.h"
// #include "tensorflow/lite/micro/system_setup.h"
// #include "tensorflow/lite/schema/schema_generated.h"

// // Globals, used for compatibility with Arduino-style sketches.
// namespace {
// const tflite::Model* model = nullptr;
// tflite::MicroInterpreter* interpreter = nullptr;
// TfLiteTensor* input = nullptr;
// TfLiteTensor* output = nullptr;
// int inference_count = 0;

// constexpr int kTensorArenaSize = 80 * 1024;
// // Keep aligned to 16 bytes for CMSIS
// alignas(16) uint8_t tensor_arena[kTensorArenaSize];
// }  // namespace

// // The name of this function is important for Arduino compatibility.


// void setup() {
//   tflite::InitializeTarget();

//   // Map the model into a usable data structure. This doesn't involve any
//   // copying or parsing, it's a very lightweight operation.
//   model = tflite::GetModel(g_model);
//   if (model->version() != TFLITE_SCHEMA_VERSION) {
//     MicroPrintf(
//         "Model provided is schema version %d not equal "
//         "to supported version %d.",
//         model->version(), TFLITE_SCHEMA_VERSION);
//     return;
//   }

//   // This pulls in all the operation implementations we need.
//   // NOLINTNEXTLINE(runtime-global-variables)
//   static tflite::AllOpsResolver resolver;

//   // Build an interpreter to run the model with.
//   static tflite::MicroInterpreter static_interpreter(
//       model, resolver, tensor_arena, kTensorArenaSize);
//   interpreter = &static_interpreter;

//   // Allocate memory from the tensor_arena for the model's tensors.
//   TfLiteStatus allocate_status = interpreter->AllocateTensors();
//   if (allocate_status != kTfLiteOk) {
//     MicroPrintf("AllocateTensors() failed");
//     return;
//   }
//   MicroPrintf("SETUP Model running...\n");

//   // Obtain pointers to the model's input and output tensors.
//   input = interpreter->input(0);
//   output = interpreter->output(0);

//   // Keep track of how many inferences we have performed.
//   inference_count = 0;

//   Serial.print("Model size: ");
//   Serial.println(g_model_len);
//   Serial.print("First element: ");
//   Serial.println(g_model[0]);
// }

// // The name of this function is important for Arduino compatibility.
// void loop() {
//     // Generate random input data
//   for (int i = 0; i < 7680; i++) {
//     input->data.int8[i] = random(-128, 127) / 127.0f / input->params.scale + input->params.zero_point;
//   }

//   // Run the inference
//   TfLiteStatus invoke_status = interpreter->Invoke();
//   if (invoke_status != kTfLiteOk) {
//     MicroPrintf("Invoke failed");
//     return;
//   }
//   // TfLiteIntArray* outputDims = output->dims;
//   // int numDims = outputDims->size;
//   //  Serial.print("Output tensor dimensions: ");
//   // for (int i = 0; i < numDims; i++) {
//   //   Serial.print(outputDims->data[i]);
//   //   if (i < numDims - 1) {
//   //     Serial.print(" x ");
//   //   }
//   // }
//   // Serial.println();
  
//   // Print the output
//   // for (int i = 0; i < 32; i++) {
//   //   MicroPrintf("Output[%d]: %f\n", i, (output->data.int8[i] - output->params.zero_point) * output->params.scale);
//   //   MicroPrintf("Invoke failed on x: %f\n", static_cast<double>(output->data.int8[i]));
//  int outputSize = output->dims->data[0];

//   // Check if the output tensor is 1-dimensional and has 5 elements
//   if (output->dims->size == 1 && outputSize == 5) {
//     // Cast the output data pointer to a float pointer
//     float* outputData = output->data.f;

//     // Print out the values of the output tensor
//     Serial.print("Output tensor values: ");
//     for (int i = 0; i < outputSize; i++) {
//       Serial.print(outputData[i]);
//       if (i < outputSize - 1) {
//         Serial.print(", ");
//       }
//     }
//     Serial.println();
//   }
  

//   // Increment the inference counter
//   inference_count += 1;
//   if (inference_count >= kInferencesPerCycle) inference_count = 0;

//   // Add a delay between inferences
//   delay(1000);
// }






// //   /*MicroPrintf("LOOP START Model running...\n");

// //   MicroPrintf("input->dims->size = %d\n", input->dims->size);
// //   MicroPrintf("input->dims->data[0] = %d\n", input->dims->data[0]);
// //   MicroPrintf("input->dims->data[1] = %d\n", input->dims->data[1]);
// //   MicroPrintf("input->dims->data[2] = %d\n", input->dims->data[2]);
// //   MicroPrintf("input->dims->data[3] = %d\n", input->dims->data[3]);

// //   MicroPrintf("output->dims->size = %d\n", output->dims->size);
// //   MicroPrintf("output->dims->data[0] = %d\n", output->dims->data[0]);
// //   MicroPrintf("output->dims->data[1] = %d\n", output->dims->data[1]);*/

// //   // for(int i = 0; i < 7680; i++) {
// //   //     float x = (float)rand()/(float)(RAND_MAX/1);
// //   //     // MicroPrintf("%f", x);
// //   //     //int8_t x_quantized = sleep_array[i] / input->params.scale + input->params.zero_point;
// //   //     input->data.f[i] = x;
// //   // }
// //   // for (int i = 0; i < 7680; ++i) {
// //   //   // Generate a random integer between 0 and 32767
// //   //   int rand_int = random(32768);
// //   //   // Scale the random integer to the range of [-1, 1]
// //   //   float rand_float = (float)rand_int / 16384.0 - 1.0;
// //   //   // Set the input tensor value at index i
// //   //   input->data.f[i] = rand_float;
// //   // }

// //   float input_data[7680];
// //   for (int i = 0; i < 7680; i++) {
// //     input_data[i] = (float)random(-32767, 32767) / 32767;
// //   }
// //   input->data.f = input_data;
// //   //MicroPrintf("%f", sleep_array[0]);
// //   //MicroPrintf("%f", sleep_array[1]);
// //   //MicroPrintf("%f", sleep_array[2]);
// //   //MicroPrintf("LOOP END1 Model running...\n");

// //   TfLiteStatus invoke_status = interpreter->Invoke();

// //   if (invoke_status != kTfLiteOk) {
// //     MicroPrintf("Invoke failed");
// //     return;
// //   }

// //   // MicroPrintf("Input dimensions: size=%d, shape=[", input->dims->size);
// //   // for (int i = 0; i < input->dims->size; i++) {
// //   //   MicroPrintf("%d", input->dims->data[i]);
// //   //   if (i < input->dims->size - 1) {
// //   //     MicroPrintf(", ");
// //   //   }
// //   // }
// //   // MicroPrintf("]\n");
// //   // //size=2, shape = [1,5]



// //   MicroPrintf("Output data:\n");
// //   for (int i = 0; i < output->dims->data[0]; i++) {
// //     for (int j = 0; j < output->dims->data[1]; j++) {
// //       MicroPrintf("%f ", output->data.f[i*output->dims->data[1] + j]);
// //     }
// //     MicroPrintf("\n");
// //   }



// //   // int output_size = output->dims->size;
// //   // Serial.print("Output size: ");
// //   // Serial.println(output_size);
// //   // Serial.print("Output dimensions: [");
// //   // for (int i = 0; i < output_size; i++) {
// //   //   Serial.print(output->dims->data[i]);
// //   //   if (i < output_size - 1) {
// //   //     Serial.print(", ");
// //   //   }
// //   // }
// //   // Serial.println("]");
// //   // output->type == kTfLiteFloat32

// //   //MicroPrintf("LOOP END2 Model running...\n");
// //   // float prediction[5];
// //   // for (int i = 0; i < 5; i++) {
// //   //   prediction[i] = output->data.f[i];
// //   // }

// //   // Serial.print("output->data.f [");
// //   // for (int i = 0; i < output->dims->data[0]; i++) {
// //   //   Serial.println(output->data.f[i]);
// //   // }
// //   // MicroPrintf("]\n");


// //   // float* output_data = output->data.f;
// //   // for (int i = 0; i < output->bytes / sizeof(float); i++) {
// //   //   Serial.print(output_data[i]);
// //   // }

// //   delay(1000); 


// //   // Calculate an x value to feed into the model. We compare the current
// //   // inference_count to the number of inferences per cycle to determine
// //   // our position within the range of possible x values the model was
// //   // trained on, and use this to calculate a value.
// //   /*float position = static_cast<float>(inference_count) /
// //                    static_cast<float>(kInferencesPerCycle);
// //   float x = position * kXrange;

// //   // Quantize the input from floating-point to integer
// //   int8_t x_quantized = x / input->params.scale + input->params.zero_point;
// //   // Place the quantized input in the model's input tensor
// //   input->data.int8[0] = x_quantized;

// //   // Run inference, and report any error
// //   TfLiteStatus invoke_status = interpreter->Invoke();
// //   if (invoke_status != kTfLiteOk) {
// //     MicroPrintf("Invoke failed on x: %f\n", static_cast<double>(x));
// //     return;
// //   }

// //   // Obtain the quantized output from model's output tensor
// //   int8_t y_quantized = output->data.int8[0];
// //   // Dequantize the output from integer to floating-point
// //   float y = (y_quantized - output->params.zero_point) * output->params.scale;

// //   // Output the results. A custom HandleOutput function can be implemented
// //   // for each supported hardware target.
// //   HandleOutput(x, y);

// //   // Increment the inference_counter, and reset it if we have reached
// //   // the total number per cycle
// //   inference_count += 1;
// //   if (inference_count >= kInferencesPerCycle) inference_count = 0;*/


